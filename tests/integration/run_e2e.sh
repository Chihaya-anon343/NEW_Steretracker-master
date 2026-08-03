#!/bin/bash
# ========================================================
# Steretracker 端到端集成测试脚本
# 运行要求: Linux 环境, 已编译 build/Steretracker
# 用法: bash tests/integration/run_e2e.sh [--verbose]
# ========================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_BIN="$PROJECT_ROOT/build/Steretracker"
CONFIGS_DIR="$SCRIPT_DIR/configs"
OUTPUT_DIR="/tmp/stereotracker_e2e_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$OUTPUT_DIR/logs_$TIMESTAMP"

PASS=0
FAIL=0
SKIP=0
VERBOSE=false

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --------------------------------------------------------
# 辅助函数
# --------------------------------------------------------

log_pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
log_fail() { echo -e "  ${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
log_skip() { echo -e "  ${YELLOW}[SKIP]${NC} $1"; SKIP=$((SKIP + 1)); }
log_info() { echo "  [INFO] $1"; }

check_prereq() {
    if [ ! -f "$BUILD_BIN" ]; then
        echo "错误: $BUILD_BIN 不存在，请先编译项目:"
        echo "  cd $PROJECT_ROOT && mkdir -p build && cd build && cmake .. && cmake --build ."
        exit 1
    fi
    if [ ! -f "$PROJECT_ROOT/best.onnx" ]; then
        echo "错误: $PROJECT_ROOT/best.onnx 不存在"
        exit 1
    fi
    echo "二进制: $BUILD_BIN"
    echo "模型:   $PROJECT_ROOT/best.onnx"
}

run_tracker() {
    local test_id="$1"
    local config_file="$2"
    local log_file="$LOG_DIR/${test_id}.log"

    mkdir -p "$LOG_DIR"

    if "$BUILD_BIN" "$config_file" > "$log_file" 2>&1; then
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "$log_file"  # 返回日志路径
        else
            echo "EXIT_FAILURE"
        fi
    else
        echo "EXIT_FAILURE"
    fi
}

gen_config_mono_debug() {
    local output="$1"
    local left_img="$2"
    local roi_x="${3:-0}"
    local roi_y="${4:-0}"
    local roi_w="${5:-0}"
    local roi_h="${6:-0}"

    cat > "$output" << EOF
{
    "mode": "debug",
    "mono_mode": true,
    "visualize": false,
    "verbose_console": true,
    "camera": {
        "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 512.0,
        "baseline_mm": 120.0
    },
    "input": {
        "left_path": "$left_img",
        "right_path": ""
    },
    "yolo": {
        "model_path": "$PROJECT_ROOT/best.onnx",
        "device_type": "CPU",
        "conf_threshold": 0.5,
        "target_class_id": 0,
        "roi_expand_ratio": 0.1,
        "roi_min_size": 50,
        "intra_op_threads": 1
    },
    "manual_roi": {
        "enabled": $([ "$roi_w" -gt 0 ] && echo "true" || echo "false"),
        "left":  {"x": $roi_x, "y": $roi_y, "width": $roi_w, "height": $roi_h},
        "right": {"x": 0, "y": 0, "width": 0, "height": 0}
    },
    "strategies": {
        "akaze_min_area": 40001,
        "tiny_max_area": 800,
        "akaze_gpnp": {
            "template_path": "data/NewMuBan(reordered)/",
            "real_w": 200.0, "real_h": 150.0,
            "scale": 0.5, "min_pts": 3,
            "use_initial_pnp": true, "mad_sigma": 3.0
        },
        "binary_corner": {
            "corners": 10, "kernel_size": 3,
            "scale": 1.0, "target_size": 100,
            "pixel_to_meter_scale": 0.5,
            "roi_pad_pixels": 0, "otsu_ratio": 1.3
        },
        "tiny_target": {
            "target_size": 50, "scale_factor": 4.0,
            "square_size_m": 0.05, "roi_pad_pixels": 0
        },
        "dual_roi": {
            "trigger_area": 490000,
            "secondary_expand_pixels": 10,
            "akaze_scale": 0.5
        },
        "close_range": {
            "enabled": true, "class1_min_area": 100,
            "roi_expand_ratio": 0.1, "min_expand_pixels": 10
        }
    }
}
EOF
}

gen_config_stereo_debug() {
    local output="$1"
    local left_img="$2"
    local right_img="$3"

    cat > "$output" << EOF
{
    "mode": "debug",
    "mono_mode": false,
    "visualize": false,
    "verbose_console": true,
    "camera": {
        "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 512.0,
        "baseline_mm": 120.0
    },
    "input": {
        "left_path": "$left_img",
        "right_path": "$right_img"
    },
    "yolo": {
        "model_path": "$PROJECT_ROOT/best.onnx",
        "device_type": "CPU",
        "conf_threshold": 0.5,
        "target_class_id": 0,
        "roi_expand_ratio": 0.1,
        "roi_min_size": 50,
        "intra_op_threads": 1
    },
    "manual_roi": {
        "enabled": false
    },
    "strategies": {
        "akaze_min_area": 40001,
        "tiny_max_area": 800,
        "akaze_gpnp": {
            "template_path": "data/NewMuBan(reordered)/",
            "real_w": 200.0, "real_h": 150.0,
            "scale": 0.5, "min_pts": 3,
            "use_initial_pnp": true, "mad_sigma": 3.0
        },
        "binary_corner": {
            "corners": 10, "kernel_size": 3,
            "scale": 1.0, "target_size": 100,
            "pixel_to_meter_scale": 0.5,
            "roi_pad_pixels": 0, "otsu_ratio": 1.3
        },
        "tiny_target": {
            "target_size": 50, "scale_factor": 4.0,
            "square_size_m": 0.05, "roi_pad_pixels": 0
        },
        "dual_roi": {
            "trigger_area": 490000,
            "secondary_expand_pixels": 10,
            "akaze_scale": 0.5
        },
        "close_range": {
            "enabled": true, "class1_min_area": 100,
            "roi_expand_ratio": 0.1, "min_expand_pixels": 10
        }
    }
}
EOF
}

gen_config_normal_dir() {
    local output="$1"
    local source_dir="$2"

    cat > "$output" << EOF
{
    "mode": "normal",
    "mono_mode": false,
    "visualize": false,
    "verbose_console": false,
    "camera": {
        "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 512.0,
        "baseline_mm": 120.0
    },
    "input_system": {
        "source_type": "directory",
        "source_dir": "$source_dir",
        "pattern": "*.png"
    },
    "yolo": {
        "model_path": "$PROJECT_ROOT/best.onnx",
        "device_type": "CPU",
        "conf_threshold": 0.5,
        "target_class_id": 0,
        "roi_expand_ratio": 0.1,
        "roi_min_size": 50,
        "intra_op_threads": 1
    },
    "strategies": {
        "akaze_min_area": 40001,
        "tiny_max_area": 800,
        "akaze_gpnp": {
            "template_path": "data/NewMuBan(reordered)/",
            "real_w": 200.0, "real_h": 150.0,
            "scale": 0.5, "min_pts": 3,
            "use_initial_pnp": true, "mad_sigma": 3.0
        },
        "binary_corner": {
            "corners": 10, "kernel_size": 3,
            "scale": 1.0, "target_size": 100,
            "pixel_to_meter_scale": 0.5,
            "roi_pad_pixels": 0, "otsu_ratio": 1.3
        },
        "tiny_target": {
            "target_size": 50, "scale_factor": 4.0,
            "square_size_m": 0.05, "roi_pad_pixels": 0
        },
        "dual_roi": {
            "trigger_area": 490000,
            "secondary_expand_pixels": 10,
            "akaze_scale": 0.5
        },
        "close_range": {
            "enabled": true, "class1_min_area": 100,
            "roi_expand_ratio": 0.1, "min_expand_pixels": 10
        }
    }
}
EOF
}

check_log() {
    local test_id="$1"
    local log_file="$2"
    local pattern="$3"
    local desc="$4"

    if grep -q "$pattern" "$log_file"; then
        log_pass "$desc"
        return 0
    else
        log_fail "$desc (pattern '$pattern' not found in $log_file)"
        if $VERBOSE; then
            echo "  ------ LOG SNIPPET ------"
            tail -20 "$log_file" | head -10
            echo "  -------------------------"
        fi
        return 1
    fi
}

# --------------------------------------------------------
# 测试用例
# --------------------------------------------------------

test_T8_01() {
    echo ""
    echo "--- T8-01: Debug 单目 (YOLO ROI) ---"
    local img="$PROJECT_ROOT/data/大图/cj01_image_0032.jpg"
    if [ ! -f "$img" ]; then
        log_skip "T8-01: 图像 $img 不存在"
        return
    fi

    local cfg="$CONFIGS_DIR/t8_01_debug_mono.json"
    gen_config_mono_debug "$cfg" "$img"
    local log_file=$(run_tracker "t8_01" "$cfg")

    if [ "$log_file" = "EXIT_FAILURE" ]; then
        log_fail "T8-01: 程序异常退出"
        return
    fi

    check_log "t8_01" "$log_file" "Strategy" "T8-01: 终端输出包含 'Strategy'"
    check_log "t8_01" "$log_file" "Frame" "T8-01: 终端输出包含 'Frame'"
}

test_T8_02() {
    echo ""
    echo "--- T8-02: Debug 单目 (手动 ROI) ---"
    local img="$PROJECT_ROOT/data/大图/cj01_image_0032.jpg"
    if [ ! -f "$img" ]; then
        log_skip "T8-02: 图像 $img 不存在"
        return
    fi

    local cfg="$CONFIGS_DIR/t8_02_debug_mono_manual_roi.json"
    gen_config_mono_debug "$cfg" "$img" 300 200 400 400
    local log_file=$(run_tracker "t8_02" "$cfg")

    if [ "$log_file" = "EXIT_FAILURE" ]; then
        log_fail "T8-02: 程序异常退出"
        return
    fi

    check_log "t8_02" "$log_file" "Strategy" "T8-02: 手动ROI模式输出包含 'Strategy'"
}

test_T8_03() {
    echo ""
    echo "--- T8-03: Debug 双目 ---"
    local left_img="$PROJECT_ROOT/data/大图/cj01_image_0032.jpg"
    local right_img="$PROJECT_ROOT/data/大图/cj01_image_0033.jpg"
    if [ ! -f "$left_img" ] || [ ! -f "$right_img" ]; then
        log_skip "T8-03: 双目图像对不存在"
        return
    fi

    local cfg="$CONFIGS_DIR/t8_03_debug_stereo.json"
    gen_config_stereo_debug "$cfg" "$left_img" "$right_img"
    local log_file=$(run_tracker "t8_03" "$cfg")

    if [ "$log_file" = "EXIT_FAILURE" ]; then
        log_fail "T8-03: 程序异常退出"
        return
    fi

    check_log "t8_03" "$log_file" "Strategy" "T8-03: 双目模式输出包含 'Strategy'"
}

test_T8_04() {
    echo ""
    echo "--- T8-04: Normal 双目 Directory ---"
    local src_dir="$PROJECT_ROOT/data/大图"
    if [ ! -d "$src_dir" ]; then
        log_skip "T8-04: 目录 $src_dir 不存在"
        return
    fi

    # 检查目录中是否有文件
    local count=$(find "$src_dir" -maxdepth 1 -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    if [ "$count" -lt 2 ]; then
        log_skip "T8-04: 目录中图像不足 ($count < 2)"
        return
    fi

    local cfg="$CONFIGS_DIR/t8_04_normal_dir.json"
    gen_config_normal_dir "$cfg" "$src_dir"
    local log_file=$(run_tracker "t8_04" "$cfg")

    if [ "$log_file" = "EXIT_FAILURE" ]; then
        log_fail "T8-04: 程序异常退出"
        return
    fi

    # Normal 模式的简洁输出应包含 Frame 信息
    if grep -q "Frame" "$log_file"; then
        log_pass "T8-04: Normal Directory 模式输出包含 'Frame'"
    else
        log_fail "T8-04: 输出不包含 'Frame'"
    fi
}

test_T8_06() {
    echo ""
    echo "--- T8-06: 配置文件缺失 ---"
    local log_file="$LOG_DIR/t8_06.log"
    if "$BUILD_BIN" "/nonexistent/config.json" > "$log_file" 2>&1; then
        log_fail "T8-06: 应返回非零退出码"
    else
        log_pass "T8-06: 非法配置路径返回非零退出码"
    fi
}

test_T8_07() {
    echo ""
    echo "--- T8-07: 非法 mode 值 ---"
    local cfg="$CONFIGS_DIR/t8_07_invalid_mode.json"
    cat > "$cfg" << 'EOF'
{
    "mode": "invalid",
    "mono_mode": false,
    "visualize": false,
    "camera": {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 512.0, "baseline_mm": 120.0},
    "input": {"left_path": "nonexistent.jpg", "right_path": "nonexistent.jpg"},
    "yolo": {"model_path": "nonexistent.onnx", "device_type": "CPU", "conf_threshold": 0.5},
    "strategies": {
        "akaze_min_area": 40001, "tiny_max_area": 800,
        "akaze_gpnp": {"template_path": "nonexistent/", "real_w": 200.0, "real_h": 150.0, "scale": 0.5, "min_pts": 3, "use_initial_pnp": true, "mad_sigma": 3.0},
        "binary_corner": {"corners": 10, "kernel_size": 3, "scale": 1.0, "target_size": 100, "pixel_to_meter_scale": 0.5, "roi_pad_pixels": 0, "otsu_ratio": 1.3},
        "tiny_target": {"target_size": 50, "scale_factor": 4.0, "square_size_m": 0.05, "roi_pad_pixels": 0},
        "dual_roi": {"trigger_area": 490000, "secondary_expand_pixels": 10, "akaze_scale": 0.5},
        "close_range": {"enabled": true, "class1_min_area": 100, "roi_expand_ratio": 0.1, "min_expand_pixels": 10}
    }
}
EOF

    mkdir -p "$LOG_DIR"
    if "$BUILD_BIN" "$cfg" > "$LOG_DIR/t8_07.log" 2>&1; then
        log_fail "T8-07: 非法 mode 应返回非零退出码或报错"
    else
        log_pass "T8-07: 非法 mode 被正确拒绝"
    fi
}

# --------------------------------------------------------
# 主入口
# --------------------------------------------------------

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verbose|-v) VERBOSE=true; shift ;;
            *) shift ;;
        esac
    done

    echo "============================================================"
    echo "  Steretracker 端到端集成测试"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  日志: $LOG_DIR"
    echo "============================================================"

    mkdir -p "$CONFIGS_DIR"
    rm -f "$CONFIGS_DIR"/*.json

    check_prereq

    # 运行测试
    test_T8_01
    test_T8_02
    test_T8_03
    test_T8_04
    test_T8_06
    test_T8_07

    # 清理临时配置
    rm -f "$CONFIGS_DIR"/*.json

    # 汇总
    echo ""
    echo "============================================================"
    echo -e "  结果: ${GREEN}通过 $PASS${NC} | ${RED}失败 $FAIL${NC} | ${YELLOW}跳过 $SKIP${NC}"
    echo "============================================================"

    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
}

main "$@"