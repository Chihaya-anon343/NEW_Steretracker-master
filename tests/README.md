# Steretracker 测试方案

本目录包含 Steretracker（GPNP 双目视觉跟踪器）的完整功能测试方案与测试基础设施。

## 目录结构

```
tests/
├── README.md                   # 本文件 — 测试方案总览与执行入口
├── docs/
│   └── TEST_PLAN.md            # 完整测试方案（测试域、用例矩阵、优先级、通过准则）
├── test_data/
│   └── synthetic/              # 合成测试图像（由生成脚本预生成）
│       └── generate.py         # Python+OpenCV 合成图像生成脚本
├── unit/                       # 纯算法层单元测试（不依赖 ONNX Runtime）
│   ├── CMakeLists.txt          # 独立编译配置
│   ├── pose/                   # 位姿有效性测试用例
│   └── strategy/               # 策略链退化测试用例
├── test_utils/                 # 测试辅助工具（断言宏）
└── integration/                # 端到端集成测试脚本（Linux / Docker 环境）
    └── run_e2e.sh              # 批量运行主程序 + 验证输出
```

## 测试环境要求

| 组件 | 最小版本 | 备注 |
|------|---------|------|
| C++ | 17 | 编译单元测试 |
| OpenCV | 4.x | `core, imgproc, features2d, calib3d, highgui` |
| Eigen | 3.x | 线性代数 |
| ONNX Runtime | 1.20+ (Linux) | 仅集成测试需要 |
| CMake | 3.22+ | 构建工具 |
| Python | 3.8+ | 合成图像生成（可选） |

## 快速开始

### 1. 生成合成测试图像（可选）

```bash
cd tests/test_data/synthetic
python generate.py
```

### 2. 编译并运行单元测试（Windows / Linux 均支持）

单元测试仅依赖 OpenCV + Eigen，不依赖 ONNX Runtime，可在 Windows 下直接编译：

```bash
cd tests/unit
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./test_stereotracker    # Linux
# 或 test_stereotracker.exe   # Windows
```

### 3. 运行端到端集成测试（仅 Linux / Docker）

```bash
# 在项目根目录执行
bash tests/integration/run_e2e.sh
```

## 测试范围速览

| 编号 | 测试域 | 类型 | 环境 | 优先级 |
|------|--------|------|------|--------|
| T1 | 配置与工厂函数 | 单元测试 | Win/Linux | P0 |
| T2 | 五状态分级（RoiGenerator） | 单元测试 | Win/Linux | P0 |
| T3 | 特征提取器（TinyTarget/BC/AKAZE） | 单元测试 | Win/Linux | P0 |
| T4 | 位姿解算（GPnP/InitialPnP/MonoPnP） | 单元测试 | Win/Linux | P0 |
| T5 | 退化链（策略降级） | 单元测试 | Win/Linux | P1 |
| T6 | 退化全景（13类退化路径） | 单元测试 | Win/Linux | P1 |
| T7 | 输入系统（File/Directory/Sequence） | 单元测试 | Win/Linux | P1 |
| T8 | 端到端集成（完整流水线） | 集成测试 | Linux/Docker | P0 |
| T9 | 性能基线 | 基准测试 | Win/Linux | P2 |

## 测试设计原则

1. **合成图像优先** — 使用已知几何的图像验证位姿解算正确性（真实图像缺乏 ground truth）
2. **自动化判定** — 每个用例定义输入、预期输出（含数值容差）、判定标准
3. **不修改主项目源码** — 单元测试通过引用 include 头文件 + 链接源码编译
4. **分层隔离** — 单元测试不依赖 ONNX Runtime，可在任意平台运行
5. **优先级驱动** — P0 必测（影响核心功能）、P1 应测（影响鲁棒性）、P2 可测（性能基线）

## 相关文档

- 主项目 README：`../README.md`
- AI Agent 说明书：`../CLAUDE.md`
- 完整测试方案：`docs/TEST_PLAN.md`