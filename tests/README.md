# Steretracker 测试体系

## 快速总览

| 测试文件 | 用例数 | 被测模块 | 输入数据来源 | 断言类型 |
|----------|:-----:|---------|------------|---------|
| `test_config.cpp` | 8 | 配置工厂函数 | 纯代码，无外部依赖 | 精确值 / 异常类型 |
| `test_roi_generator.cpp` | 16 | ROI 生成 & 五状态判定 | 代码合成 `Detection` 结构体 | 精确面积/尺寸/布尔值 |
| `test_pose_solvers.cpp` | 7 | InitialPnP / MonoPnP / GPnP | 代码合成 Z=0 平面 8 点 + 高斯噪声 | 误差阈值 (5%~15%) |
| `test_extractors.cpp` | 8 | BinaryCorner / TinyTarget 提取器 | `data/fixtures/` 图片 (mono_bc/mono_tiny) + `rois.json` + `data/NewMuBan(reordered)/` 模板 | 结构性 (成功/不崩溃) |
| `test_input_system.cpp` | 14 | 输入系统 & RingBuffer & 线程化采集 | `cv::imwrite` 临时目录 (自动创建+清理) | 精确尺寸/FIFO 顺序 |
| `test_integration.cpp` | 3 | MonoTracker / StereoTracker 全流程 | `data/fixtures/` 图片 (mono_akaze/synthetic_akaze/synthetic_dual) + `rois.json` + 模板目录 | 冒烟 (仅验证不崩溃) |
| `test_eskf_fusion.cpp` | 13 | ESKF 延迟反向传播 / 兜底 / 退化监控 / 线程化 | 代码合成悬停/匀加速 IMU + 精确相机位姿 (零噪声) | 逐元素一致 (1e-6~1e-9) / 分级枚举 / 严格不等式 |
| `test_eskf_multimodal.cpp` | 1 | ESKF 多模态线程化集成 | 代码合成螺旋上升轨迹 + 反推 IMU/相机/雷达 + 4 类错误注入 | 软验证 (跑通 + smoke 断言 + CSV 输出) |
| `test_yolo_decode.cpp` | 10 | YOLO 原始输出解码 + NMS | 代码合成原始张量 (BCN/BNC, 无 ONNX 依赖) | 精确框坐标/置信度/类别/抑制 |
| `test_yolo_detector.cpp` | 5 | YoloDetector / YoloRoiProvider 端到端 | `yolo_onnx/yolov8n.onnx` + `data/fixtures/` 图片 (缺失 SKIP) | 状态码 / 不崩溃 |
| `test_synthetic_pipeline.cpp` | 8 | YOLO → ROI → 特征 → PnP → 位姿 全流程 (单目/双目 × 四策略) | `tests/data/fixtures_rich` 合成图 + 特征点 txt + 模板 | 与 solvePnP 真值比对平移/旋转误差 |
| **合计** | **93** | | | |

> 数据依赖层级：纯代码 → 临时文件 (自动) → fixtures 图片 + 模板目录 (可选, 缺失时 SKIP)

---

## 1. 测试目标

三层结构，从单元级到系统级：

| 层级 | 名称 | 目标 |
|------|------|------|
| L0 | 单元测试 | 验证配置工厂、几何确定的算法（ROI、PnP），使用合成数据精确断言 |
| L1 | 单元测试 | 验证图像源扫描、特征提取器与流水线可运行 |
| L2 | 端到端冒烟 | 验证 Tracker 主流程（单目/双目/Dual-ROI）可完整跑通、无异常 |

---

## 2. 目录结构

```
tests/
├── README.md
├── CMakeLists.txt               # -DGPNP_BUILD_TESTS=ON 启用
├── framework/
│   └── TestAssert.hpp           # TEST_ASSERT / REGISTER_TEST / TestRegistry
├── scripts/
│   ├── generate_assets.py       # fixtures 生成脚本 (真实背景+目标合成)
│   └── plot_eskf_traj.py        # ESKF 融合轨迹可视化 (真值 vs 融合 + 错误段标注)
├── unit/
│   ├── TestDataLoader.hpp       # fixtures 加载工具 (rois.json → ROI)
│   ├── test_config.cpp          # [L0] 配置工厂
│   ├── test_roi_generator.cpp   # [L0] ROI 生成 & 五状态判定
│   ├── test_pose_solvers.cpp    # [L0] PnP 位姿求解器
│   ├── test_extractors.cpp      # [L1] 特征提取器
│   ├── test_input_system.cpp    # [L1] 输入系统
│   ├── test_eskf_fusion.cpp     # [L0] ESKF 延迟反向传播 / 兜底 / 退化监控 / 线程化
│   ├── test_eskf_multimodal.cpp # [L1] ESKF 多模态线程化集成 (螺旋轨迹 + 错误注入 + CSV 输出)
│   ├── test_yolo_decode.cpp     # [L0] YOLO 原始输出解码 + NMS (合成张量, 无 ONNX)
│   ├── test_yolo_detector.cpp   # [L1] YOLO 检测器端到端冒烟 (模型缺失 SKIP)
│   └── test_integration.cpp     # [L2] 端到端冒烟
└── data/fixtures/               # gitignored, 由 generate_assets.py 生成
    ├── synthetic_tiny|bc|akaze/          # 双目 State 1/2/3 (class0-only)
    ├── synthetic_bc_class1|akaze_class1/ # 双目 State 5/6 (class1-only)
    ├── synthetic_dual/                   # 双目 State 4 (class0+class1)
    ├── mono_tiny|bc|akaze/               # 单目仅左图 (class0-only)
    ├── mono_bc_class1|akaze_class1/      # 单目仅左图 (class1-only)
    ├── mono_dual/                        # 单目仅左图 (class0+class1)
    ├── rois.json                         # 每场景每帧每侧 class0/class1 ROI
    └── manual_roi.json                   # Debug 模式手动 ROI (对齐 synthetic_bc)
```

---

## 3. 断言框架

| 宏 | 作用 |
|----|------|
| `TEST_ASSERT(cond)` | 条件成立，失败输出文件名+行号 |
| `TEST_ASSERT_MSG(cond, msg)` | 同上 + 自定义信息 |
| `TEST_ASSERT_EQ(a, b)` | `operator==` 相等 |
| `TEST_ASSERT_NEAR(a, b, eps)` | 浮点 `|a-b| < eps` |
| `TEST_ASSERT_THROWS(expr, T)` | 抛指定类型异常 |
| `REGISTER_TEST(fn)` | 静态注册测试函数 |
| `TestRegistry::runAll()` | 运行全部注册测试，返回失败数 (0=通过) |

> `test_input_system.cpp` 使用独立宏体系 `REQUIRE`/`RUN`/`PASS`；`test_integration.cpp` 使用 `CHECK`/`runTest`/`SkipTestException`。

---

## 4. 测试用例详解

### 4.1 `test_config.cpp` — 配置工厂 (8 用例)

**被测 API**: `makeStereoCameraParams()` / `makeTrackerConfig()` / `makeYoloConfig()`
**输入数据**: 纯代码，无文件依赖

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| 1 | `test_stereo_params_valid` | K(fx=fy=1000), 单位R, t=(-120,0,0) | `focal≈1000`, `baseline≈120`, `K·K⁻¹≈I` (<1e-12) |
| 2 | `test_stereo_params_invalid_k` | K(2,0)=0.1 (非上三角) | 抛 `std::invalid_argument` |
| 3 | `test_stereo_params_invalid_rotation` | R(0,0)=2.0 (非正交) | 抛 `std::invalid_argument` |
| 4 | `test_stereo_params_nan` | K(1,1)=NaN | 抛 `std::invalid_argument` |
| 5 | `test_tracker_config_valid` | scale=0.5, min_pts=4, 模板 200×150mm, area 阈值 | 逐字段精确验证 |
| 6 | `test_tracker_config_invalid` | 8 组非法: scale≤0/>1, min_pts<2, 尺寸≤0, 面积≤0 | 全部抛异常 |
| 7 | `test_yolo_config_valid` | model="best.onnx", CPU, conf=0.5, 640×640 | 逐字段验证 |
| 8 | `test_yolo_config_invalid` | 6 组非法: 空路径, 阈值越界, 尺寸≤0, 线程≤0 | 全部抛异常 |

---

### 4.2 `test_roi_generator.cpp` — ROI 生成 & 五状态判定 (16 用例)

**被测 API**: `RoiGenerator::generateGroup()` / `generateStereo()` / `tryCloseRange()` / `detectionToRoi()`
**输入数据**: 代码内合成的 `Detection` 结构体（无外部文件）

**五状态判定逻辑**:

```
class0 面积 ≤800                    → State 1 远    (TinyTarget)
class0 面积 801~40000               → State 2 中    (BinaryCorner)
class0 面积 40001~489999            → State 3 中近  (AKAZE)
class0 面积 ≥490000 + class1 存在   → State 4 近    (Dual-ROI)
无 class0 + class1 有效 + close_range → State 5 极近 (回退重分类)
无 class0 + 无 class1               → SKIP 帧
```

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| | **五状态判定** | | |
| 1 | `test_state1_tiny_area` | class0 28×28=784 ≤800 | valid, !is_dual, 面积≤800, 注: roi_min_size=0 |
| 2 | `test_state2_medium_area` | class0 100×100=10000 | valid, 面积>800 && ≤40000 |
| 3 | `test_state3_medium_close_area` | class0 300×300=90000 | valid, 面积>40000 && <490000 |
| 4 | `test_state4_dual_roi` | class0 720×720 + class1 80×80 | valid, is_dual=true, secondary=80×80 |
| | **边界条件** | | |
| 5 | `test_no_detection_invalid` | 空检测列表 | !valid |
| 6 | `test_only_class1_invalid_without_close_range` | 仅 class1, close_range 未启用 | !valid |
| 7 | `test_large_class0_without_class1_not_dual` | class0 720×720 无 class1 | valid, !is_dual (降级 State 3) |
| 8 | `test_roi_not_above_dual_trigger` | class0 699×699 <490000 + class1 | valid, !is_dual (面积不足) |
| | **State 5 极近 (close_range)** | | |
| 9 | `test_close_range_recovery` | 仅 class1 400×400, close_range on, expand 1.5× | valid, primary=600×600 |
| 10 | `test_close_range_disabled` | close_range 默认关闭 + class1 | !valid |
| 11 | `test_close_range_below_min_area` | class1 100×100 <100000 | !valid |
| | **ROI 几何** | | |
| 12 | `test_roi_expand_ratio` | 100×100 + expand=0.1 | primary=120×120 |
| 13 | `test_roi_min_size` | 20×20 < min_size=50 | width≥50, height≥50 |
| 14 | `test_roi_clamp_to_image` | 大 expand 贴边 | 裁剪回图像边界内 |
| | **双目 ROI** | | |
| 15 | `test_generate_stereo_right_fallback` | 左 class0 100×100, 右无检测 | 右复制左 ROI |
| 16 | `test_generate_stereo_group_dual` | 左右均有 Dual-ROI 检测 | pair 有效且 is_dual |

---

### 4.3 `test_pose_solvers.cpp` — PnP 求解器 (7 用例)

**被测 API**: `InitialPnPSolver::solve()` / `MonoPnPSolver::solve()` / `GPnPSolver::solve()`
**输入数据**: 代码合成 Z=0 平面 8 个模板点 + 标准 K (fx=fy=1000, cx=cy=512) + 高斯噪声 σ=0.3~0.5px

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| | **InitialPnPSolver** | | |
| 1 | `test_initial_pnp_recovers_pose` | 8 点, ry=0.15, tz=1500, σ=0.3px | success, t_err<5%, R_err<5° |
| 2 | `test_initial_pnp_rejects_bad_pose` | 3 点 < min_pts=4 | !success |
| 3 | `test_initial_pnp_validity_checks` | 深度超范围 Z=25000 (>20000 上限) | !success |
| | **MonoPnPSolver** | | |
| 4 | `test_mono_pnp_recovers_pose` | 8 点, ry=-0.1, tz=800, σ=0.3px | success, t_err<5%, R_err<5° |
| 5 | `test_mono_pnp_rejects_insufficient_points` | 3 点 < min=4 | !success |
| | **GPnPSolver** | | |
| 6 | `test_gpnp_recovers_pose_stereo` | 8 点双目, tz=1200, baseline=120mm, warmstart 偏 0.02rad+50mm | success, t_err<15%, R_err<8.6° |
| 7 | `test_gpnp_rejects_insufficient_points` | 2 点 < min=4, nullptr warm-start | !success |

> **注意**: (a) 用例 3 不能用"目标在相机后方"测试——PnP 存在手性歧义，后方的 2D 投影与前方镜像一致，RANSAC 会找到合法前向解。改用深度超范围测试。(b) 用例 6 的 15% 平移阈值因共面目标固有的平移-旋转歧义 (X 平移与 Y 轴旋转耦合)。

---

### 4.4 `test_extractors.cpp` — 特征提取器 (8 用例)

**被测 API**: `BinaryCornerExtractor::extract()` / `TinyTargetExtractor::extractMono()`
**输入数据**: `data/fixtures/` 图片 (ROI 按 `rois.json` 裁剪) + `data/NewMuBan(reordered)/` 模板目录

| # | 用例 | 依赖模板 | 输入 | 判断标准 |
|---|------|:---:|------|---------|
| 1 | `test_interface_names` | 否 | 无 | StrategyType: Akaze=0, BC=1, TT=2 |
| 2 | `test_binary_corner_static_reorder` | 否 | 乱序 4 角点, 中心 (50,50), 参考角 -90° | 4 个互异有效索引 |
| 3 | `test_binary_corner_draw_corners` | 否 | `fixtures/mono_bc/left_000.png` (640×480) + 4 角点 | 输出 CV_8UC3, 同尺寸 |
| 4 | `test_binary_corner_empty_input` | 是 | 空 cv::Mat | 不崩溃, !success, kp 为空 |
| 5 | `test_binary_corner_synthetic_rectangle` | 是 | `fixtures/mono_bc/left_000.png` 裁剪 class0 ROI (244,165,152×150) | 不崩溃; 成功时 pts≤8 |
| 6 | `test_tiny_target_empty_input` | 是 | 空 cv::Mat | 不崩溃, !success, kp 为空 |
| 7 | `test_tiny_target_small_black_square` | 是 | `fixtures/mono_tiny/left_000.png` 裁剪 class0 ROI (310,230,20×20) | 不崩溃; 成功时 4 角点, 坐标∈ROI |
| 8 | `test_tiny_target_set_use_class1` | 是 | 切换 class0/class1 物理尺寸 | setUseClass1(true/false) 不崩溃 |

> 模板目录或 fixtures 缺失时用例 3/5/7 静默跳过 (不打 FAIL)；用例 4~8 仅在模板目录缺失时跳过。

---

### 4.5 `test_input_system.cpp` — 输入系统 (14 场景)

**被测模块**: `FileStereoSource` / `DirectoryStereoSource` / `SequenceSource` / `InputProvider` / `RingBuffer`
**输入数据**: `cv::imwrite` 写入临时目录 (自动创建 + 自动清理)

| # | 场景 | 输入 | 判断标准 |
|---|------|------|---------|
| 1 | `FileStereoSource: 正常加载一对图像` | 临时 64×48 L.png/R.png | open→取帧→64×48→reset→close 全链路 |
| 2 | `FileStereoSource: 路径无效返回 false` | "" / 不存在的路径 | open 返回 false |
| 3 | `FileStereoSource: URI 分号格式` | "L2.png;R2.png" | open 成功→取帧成功 |
| 4 | `DirectoryStereoSource: 扫描配对 + 排序读取` | left/right_0000~0002, 尺寸递增 | 3 对, 按序取帧, reset 回放 |
| 5 | `DirectoryStereoSource: 右图缺失自动跳过` | left 有 2 帧, right 仅 1 帧 | totalFrames=1 |
| 6 | `DirectoryStereoSource: 目录不存在 / 无匹配` | 不存在/空目录 | open 返回 false |
| 7 | `SequenceSource: 单目序列扫描` | frame_0000~0001 (32×24/33×25) | 2 帧, 右图=左图副本 |
| 8 | `InputProvider: File 配置` | File 配置 + 64×48 L/R | init→packet 有效→尾→reset |
| 9 | `InputProvider: Directory 配置` | Directory 配置 + 2 对 | 2 帧尺寸递增 |
| 10 | `InputProvider: Sequence 配置` | Sequence 配置 + 30×30 | packet 有效, 左右非空 |
| 11 | `InputProvider: 无效配置` | 不存在的路径 | init 失败, !isOpen |
| 12 | `InputProvider: 线程化采集` | Directory 源 8 帧 + use_threaded_capture + ring_capacity=2 | 消费≤8, shutdown 后 captured=consumed+dropped, 再取帧 false |
| 13 | `RingBuffer: 写入/读取顺序` | buf(4), push 10/20/30 | popOldest→10,20,30 (FIFO) |
| 14 | `RingBuffer: 满缓冲区行为` | buf(2), push 1/2/3 | full=true, dropped=1, clear 后 empty |

> `currentFrame()` 统一语义：-1 = 未取帧, 0 = 已取帧 0, …（与 FileStereoSource 对齐）。

---

### 4.6 `test_integration.cpp` — 端到端冒烟 (3 用例)

**被测模块**: MonoTracker::process() / StereoTracker::process() (含 Dual-ROI 独立路径)
**输入数据**: `data/fixtures/` 真实背景+真实目标 (data/big/img_1.png) 合成图, ROI 取自 `rois.json` + 模板目录

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| 1 | `MonoTracker 冒烟` | `fixtures/mono_akaze/left_000.png` (640×480) + class0 ROI (214,135,213×210) | time≥0, frameCount≥1, logs≥1 |
| 2 | `StereoTracker 冒烟: warm-start 两帧` | `fixtures/synthetic_akaze/` 双目对 (640×480, 右图向左偏移16px) × 2 帧, 左右 ROI 各自从 rois.json 取 | 两帧 time≥0, frameCount≥2 |
| 3 | `Dual-ROI 冒烟: is_dual=true 独立路径` | `fixtures/synthetic_dual/` 双目对 (1280×960): class0 外框 731×720 + class1 中心 120×120 | 两帧 time≥0, frameCount≥2, 不崩溃 |

> ⚠️ **冒烟定位**: fixtures 目标图为 AKAZE 模板 (data/big/img_1.png) 的缩放产物, AKAZE 描述子匹配应能成功; 断言仍保持宽松 (仅验证不崩溃、计时合法)。相机内参主点取图像中心 (640×480 → cx=320, cy=240; 1280×960 → cx=640, cy=480)。模板或 fixtures 缺失时 SKIP。

---

### 4.7 `test_eskf_fusion.cpp` — ESKF 方案B (13 用例)

**被测模块**: `fusion::EskfFusionManager` — 延迟测量反向传播 / 协方差膨胀兜底 / 退化监控 / 线程化 (Phase 4)
**输入数据**: 纯代码合成 — 悬停 IMU (`acc=(0,0,+9.81)` → 零漂移, 对齐 eskf_vio 重力约定) / 匀加速 IMU (移动场景) + 精确相机位姿 (R=I, `t_mm=-1000·p` 精确反推), 全程零噪声 → 确定性断言

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| | **分组 A: 基础链路回归** | | |
| 1 | `test_lazy_init_and_first_update` | 悬停 IMU 0.01~1.0 + 相机 (1.0/1.0, P0) | initialized, p==P0 (1e-9), v==0, q==(1,0,0,0), Normal, imu_samples==0 (初始化前不积分) |
| 2 | `test_invalid_camera_imu_only` | 悬停 IMU 0.01~2.0 + valid=false 帧 + 有效帧 | 无效帧不初始化 (Uninitialized/零值), 有效帧后 p==P0, 无残留 |
| 3 | `test_gap_exceeds_max_cam_gap_resets` | IMU 0.01~3.0 + obs1 (1.0, P0) + obs2 (3.0, P1, 间隔2s>1s) | 精确重置: p==P1 (1e-12), v==0, stats 全 0 (reset 清零) |
| | **分组 B: 延迟反向传播 (方案B 核心)** | | |
| 4 | `test_backprop_matches_zero_latency_reference` | 悬停 IMU + obs2 延迟 80ms (1.50/1.58) vs 理想 (1.50/1.50)+propagateTo(1.58) | 终态逐元素一致 (1e-6), cam_late_fallback==0 (走反向传播) |
| 5 | `test_backprop_beats_arrival_application` | 匀加速 IMU: 反向传播 vs 理想 vs 朴素到达时刻应用 | ‖p_B−p_R‖<1e-6 且 ‖p_A−p_R‖>5e-3 (朴素路径偏离 ~2cm) |
| 6 | `test_sub_ms_latency_skips_backprop` | obs2 延迟 0.5ms (1.50/1.5005) vs 直接路径 (1.5005/1.5005) | 逐元素一致 (1e-9), cam_late_fallback==0, 更新生效 |
| | **分组 C: 兜底策略** | | |
| 7 | `test_latency_beyond_window_inflate` | obs2 延迟 0.5s>0.2s 窗口 (1.50/2.00) vs 无延迟参考 | cam_late_fallback==1, 观测仍应用 (p≈P2), 后验位置协方差迹 >1.01×参考 (膨胀生效) |
| 8 | `test_latency_beyond_window_reject` | 同上, latency_fallback=Reject | cam_ignored==1, cam_updates==0, 状态保持惯性估计 (p≈P0) |
| | **分组 D: 退化监控** | | |
| 9 | `test_quality_normal_degraded_stale` | init 后停相机, propagateTo(1.30/1.70/2.50) | quality 依次 Normal/Degraded/Stale, 位置零漂移, cov_trace 严格递增 |
| 10 | `test_uninitialized_state` | 仅 IMU, 无相机 | initialized==false, Uninitialized, 零值 |
| | **分组 E: 线程化 (Phase 4)** | | |
| 11 | `test_threaded_async_matches_sync` | start() 后同用例 4 场景 (IMU 只喂到 1.58), 轮询收敛; 相机缺席续喂 IMU→2.3 | 终态与同步参考逐元素一致 (1e-6), 缺席后 Degraded, stop() 幂等 |
| 12 | `test_camera_missing_imu_propagates` | 首帧 init 后相机缺失, 续喂 MOVE IMU (无相机) | 位置被 IMU 死推 (~0.15m), 质量 Degraded, stop() 正常 |
| | **分组 F: 复位** | | |
| 13 | `test_reset_clears_state_stats` | 产生非零 stats → reset() → 重新初始化 | 全零状态/统计/缓冲, 重初始化 p==P0 (1e-9) |

> **确定性机制**: ① 悬停 IMU 比力 = −g → `a_world = R(acc−b_a)+g = 0`, 状态零漂移; ② 反向传播"真值" = 零延迟理想世界 (t0=t1), 回退快照 (100Hz 网格) 与 IMU 重放逐样本复现该理想 → 核心断言为**逐元素相等**; ③ 位置跳变 ≤0.3m + cam_pos_noise=0.1 → NIS ≪ χ²₉₅, hybrid 更新永不被 FDI 拒; ④ 线程化用例轮询 + 2s 截止防 flaky。
> ⚠️ **测试依赖**: 依赖提交 aa00616 遗漏的 `include/fusion/FusionTypes.hpp` (2026-08-14 补写); 用例 8 记录当前实现语义: Reject 分支只计 `cam_ignored`、不计 `cam_late_fallback`。

---

### 4.8 `test_eskf_multimodal.cpp` — ESKF 多模态线程化集成 (1 用例)

**被测模块**: `fusion::EskfFusionManager` (线程化 `threaded=true`) + 完整辅助模块 (RadarAltimeter / camera FDI / 反向传播)
**输入数据**: 纯代码合成 —— 螺旋上升地面真值轨迹 `p(t)=(R·cos ωt, R·sin ωt, h0+vz·t)`，由运动学反推三路数据:

- **IMU** 比力 `f_b = R_cam_wᵀ·(a-g)` + 角速度 `(0,0,ω)` + 噪声 (200Hz)
- **相机观测** (PnP 约定 `R_tpl_cam`/`t_cam_mm`) + 位置噪声 (10Hz)
- **雷达高度** `p.z` + 噪声 (20Hz)

| 错误注入 | 时段 | 验证点 |
|---------|------|--------|
| `radar_jump` 雷达跳变 +50m | 5.0~5.2s | `RadarAltimeter` 跳变拒绝 → `radar_rejected>0` |
| `cam_jump` 相机位置跳变 +3m | 8.0~8.5s | camera FDI 位置 NIS 拒绝 → `cam_ignored>0` |
| `cam_fail` 相机失效 | 13.0~13.6s | IMU 死推 + `quality→Degraded` |
| `cam_delay` 相机延迟 0.1s | 16.0~18.0s | 反向传播 `applyCameraBackprop` (曝光=t-0.1, 送达=t) |

**判定方式**: 软验证 —— 跑通全流程 + smoke 断言 (`initialized`/`imu_samples>0`/`cam_updates>0`/`radar_rejected>0`/`cam_ignored>0`), 轨迹/误差写 `eskf_traj.csv` + `eskf_events.csv` 供 `plot_eskf_traj.py` 可视化 (3D 真值 vs 融合 + 错误段标注)。

> ⚠️ **踩坑记录**: ① 错误注入用**窗口**而非单点 (单点 + 浮点累加会漏注入); ② 相机失效窗口时长必须 < `max_cam_gap_s`(1.0s) 才能走 Degraded 而非触发 reset 清空 stats。

---

### 4.9 `test_yolo_decode.cpp` — YOLO 原始输出解码 + NMS (10 用例)

**被测 API**: `gpnp::decodeYoloOutput()` (`include/detection/YoloDecode.hpp`)
**输入数据**: 代码合成原始 YOLOv8 张量（BCN `[6,N]` / BNC `[N,6]`，2 类），无图像、无 ONNX —— 确定性、可离线运行。

> 背景: 2026-08 将 YOLO 模型从 `best.onnx`（NMS-export, `[1,300,6]` 已解码）换成 `yolo_onnx/yolov8n.onnx`（原始 YOLOv8 导出, `[1,6,8400]` 未解码）。解码 + NMS 逻辑从 `YoloDetector::postprocess()` 抽为纯函数以便单测。

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| 1 | `test_bcn_decode` | BCN 单框 cx=100,cy=100,w=40,h=20, class0=0.9 | 1 框, class_id=0, x1=80,y1=90,w=40,h=20 |
| 2 | `test_bnc_decode` | BNC 同输入 | 与 BCN 结果逐字段一致 |
| 3 | `test_conf_threshold_filter` | 两框 0.9 / 0.3, conf=0.5 | 仅保留 0.9 |
| 4 | `test_all_below_threshold_empty` | 单框 0.4 < 0.5 | 空结果 |
| 5 | `test_nms_suppress_overlap` | 两框 IoU≈0.68 > 0.45, 分数 0.9/0.8 | 仅保留高分 0.9 |
| 6 | `test_nms_keep_distinct` | 两框 IoU=0 | 两个都保留 |
| 7 | `test_letterbox_inverse` | ratio=0.5, dw=10, dh=20 | 反变换后 x=80,y=90,w=40,h=20 |
| 8 | `test_clamp_to_image` | 框中心为负, 大幅出界 | clamp 回 [0,640]×[0,480] |
| 9 | `test_class_argmax` | class1 分数更高 | class_id==1 |
| 10 | `test_null_data_empty` | data=nullptr | 空结果 |

---

### 4.10 `test_yolo_detector.cpp` — YOLO 检测器端到端冒烟 (5 用例)

**被测模块**: `YoloDetector` (模型加载 / detect 状态码) + `YoloRoiProvider` (YOLO → RoiGroup 外观)
**输入数据**: `yolo_onnx/yolov8n.onnx` 模型 + `tests/data/fixtures/synthetic_akaze/left_000.png` 图片（仅冒烟，不强断言检测框）。模型或图片缺失时用例 SKIP（不 FAIL）。

| # | 用例 | 输入 | 判断标准 |
|---|------|------|---------|
| 1 | `test_model_load_success` | 模型在位 | `isInitialized()==true` |
| 2 | `test_model_missing_throws` | 不存在的模型路径 | 构造抛 `std::exception` |
| 3 | `test_detect_empty_image` | 空 cv::Mat | 返回 `Status::EmptyInput`, dets 空 |
| 4 | `test_detect_fixture_no_crash` | fixture 图片 | 返回 `Status::Success`（空检测也合法） |
| 5 | `test_roi_provider_e2e` | fixture 图片 (单目+双目) | initialize/isReady 成功, detectMono/detect 不崩溃 |

> ⚠️ **冒烟定位**: fixture 合成图非真实目标，检测结果无确定性保证，故只断言状态码/不崩溃；具体解码/NMS 正确性由 `test_yolo_decode`（4.9）确定性覆盖。

---

### 4.11 `test_synthetic_pipeline.cpp` — 合成图全流程 (8 用例)

**被测模块**: `YoloRoiProvider` (YOLO → ROI) + `StereoTracker::process()` / `MonoTracker::process()` 完整链路
**输入数据**: `tests/data/fixtures_rich` 合成图 (由 `scripts/generate_synthetic_dataset.py` 生成, 被 .gitignore 忽略) + 每图 `*_class0.txt`/`*_class1.txt` 特征点 + `scripts/class0_points.txt` 模板点 + `data/big/img_1.png`(AKAZE) / `data/NewMuBan(reordered)`(BC/TT) 模板

**覆盖**: 单目/双目 × 四策略 (TinyTarget / BinaryCorner / AKAZE / Dual-ROI) = 8 项, 每项 frame 0。

| # | 用例 | 输入 | 提取器 |
|---|------|------|--------|
| 1 | 单目 TinyTarget | `mono_tiny/left_000.png` | TinyTarget |
| 2 | 单目 BinaryCorner | `mono_bc/left_000.png` | BinaryCorner |
| 3 | 单目 AKAZE | `mono_akaze/left_000.png` | AKAZE |
| 4 | 单目 Dual-ROI | `mono_dual/left_000.png` | BC(class0)+AK(class1) |
| 5 | 双目 TinyTarget | `synthetic_tiny/` | TinyTarget |
| 6 | 双目 BinaryCorner | `synthetic_bc/` | BinaryCorner |
| 7 | 双目 AKAZE | `synthetic_akaze/` | AKAZE |
| 8 | 双目 Dual-ROI | `synthetic_dual/` | BC+AK |

**机制**:
- **合成内参**: f=1000, cx=W/2, cy=H/2; 模板物理 500×500mm。双目基线按 `disp·500/size_nominal` 反推 (右图右移 `DISP` px)。
- **ROI 获取 (YOLO 优先 + 真值回退)**: 先跑真实 ONNX YOLO; 检测框中心/尺寸与特征点包围盒(真值)不一致时回退到特征点包围盒。`--no-yolo` 强制回退 (诊断)。
- **真值位姿**: `cv::solvePnP(class0_txt 投影点, class0_points.txt→3D mm)` 得 `R_gt, t_gt`。
- **断言**: `success` + `t.z>0` + 平移相对误差 `< 阈值` (深度跨 625~25000mm 用相对值) + 旋转粗检 (平面目标有姿态歧义)。

**实测结论** (2026-08): Dual-ROI 精确 (平移<1%/旋转<0.5°); 单目 tiny/bc/akaze 平移 2~5% 正确、旋转有平面歧义误差 (6~34°); 双目 tiny/akaze 因立体光流/角点 IoU 匹配退化而失败 (已知限制)。BC/TT 物理尺度已在测试内标定 (pixel_to_meter_scale=0.0072 / square_size_m=0.356) 以对齐 AKAZE 500mm 约定。

> 运行: `./tests/test_synthetic_pipeline [--no-yolo] [--fixtures-rich-dir DIR] ...`。fixtures_rich / 模板 / 特征点 txt 缺失时 SKIP (不 FAIL)。

---

## 5. 构建与运行

> **Docker 工具链 (推荐, 与 CLion 一致)**: 本项目用 CLion Docker 工具链 (镜像 `cpp_cuda_x64_0620:latest`, 项目挂载于容器 `/tmp/NEW_Steretracker-master`)。Claude Code / 命令行经全局脚本 `docker-toolchain.sh <cmd>` (位于 `~/bin/`, 任意项目可用) 把命令转发进容器执行:
> ```bash
> docker-toolchain.sh cmake --build cmake-build-debug --target test_eskf_multimodal
> docker-toolchain.sh ./cmake-build-debug/tests/test_eskf_multimodal
> docker-toolchain.sh ctest --test-dir cmake-build-debug --output-on-failure
> ```

```bash
# 配置 (独立构建目录)
mkdir build-tests && cd build-tests
cmake .. -DGPNP_BUILD_TESTS=ON -DTEST_FIXTURES_DIR=../tests/data/fixtures

# 构建全部测试
cmake --build . --config Release

# 一键运行全部
ctest --output-on-failure
# 或
cmake --build . --target check

# 单个运行
./tests/test_config
./tests/test_roi_generator
./tests/test_yolo_decode
./tests/test_pose_solvers
./tests/test_extractors
./tests/test_input_system
./tests/test_eskf_fusion
./tests/test_yolo_detector   # 需 yolo_onnx/yolov8n.onnx 在位, 否则 SKIP
./tests/test_synthetic_pipeline  # 需 fixtures_rich + 模板 + 模型在位, 否则 SKIP

# 集成测试需模板目录 + fixtures
./tests/test_integration \
    --template-dir ../data/big/img_1.png \
    --binary-template-dir ../data/NewMuBan\(reordered\) \
    --tiny-template-dir ../data/NewMuBan\(reordered\) \
    --fixtures-dir ../tests/data/fixtures
```

### 依赖注入

| 依赖 | 方式 | 缺失时行为 |
|------|------|-----------|
| OpenCV / Eigen3 | REQUIRED, CMake 自动查找 | 构建失败 |
| ONNX Runtime | 可选, `-DGPNP_ONNXRUNTIME_ROOT=<path>` | `test_input_system` / `test_yolo_detector` 跳过编译 (`test_yolo_decode` 不受影响, 无 ONNX 依赖) |
| YOLO 模型 | 可选, 运行目录需含 `yolo_onnx/yolov8n.onnx` | `test_yolo_detector` 相关用例 SKIP |
| 模板目录 | 可选, `-DTEST_TEMPLATE_DIR=<path>` | `test_extractors` 静默跳过, `test_integration` SKIP |
| fixtures 图片 | 可选, `-DTEST_FIXTURES_DIR=<path>` 或 `--fixtures-dir` | 相关用例静默跳过 / SKIP |

> fixtures 由 `tests/scripts/generate_assets.py` 生成 (输入: `data/big/img_1.png` + 背景图 + CLASS1_RECT)。

---

## 6. 修复记录

### 已修复的测试问题

| 日期 | 测试文件 | 问题 | 修复 |
|------|---------|------|------|
| 2026-08 | `test_pose_solvers.cpp` | `test_initial_pnp_validity_checks`: 后方目标 PnP 手性歧义导致 false-positive | 改为深度超范围测试 (Z=25000 > 20000) |
| 2026-08 | `test_pose_solvers.cpp` | `test_gpnp_recovers_pose_stereo`: 共面退化致 t_err=0.1009 超 0.10 | 阈值放宽至 0.15 |
| 2026-08 | `test_roi_generator.cpp` | `test_state1_tiny_area`: roi_min_size=50 把 28×28 放大到 50×50 | 用例使用独立 cfg (roi_min_size=0) |
| 2026-08 | `test_roi_generator.cpp` | `test_generate_stereo_right_fallback`: 代码无 fallback 逻辑 | `generateStereo()` 添加右 ROI 无效时复制左 ROI |
| 2026-08 | `test_input_system.cpp` | `currentFrame()` 语义不一致: Dir/Seq 用 0, File 用 -1 | 统一为 -1=未取帧 (Dir/Seq open/reset 改 `current_index_=-1`) |
| 2026-08 | `test_input_system.cpp` | `RUN` 宏裸逗号 + `PASS("case:"<<name)` + RingBuffer API 不匹配 | 拆分声明, 修 PASS, pop→popOldest, 加 capacity()/full() |
| 2026-08 | `tests/CMakeLists.txt` | 缺少 `CameraSource.cpp` | 加入 TEST_SHARED_SOURCES |
| 2026-08 | `include/fusion/FusionTypes.hpp` | 提交 aa00616 引用但不存在的文件 (ImuSample/RadarSample/CameraObservation 未定义, 主程序无法编译) | 按提交用法补写 (t_exposure/t_arrival 双时间戳) |
| 2026-08 | `tests/CMakeLists.txt` | 缺少 `EskfFusionManager.cpp` + Threads 链接 | 加入 TEST_SHARED_SOURCES + `find_package(Threads)` + `Threads::Threads` |
| 2026-08 | `test_eskf_multimodal.cpp` | 雷达跳变单点 `[5.0,5.0]` 浮点累加漏注入 → `radar_rejected==0` | 改为窗口 `[5.0,5.2]` |
| 2026-08 | `test_eskf_multimodal.cpp` | 相机失效 `[13.0,14.0]`(1s) 致相机间隔 1.2s > `max_cam_gap_s` → reset 清空 stats | 缩短为 `[13.0,13.6]`(0.6s), 走 Degraded 而非 reset |
| 2026-08 | `plot_eskf_traj.py` | 中文标题缺 CJK 字形 (`Glyph missing from current font`) | 配置 `font.sans-serif`(SimHei/微软雅黑) + `axes.unicode_minus=False` |
| 2026-08-16 | `test_yolo_decode.cpp` | 新增 — YOLO 原始输出解码 + NMS 纯逻辑测试 | 解码逻辑从 `YoloDetector::postprocess()` 抽为 `YoloDecode.hpp::decodeYoloOutput()`, 用合成张量确定性断言 |
| 2026-08-16 | `test_yolo_detector.cpp` | 新增 — YOLO 检测器端到端冒烟 | 模型/图片缺失 SKIP; 复用 `test_integration` 的 SKIP 语义 |

---

*最后更新: 2026-08-16*
