# Steretracker 测试体系

本文档描述 `tests/` 目录下的测试体系：分层目标、目录结构、运行方式、每个测试文件的用例详解、风险与已知限制。

---

## 1. 测试目标

测试体系采用 **L0 → L2 三层结构**，从单元级逐步上升到系统级：

| 层级 | 名称 | 目标 | 对应目录 |
|------|------|------|---------|
| L0 | 单元测试 | 验证核心数据结构、配置工厂与几何确定的算法（ROI 生成、PnP 求解） | `tests/unit/` |
| L1 | 单元测试（输入/特征） | 验证图像源扫描配对、特征提取器与流水线流程可运行 | `tests/unit/` |
| L2 | 端到端冒烟测试 | 验证 Tracker 主流程（单目/双目/双 ROI）可完整跑通、无异常 | `tests/unit/test_integration.cpp` |

> **说明**：当前 `tests/unit/` 同时承载 L0/L1/L2 三层用例（见下方"文件总览"）。`integration/` 目录、旧目录结构中的独立 L2 系统脚本（`l2_system_test.py`）已按当前代码状态移除，L2 冒烟用例已并入 `test_integration.cpp`。

### 为什么需要单测

- **提取器/位姿算法依赖真实相机与模板**，无数据环境下难以端到端验证。
- **单元测试使用合成数据 + 几何确定性**（如 Z=0 平面模板、已知位姿投影），可精确验证算法的数值正确性。
- **冒烟测试验证"管线能跑"**：不追求精确位姿，只断言可运行、无崩溃、计时合法。

> **用例查阅指引**：§5 按测试文件逐用例说明，每个用例均以「用例｜输入｜输出｜判断标准」四列呈现——其中「输出」列出被测 API 返回/填充的对象与字段，「判断标准」给出通过/失败的具体判定（断言条件、数值阈值、异常类型、SKIP 条件）。

---

## 2. 技术栈与依赖

| 组件 | 用途 |
|------|------|
| C++17 | 测试代码语言 |
| OpenCV (`core` / `imgproc` / `imgcodecs`) | 图像处理与数据结构 |
| Eigen (`Dense` / `Geometry`) | 矩阵运算与位姿 |
| Python 3 + numpy + opencv-python | 生成合成测试资产（`tests/scripts/generate_assets.py`） |
| CMake | 构建与测试集成（`tests/CMakeLists.txt`） |

所有测试文件均可独立编译为可执行文件（`main()` 直接调用注册的测试）。

---

## 3. 目录结构

```
tests/
├── README.md                    # 本文档
├── CMakeLists.txt               # 测试构建配置（根项目 -DGPNP_BUILD_TESTS=ON 启用）
├── framework/
│   └── TestAssert.hpp           # 断言宏与测试注册表（见 §4）
├── unit/                        # L0/L1 单元测试 + L2 冒烟测试
│   ├── test_config.cpp          # 配置工厂（Stereo/Tracker/YOLO）合法性
│   ├── test_roi_generator.cpp   # ROI 生成：五状态判定 / 边界 / 几何 / 双目
│   ├── test_pose_solvers.cpp    # PnP 求解器：InitialPnP / MonoPnP / GPnP
│   ├── test_extractors.cpp      # 特征提取器：BinaryCorner / TinyTarget
│   ├── test_input_system.cpp    # 输入系统：图像源 / InputProvider / RingBuffer
│   └── test_integration.cpp     # L2 冒烟：单目/双目/双 ROI 主流程
├── scripts/
│   └── generate_assets.py       # 合成测试资产生成脚本（见 §6）
└── data/                        # 可选：生成资产输出（缺省不提交）
```

---

## 4. 断言框架（`tests/framework/TestAssert.hpp`）

所有测试（除 `test_input_system.cpp` 使用自定义宏，见 §5.5）统一使用该头文件的断言宏与注册表：

| 宏 | 作用 |
|----|------|
| `TEST_ASSERT(cond)` | 断言条件成立，失败时输出文件名+行号 |
| `TEST_ASSERT_MSG(cond, msg)` | 断言条件成立，失败时附带自定义信息 |
| `TEST_ASSERT_EQ(a, b)` | 断言 a == b（`operator==`） |
| `TEST_ASSERT_NEAR(a, b, eps)` | 断言 \|a-b\| < eps |
| `TEST_ASSERT_THROWS(expr, ExType)` | 断言表达式抛出指定类型异常 |
| `REGISTER_TEST(func)` | 注册测试函数（内部定义 `RUN_TEST_<n>()` 并在 `main` 前的静态构造中登记） |
| `gpnp_test::TestRegistry::instance().runAll()` | 运行全部注册测试，返回失败用例数（0=成功） |

`runAll()` 统计并打印通过数 / 失败数；测试失败不中断后续用例，全部运行完后返回非零退出码（便于 CI 判定）。

---

## 5. 测试文件详解

> 下表约定：**输出**列 = 被测 API 返回或填充的对象与关键字段；**判断标准**列 = 通过/失败的具体判定（含阈值、异常类型、SKIP 条件）。

### 5.1 `test_config.cpp` — 配置工厂合法性（8 用例）

覆盖三个配置工厂的合法输入与非法输入（抛异常）验证。测试辅助：`makeK()` 构造标准内参（fx=fy=1000, cx=640, cy=512）。

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_stereo_params_valid` | K（fx=fy=1000）、单位 R、t=(-120,0,0) | `StereoCameraParams p` | `p.focal_length≈1000`（NEAR 1e-9）、`p.baseline≈120`（NEAR 1e-9）、`p.K·p.K_inv ≈ I`（范数 <1e-12） |
| `test_stereo_params_invalid_k` | K(2,0)=0.1（末行非标准形式） | 无（构造即抛） | 抛 `std::invalid_argument`（`TEST_ASSERT_THROWS`） |
| `test_stereo_params_invalid_rotation` | R(0,0)=2.0（行列式 ≠1） | 无（构造即抛） | 抛 `std::invalid_argument` |
| `test_stereo_params_nan` | K(1,1)=NaN | 无（构造即抛） | 抛 `std::invalid_argument` |
| `test_tracker_config_valid` | `makeTrackerConfig(0.5, 4, true, 200, 150, 40000, 800, 10, 0.5)` | `TrackerConfig cfg` | 逐字段校验：`scale≈0.5`、`gpnp_min_pts==4`、`use_initial_pnp==true`、`template_real_width_mm≈200`、`template_real_height_mm≈150`、`akaze_min_area==40000`、`tiny_max_area==800`、`dual_roi_secondary_expand==10`、`dual_roi_akaze_scale≈0.5` |
| `test_tracker_config_invalid` | 8 组非法参数：scale=0 / scale=1.5 / min_pts=2 / min_pts=-3 / 模板宽=0 / 模板高=-1 / akaze_min_area=0 / tiny_max_area=0 | 无（构造即抛） | 8 项全部抛 `std::invalid_argument` |
| `test_yolo_config_valid` | `makeYoloConfig("best.onnx", CPU, 0.5f, 0.45f, Size(640,640), 4)` | `YoloConfig cfg` | `model_path=="best.onnx"`、`device==CPU`、`conf_threshold≈0.5f`、`iou_threshold≈0.45f`、`input_size` 640×640、`intra_op_threads==4` |
| `test_yolo_config_invalid` | 6 组非法输入：空路径 / conf=0 / conf=1.5 / iou=0 / 输入宽=0 / 线程=0 | 无（构造即抛） | 6 项全部抛 `std::invalid_argument` |

> `makeTrackerConfig` 完整签名为 `(scale, gpnp_min_pts, use_initial_pnp, template_real_width_mm, template_real_height_mm, akaze_min_area, tiny_max_area, akaze_min_area_class1=0, tiny_max_area_class1=0, dual_roi_secondary_expand=10, dual_roi_akaze_scale=0.5)`，多余参数有默认值。

### 5.2 `test_roi_generator.cpp` — ROI 生成器（16 用例）

核心验证 `RoiGenerator` 的五状态判定、边界条件、close-range 回退、ROI 几何与双目生成。测试辅助：`makeDet(class_id, x, y, w, h)` 构造检测框，`makeGenerator()` 使用 `roi_expand_ratio=0`、`roi_min_size=50`、`dual_trigger_area=490000` 以便精确断言面积。**输出 = `RoiGroup`（valid() / is_dual / primary(x,y,w,h) / secondary）**。

**（1）五状态判定（State 1~4）**

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_state1_tiny_area` | class0 28×28（面积 784 ≤800） | `RoiGroup group` | `valid()==true`、`is_dual==false`、`primary` 面积 ≤800、`secondary.valid()==false` |
| `test_state2_medium_area` | class0 100×100（801~40000） | `RoiGroup group` | `valid()==true`、`primary` 面积 >800 且 ≤40000、`secondary.valid()==false` |
| `test_state3_medium_close_area` | class0 300×300（=90000，40001~489999） | `RoiGroup group` | `valid()==true`、`primary` 面积 >40000 且 <490000、`secondary.valid()==false` |
| `test_state4_dual_roi` | class0 720×720（≥490000）+ class1 80×80 | `RoiGroup group` | `valid()==true`、`is_dual==true`、`secondary.valid()==true`、`secondary` 尺寸恰为 80×80 |

**（2）边界条件**

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_no_detection_invalid` | 空检测列表 | `RoiGroup group` | `valid()==false` |
| `test_only_class1_invalid_without_close_range` | 仅 class1 200×200（close_range 关闭） | `RoiGroup group` | `valid()==false` |
| `test_large_class0_without_class1_not_dual` | class0 720×720 无 class1 | `RoiGroup group` | `valid()==true` 但 `is_dual==false`（降级为单 ROI） |
| `test_roi_not_above_dual_trigger` | class0 699×699（<490000）+ class1 80×80 | `RoiGroup group` | `valid()==true` 且 `is_dual==false`（面积未达阈值，不触发双 ROI） |

**（3）State 5 极近 — class1 回退（`tryCloseRange`）**

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_close_range_recovery` | 仅 class1 400×400，close_range 开启（class1_min_area=100000、扩展 1.5×） | `RoiGroup group` | `valid()==true`，`primary` 恰为 600×600（400×1.5） |
| `test_close_range_disabled` | 仅 class1 400×400，close_range 关闭 | `RoiGroup group` | `valid()==false` |
| `test_close_range_below_min_area` | class1 100×100 <100000 | `RoiGroup group` | `valid()==false` |

**（4）ROI 几何**

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_roi_expand_ratio` | 100×100，扩展 0.1 | `RoiGroup group` | `primary` 恰为 120×120（100 + 2×10） |
| `test_roi_min_size` | 20×20 < roi_min_size=50 | `RoiGroup group` | `primary.width≥50` 且 `primary.height≥50`（强制抬升至最小尺寸） |
| `test_roi_clamp_to_image` | 600×400 附近、扩展 0.5、图像 640×480 | `RoiGroup group` | `primary.x+primary.width ≤640` 且 `primary.y+primary.height ≤480`（裁剪回图像边界） |

**（5）双目 ROI**

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_generate_stereo_right_fallback` | 左图 class0 100×100、右图无检测，图像 640×480 | `pair<RoiGroup,RoiGroup>` | `pair.first.valid()==true`、`pair.second.valid()==true`，右侧 `x` 与 `width` 与左侧完全一致（复制左侧 ROI） |
| `test_generate_stereo_group_dual` | 左右均为 class0 720×720 + class1 80×80，图像 1280×960 | `pair<RoiGroup,RoiGroup>` | `pair.first.valid()==true` 且 `pair.first.is_dual==true` |

> ROI 状态阈值常量（与 `RoiGenerator` 实现一致）：`tiny_max_area=800`、`akaze_min_area=40000`、`dual_trigger_area=490000`，次要 ROI 扩展由 `dual_roi_secondary_expand` 控制。

### 5.3 `test_pose_solvers.cpp` — PnP 求解器（7 用例）

验证三个 PnP 求解器对**合成点云**的位姿恢复能力与无效输入拒绝。使用 Z=0 平面模板（8 点，mm 单位）、标准 K（fx=fy=1000, cx=cy=512）、已知位姿投影生成真值 2D 点，可选叠加高斯像素噪声（σ=0.3px）。辅助：`makePose(ry, tz_mm)` 构造绕 Y 轴旋转位姿，`project()` 完成 [R\|t] 投影，`addPixelNoise()` 叠加噪声。**输出 = `PoseEstimate pose`（success / R / t）**。

**合成立体参数**：`StereoCameraParams` 使用单位 R、`t_rl=(-120,0,0)` 基线 120mm、focal=1000。

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| `test_initial_pnp_recovers_pose` | 8 个 3D 点，真值 ry=0.15rad、tz=1500mm，2D 投影 + σ=0.3px 噪声，`InitialPnPSolver(4)` | `PoseEstimate pose` | `pose.success==true`；平移相对误差 `‖t-t_gt‖/‖t_gt‖ < 5%`；旋转轴角误差 `< 0.09 rad`（≈5°） |
| `test_initial_pnp_rejects_bad_pose` | 仅 3 个 2D 点（< min_pts=4） | `PoseEstimate pose` | `pose.success==false` |
| `test_initial_pnp_validity_checks` | 目标位于相机后方（tz=-1500mm）投影所得 2D 点 | `PoseEstimate pose` | `pose.success==false`（位姿非法被拒绝） |
| `test_mono_pnp_recovers_pose` | 8 点，真值 ry=-0.1rad、tz=800mm，2D 投影 + σ=0.3px 噪声，`MonoPnPSolver` | `PoseEstimate pose` | `pose.success==true`；平移相对误差 `< 5%`；旋转轴角误差 `< 0.09 rad`（≈5°） |
| `test_mono_pnp_rejects_insufficient_points` | 仅 3 个 2D 点（EPnP 需 ≥4） | `PoseEstimate pose` | `pose.success==false` |
| `test_gpnp_recovers_pose_stereo` | 8 点，真值 ry=0.1rad、tz=1200mm，基线 120mm 合成左右图/视差；warm-start 带偏差（ry+0.02rad、tz+50mm），`GPnPSolver(params, 4)` | `PoseEstimate pose`、`double timing` | `pose.success==true`；平移相对误差 `< 10%`；旋转轴角误差 `< 0.15 rad` |
| `test_gpnp_rejects_insufficient_points` | 仅 2 点（< min_pts=4），nullptr warm-start | `PoseEstimate pose` | `pose.success==false` |

> **容差说明**：GPnP 使用双目几何约束，且 warm-start 已带微小偏差，故平移/旋转容差（10% / 0.15rad）比单目解（5% / 5°）宽松，避免因初始化偏差导致测试脆弱。

### 5.4 `test_extractors.cpp` — 特征提取器（8 用例）

验证 `BinaryCornerExtractor` 与 `TinyTargetExtractor`。**注意：部分用例依赖真实模板目录 `data/NewMuBan(reordered)`**；目录不存在时这些用例**静默跳过**（`if (!templateDirExists()) return;`），不打 SKIP 也不打 FAIL。**输出 = `PipelineResult r`（success / kp_left / pts_left_match / pts_right_match）+ 提取器状态（lastMatchedAngle / lastMatchOverlap）**。

| 用例 | 依赖模板 | 输入 | 输出 | 判断标准 |
|------|:---:|------|------|---------|
| `test_interface_names` | 否 | 无 | `StrategyType` 枚举值 | `Akaze==0`、`BinaryCorner==1`、`TinyTarget==2` |
| `test_binary_corner_static_reorder` | 否 | 乱序 4 角点（TL,BR,TR,BL）、中心 (50,50)、参考角 -90° | `reorderByGeometry` 返回的 `order`（4 索引） | `order.size()==4`；4 个索引均 ∈[0,4) 且两两互异 |
| `test_binary_corner_draw_corners` | 否 | 200×200 灰度图 + 4 角点 | `cv::Mat out` | `out` 非空、类型为 `CV_8UC3`、尺寸与输入一致 |
| `test_binary_corner_empty_input` | 是 | 空 `cv::Mat`（左/右/模板/掩膜全空），`BinaryCornerExtractor(cfg, kMuBanDir)` | `PipelineResult r` | 不崩溃；`kp_left` 为空、`pts_left_match` 为空、`success==false` |
| `test_binary_corner_synthetic_rectangle` | 是 | 200×200 白底 + 黑色实心 80×80 矩形，cfg（corners=4、kernel=3、otsu=1.0、像素尺度 0.002 m/px） | `PipelineResult r` | 不崩溃；若 `success==true`：`pts_left_match` 非空且 ≤8、`kp_left` 非空；`success` 不限制（宽松断言） |
| `test_tiny_target_empty_input` | 是 | 空 `cv::Mat`，`TinyTargetExtractor(cfg, kMuBanDir)` | `PipelineResult r` | 不崩溃；`success==false`、`kp_left` 为空 |
| `test_tiny_target_small_black_square` | 是 | 100×100 白底 + 黑色 40×40 方块（State 1），`square_size_m_class0=0.05` | `PipelineResult r` + lastMatchedAngle / lastMatchOverlap | 不崩溃；若 `success==true`：`pts_left_match.size()==4`、每个角点 x/y ∈[0,100]、`lastMatchedAngle()≥0`、`lastMatchOverlap()>0` |
| `test_tiny_target_set_use_class1` | 是 | cfg（class0=0.20m、class1=0.04m），切换 `setUseClass1(true)` / `(false)` | 提取器状态 | 默认 class0 查询 `lastMatchedTemplate()` 不崩溃；切换 class0/class1 不崩溃 |

> 合成图中提取成功与否与真实模板相似度相关，因此除 "不崩溃、无空结果包成功" 的结构性断言外，数值类断言仅对依赖模板的用例做条件性验证（成功时才断言）。

### 5.5 `test_input_system.cpp` — 输入系统（5 组 13 场景）

> **注意**：本文件不使用 `TestAssert.hpp`，而是自带一套 `REQUIRE` / `RUN` / `PASS` 宏（文件内第 73-80 行定义），输出格式为 `[RUN ]` / `[PASS]`，并通过 `g_pass` 统计断言数。

使用**临时目录 + 合成图像（`cv::imwrite`）** 验证各图像源与数据通路。

| 组 | 场景 | 输入 | 输出 | 判断标准 |
|----|------|------|------|---------|
| `testFileSource` | 正常加载一对图像 | 临时目录写入 64×48 的 L.png（值200）/ R.png（值100） | `open()` 返回、`totalFrames()`、`nextFrame(L,R,ts)` 输出、`currentFrame()` | `open` 返回 true、`isOpen()==true`、`totalFrames()==1`、初始 `currentFrame()==-1`；`nextFrame` 返回 true、L/R 非空、64×48、`ts>0`、`currentFrame()==0`；重复取帧返回同一帧（warm-start 兼容）；`reset` 后 `currentFrame()==-1`；`close` 后 `isOpen()==false` |
| | 路径无效 | 空路径 `("","")` / 不存在的 `nonexistent_L.png;nonexistent_R.png` | `open()` 返回 | 两次 `open` 均返回 false |
| | URI 分号格式 | 临时目录 32×32 的 L2.png;R2.png，经 `open("L2.png;R2.png")` 单参形式 | `open()`、`nextFrame(L,R,ts)` | `open` 返回 true、`isOpen()==true`；`nextFrame` 返回 true、L 非空 |
| `testDirectorySource` | 扫描配对 + 排序读取 | 临时目录写入 `left_%04d.png` / `right_%04d.png` 共 3 对，尺寸递增 40×30→42×32 | `open()`、`totalFrames()`、`nextFrame(L,R,ts)` | `open` 返回 true、`totalFrames()==3`；第 1 帧 L/R 宽 40、`currentFrame()==0`；第 3 帧 L 宽 42、`currentFrame()==2`；第 4 次取帧返回 false；`reset` 后回放成功 |
| | 右图缺失自动跳过 | left_0000 有配对、left_0001 无配对（仅 1 对） | `open()`、`totalFrames()`、`nextFrame` | `totalFrames()==1`；取帧成功 1 次，第 2 次返回 false（无配对 left 被跳过） |
| | 目录不存在 / 空目录 | 不存在的目录 / 新建空临时目录 | `open()` 返回 | 两者 `open` 均返回 false |
| `testSequenceSource` | 单目序列扫描 | 临时目录写入 `frame_%04d.png` 2 帧（32×24 / 33×25） | `open()`、`totalFrames()`、`nextFrame(L,R,ts)` | `open` 返回 true、`totalFrames()==2`、初始 `currentFrame()==-1`；取帧 L 宽 32/33 递增，R 非空且宽与 L 相同、`L.type()==R.type()`（右图=左图副本）；越界返回 false；`reset` 后回放 |
| `testInputProvider` | File 配置 | 临时目录 L.png/R.png（64×48），`InputSystemConfig{File, left, right}` | `initialize()`、`isOpen()`、`totalFrames()`、`getNextPacket(packet)` | `initialize` 返回 true、`isOpen()==true`、`totalFrames()==1`、初始 `currentFrame()==0`；packet：`valid==true`、左右图非空、`timestamp_us>0`、`imu.has_value()==false`、`height.has_value()==false`；取完返回 false；`reset` 后重播成功、`currentFrame()==1` |
| | Directory 配置 | 临时目录 2 对 left/right（40×30 / 41×31），`InputSystemConfig{Directory}` | `initialize()`、`totalFrames()`、`getNextPacket(p1/p2)` | `initialize` 返回 true、`totalFrames()==2`；p1 左图宽 40、p2 左图宽 41、第 3 次返回 false |
| | Sequence 配置 | 临时目录 `frame_0000.png`（30×30），`InputSystemConfig{Sequence}` | `initialize()`、`getNextPacket(packet)` | `initialize` 返回 true；packet `valid==true`、左右图均非空 |
| | 无效配置 | File 源指向不存在的 `no_L.png;no_R.png` | `initialize()`、`isOpen()` | `initialize` 返回 false、`isOpen()==false` |
| `testRingBuffer` | 写入/读取顺序 | `RingBuffer<int> buf(4)`，push 10/20/30 | `capacity()`、`empty()`、`full()`、`push()`、`pop(v)` | `capacity()==4`、初始 `empty()==true`、`full()==false`；3 次 push 均 true、`size()==3`；pop 依次得 10/20/30、第 4 次 pop 返回 false、最终 `empty()==true`（FIFO） |
| | 满缓冲区行为 | `RingBuffer<int> buf(2)`，push 1/2/3 | `push()`、`full()`、`size()`、clear 后 `empty()` | push 1/2 均 true、`full()==true`；push 3 返回 false 且 `size()==2`（覆盖最旧）；`clear` 后 `empty()==true` |

### 5.6 `test_integration.cpp` — L2 端到端冒烟（3 用例）

依赖合成双目图对（白底黑点目标，由 `makeTemplatePlane` 3×3 模板点投影生成）验证 Tracker 主流程**可运行、无异常、计时合法**，不追求位姿精度。Data 依赖 AKAZE / BinaryCorner / TinyTarget 模板目录，通过命令行传入：

```bash
./test_integration [--template-dir DIR] [--binary-template-dir DIR] [--tiny-template-dir DIR]
```

- 缺省 `--binary-template-dir` / `--tiny-template-dir` 时回退到 `--template-dir`。
- **目录不存在时用例 SKIP（打印 `[SKIP]`，不判 FAIL）**，便于无数据 CI 环境编译执行。

| 用例 | 输入 | 输出 | 判断标准 |
|------|------|------|---------|
| MonoTracker 冒烟 | 合成点云（Y 旋转 0.15rad、距离 1.5m）+ 手动 ROI（400,300,480,480），`MonoTracker`（K=标准内参、模板目录、`makeTrackerCfg()`） | `PipelineResult res`、`frameCount()`、`getLogs()` | `res.total_time_ms ≥ 0`、`frameCount() ≥ 1`、`getLogs().size() ≥ 1`（日志非空）、无异常 |
| StereoTracker 冒烟 | 双目合成点云（基线 120mm，Y 旋转 0.1rad、距离 1.5m）+ 左右 ROI，`StereoTracker`，连续处理两帧（warm-start 路径） | `PipelineResult r1/r2`、`frameCount()` | 两帧均 `total_time_ms ≥ 0`、`frameCount() ≥ 2`、无异常 |
| StereoTracker Dual-ROI 冒烟 | `is_dual=true`：外部 class0（椭圆 100×75mm）+ 内部 class1（40×30mm），先处理合成全图再处理仅内部图，primary/sec 手动 ROI | `PipelineResult r1/r2`、`frameCount()` | 两帧均 `total_time_ms ≥ 0`、`frameCount() ≥ 2`、无异常（Dual-ROI 独立路径可运行） |

> 冒烟测试不使用真实位姿强断言：合成目标与真实模板差异大（模板匹配/特征提取难以稳定命中），强断言会使用例过于脆弱。位姿数值精度的严格验证由 `test_pose_solvers.cpp`（L0）完成。

---

## 6. 合成资产脚本（`tests/scripts/generate_assets.py`）

生成集成/调试用的**合成图像与 ROI**，输出到 `tests/data/fixtures/`（可用 `--out` 指定）。脚本**不自动生成占位内容**，需用户自备输入（见下方"输入"）。

**思路**：输入为 背景图 + 目标图 + class1 坐标大小；class1 从目标图中按坐标裁剪，目标图本身即 class0，按测试所需大小等比缩放后布置到背景图中心区域。右图为左图水平位移副本（视差固定）。

**输入**（**硬编码在脚本顶部"输入配置"区**，运行前修改为实际路径与坐标，脚本不生成占位、缺失时报错退出）：

| 常量 | 说明 |
|------|------|
| `TARGET_IMG` | 正视目标图路径（即 class0），支持 3/4 通道（4 通道按 alpha 混合） |
| `BACKGROUND_IMG` | 背景图路径（自动 resize 到画布尺寸） |
| `CLASS1_RECT` | class1 在 target.png 中的像素矩形 `{"x":..,"y":..,"width":..,"height":..}`；无 class1 时设为 `None`（dual 场景跳过 class1） |

例：

```python
TARGET_IMG = r"C:/your/path/to/target.png"
BACKGROUND_IMG = r"C:/your/path/to/background.png"
CLASS1_RECT = {"x": 70, "y": 70, "width": 60, "height": 60}
```

**输出与占位参数**（各 State 尺寸与视差，与 §5.2 阈值一致）：

| 场景目录 | class0 短边（画布 640×480） | 视差 | 说明 |
|----------|-----------|------|------|
| `synthetic_tiny/` | 20px（面积 ≤800px²，State 1） | 4px | class0 居中粘贴 |
| `synthetic_bc/` | 150px（801~40000px²，State 2） | 8px | class0 居中粘贴 |
| `synthetic_akaze/` | 210px（>40000px²，State 3） | 16px | class0 居中粘贴 |
| `synthetic_dual/` | 720px（画布 1280×960，面积 ≥490000px²，State 4） | 20px | class0 缩放至 720 框并画白框圈定；class1 由 `class1_rect` 裁剪后缩放至 120px 叠加到中心 |
| `synthetic_bc_class1/` | class1 短边 150px（class0 不存在） | 8px | 仅有 class1：从目标图按 `class1_rect` 裁剪后缩放至 150px 居中粘贴 |
| `synthetic_akaze_class1/` | class1 短边 210px（class0 不存在） | 16px | 仅有 class1：从目标图按 `class1_rect` 裁剪后缩放至 210px 居中粘贴 |
| `mono_tiny/` `mono_bc/` `mono_akaze/` | 各 class0 场景左图副本（单目） | — | 复制 `left_*.png` |
| `mono_bc_class1/` `mono_akaze_class1/` | 各 class1 场景左图副本（单目） | — | 复制 `left_*.png` |
| `manual_roi.json` | Debug 模式手动 ROI（对齐 `synthetic_bc` 第 0 帧实际 bbox，右图带 8px 视差偏移） | — | — |
| `rois.json` | 每幅图 class0/class1 ROI 信息（覆盖全部场景全部帧） | — | 场景中不存在的类别对应 `null`；`synthetic_dual` 同时给出 class0（720 框）与 class1（中心 120px）实际 bbox |

> `mono_dual/` 不生成：dual 场景依赖 class1 叠加，单目无对应需求。
>
> `rois.json` 结构：按场景分组的 `{ "场景名": { "image_size": {...}, "left_000.png": {"class0": bbox|null, "class1": bbox|null}, ... } }`，bbox 格式 `{"x":..,"y":..,"width":..,"height":..}`，可直接用于测试读取左右图 ROI。

**用法**：

```bash
pip install numpy opencv-python
python tests/scripts/generate_assets.py --out tests/data/fixtures
```

---

## 7. 构建与运行

### 7.1 CMake 集成方式（推荐）

测试由根项目的 **`GPNP_BUILD_TESTS`** 总开关启用（`add_subdirectory(tests)`），不使用逐用例开关：

```bash
# 方式一：独立建 test 构建目录（推荐）
mkdir build-tests && cd build-tests
cmake .. -DGPNP_BUILD_TESTS=ON
cmake --build . --config Release
```

```bash
# 方式二：根项目构建时直接开启
cmake .. -DGPNP_BUILD_TESTS=ON
cmake --build . --config Release
```

**依赖与数据注入**：

- OpenCV / Eigen3 为 REQUIRED；ONNX Runtime **可选**（`-DGPNP_ONNXRUNTIME_ROOT=<onnx-root>`）——未找到时 `test_input_system` 自动跳过并打印警告
- 模板目录通过缓存变量注入（供 `test_extractors` / `test_integration` 使用，目录缺失时后者 SKIP）：

```bash
cmake .. -DGPNP_BUILD_TESTS=ON -DTEST_TEMPLATE_DIR="C:/Code/NEW_Steretracker-master/data/NewMuBan(reordered)"
```

构建完成后运行测试（二选一）：

```bash
ctest --output-on-failure --verbose          # 全部已注册用例
cmake --build . --target check               # 一键运行（check 目标）
```

### 7.2 独立编译（不依赖 CMake 集成）

每个 `tests/unit/*.cpp` 都可独立编译成可执行文件（内含 `main`），需链接 OpenCV 与 Eigen：

```bash
g++ -std=c++17 -Iinclude -Itests \
    -I<opencv_include> -I<Eigen_include> \
    tests/unit/test_config.cpp \
    -L<opencv_lib> -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
                  -lopencv_calib3d -lopencv_features2d -lopencv_video
```

运行集成测试需传入模板目录：

```bash
./test_integration --template-dir "data/NewMuBan(reordered)"
```

### 7.3 测试输出

- 无数据环境：`test_extractors.cpp` 中依赖模板的用例**静默跳过**（不打印 SKIP）；`test_integration.cpp` 打印 `[SKIP]`。
- 无 ONNX Runtime 环境：`test_input_system` 不参与编译（CMake 打印警告），其余 5 个用例正常编译运行。
- 数据齐全：全部用例跑完，断言框架输出通过/失败统计；失败时退出码非零，便于 CI 判定。

| 数据/依赖状态 | `test_extractors` | `test_integration` | `test_input_system` |
|---------------|:---:|:---:|:---:|
| 无模板目录 | 依赖用例静默跳过 | 打印 `[SKIP]` | 正常 |
| 无 ONNX Runtime | 正常 | 正常 | 不参与编译（CMake 警告） |
| 数据齐全 | 全部运行 | 全部运行 | 全部运行 |

---

## 8. 已完成 / 待办

### 已完成 ✅

- [x] **L0 配置工厂测试**（`test_config.cpp`）：三工厂合法/非法输入全覆盖（8 用例）
- [x] **L0 ROI 生成器测试**（`test_roi_generator.cpp`）：五状态判定、边界、close-range、几何、双目（16 用例）
- [x] **L0 PnP 求解器测试**（`test_pose_solvers.cpp`）：合成点云位姿恢复 + 无效输入拒绝（7 用例）
- [x] **L1 特征提取器测试**（`test_extractors.cpp`）：接口、静态几何、合成图、空输入（8 用例）
- [x] **L1 输入系统测试**（`test_input_system.cpp`）：图像源、InputProvider、RingBuffer（13 场景）
- [x] **L2 端到端冒烟测试**（`test_integration.cpp`）：单目 / 双目 warm-start / 双 ROI 三类主流程
- [x] **合成资产脚本**（`generate_assets.py`）：6 类场景（3 个 class0 档位 + dual + 2 个 class1-only 档位） + 单目副本 + 手动 ROI + 每帧 class0/class1 ROI（`rois.json`）
- [x] **断言框架**（`TestAssert.hpp`）：TEST_ASSERT / _MSG / _EQ / _NEAR / _THROWS / 注册表 / runAll 汇总
- [x] **CMake 集成**（`tests/CMakeLists.txt`）：`GPNP_BUILD_TESTS` 总开关 + ONNX Runtime 可选检测（缺失时 `test_input_system` 自动跳过），`check` 目标一键运行

### 待办 / 已知限制 ⏳

- [ ] **L2 系统级脚本（离线视频序列 / CAN 数据回灌）**：当前 L2 仅覆盖"合成图 + Tracker.process()"的冒烟路径；尚未提供真实相机序列与 CAN 总线的端到端系统测试脚本
- [ ] **长序列回归测试**：现有合成资产每场景仅 3 帧，未覆盖长时间漂移 / 位姿连续性的回归验证
- [ ] **真实模板数据依赖**：`test_extractors.cpp` 的提取用例依赖 `data/NewMuBan(reordered)`，无数据时被抑制；后续可将合成模板纳入仓库以便无数据环境也能执行
- [ ] **GPnP 外点 / RANSAC 鲁棒性用例**：`test_pose_solvers.cpp` 目前无外点注入，未覆盖 RANSAC 拒绝外点的行为
- [ ] **性能基准**：尚未固化 `total_time_ms` / 提取器耗时的性能基准（回归阈值），当前仅断言非负

---

*最后更新：与 `tests/` 目录实际代码状态对齐（2026-08）*