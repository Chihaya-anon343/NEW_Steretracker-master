# Steretracker 自动化测试方案

> 本目录存放项目自动化测试代码。方案基于对 README.md / CLAUDE.md 与全部源码的深度阅读设计。

## 1. 测试目标

| 层级 | 目标 | 覆盖模块 |
|------|------|---------|
| L0 单元 | 纯函数/单类行为验证，无外部依赖 | 配置工厂、ROI 生成与五状态判定、PnP 求解器、提取器、输入系统 |
| L1 集成 | 多模块协同（Tracker 全流程） | 双目/单目/双ROI 合成图像端到端 |
| L2 系统 | 完整二进制 + 真数据 + YOLO/ONNX | 由 CI 手动触发，脚本级对接 |

> 核心原则：**L0/L1 完全脱离 ONNX Runtime 与主工程 main.cpp**，仅依赖 OpenCV+Eigen（已验证：源码中仅 `main.cpp` 与 `YoloRoiProvider.cpp` 依赖 ONNX）。

## 2. 技术栈与框架

- **C++17** · **OpenCV 4.x** · **Eigen 3.x**（与主工程一致）
- **自研轻量断言宏** `TestAssert.hpp`（不引入 GTest 等第三方测试依赖，便于 CI 零配置构建）
- **CMake 集成**：根工程 `GPNP_BUILD_TESTS` 开关，独立 `tests/CMakeLists.txt`

## 3. 目录结构

```
tests/
├── README.md                 # 本方案文档
├── CMakeLists.txt            # 测试构建（独立，复用 src/ 全部可测源文件）
├── framework/
│   └── TestAssert.hpp        # 轻量断言宏 + 测试注册器
├── unit/
│   ├── test_config.cpp               # Config 工厂函数（参数校验/派生量）
│   ├── test_roi_generator.cpp        # 五状态判定 + 双ROI + 近距离回退
│   ├── test_pose_solvers.cpp         # GPnP/InitialPnP/MonoPnP 合成对应点
│   ├── test_extractors.cpp           # 三大提取器（合成矩形目标）
│   └── test_input_system.cpp         # File/Directory/Sequence 图像源
├── integration/
│   ├── test_stereo_pipeline.cpp      # 双目全流程（GPnP warm-start + MAD）
│   ├── test_mono_pipeline.cpp        # 单目全流程（EPnP）
│   └── test_dual_roi.cpp             # 双ROI 合并路径
├── fixtures/                 # 测试资产（脚本生成，不入库）
│   ├── gen_fixtures.py
│   ├── stereo/               # 合成双目图像对
│   ├── templates/            # 合成模板
│   └── sequences/            # 编号图像序列
└── l2_system_test.py         # L2 冒烟脚本（可选，对接真实 best.onnx）
```

## 4. 测试用例设计

### 4.1 五状态判定（test_roi_generator.cpp）— 核心

依据 `RoiGenerator` 面积分级规则设计 **表驱动测试**：

| 用例 | 输入 (class0_area, class1) | 期望 State | 期望策略 |
|------|---------------------------|-----------|---------|
| R01 | ≤800, 无 | 1 远 | TinyTarget |
| R02 | 801~40000, 无 | 2 中 | BinaryCorner |
| R03 | 40001~489999, 无 | 3 中近 | AKAZE |
| R04 | ≥490000, 无 class1 | 3 中近 | AKAZE（单ROI） |
| R05 | ≥490000, 有 class1 | 4 近 | Dual-ROI（is_dual=true） |
| R06 | 无 class0, class1 面积≥min | 5 极近 | CloseRange 回退→按面积重分类 |
| R07 | 无任何检测 | SKIP | RoiGroup 无效，valid()==false |

### 4.2 配置工厂（test_config.cpp）

- 合法参数 → 派生量正确（`focal_length = K(0,0)`, `baseline = |t_rl|`）
- 非法参数 → 抛出 `std::invalid_argument`：
  - K 非标准内参 / 含 NaN / R 行列式≠1 / scale≤0 或 >1 / gpnp_min_pts<3 / 模板尺寸≤0
  - YOLO: 空路径 / conf>1 / iou>1 / input_size≤0

### 4.3 PnP 求解器（test_pose_solvers.cpp）

- **MonoPnPSolver**：合成 8+ 个 3D 点（Z=0 平面）+ 已知 K/R/t 投影 → 解算位姿应接近真值（旋转误差 < 5°，平移误差 < 5%）
- **InitialPnPSolver**：加入 30% 外点 → RANSAC 仍收敛；纯内点 → 精化正确
- **GPnPSolver**：带外点 + warm-start（上帧位姿±5% 扰动）→ 收敛；首帧无初值 → 深度估算回退路径
- 有效性校验：t.z≤0 / |t|∉[10,20000] / NaN → `success=false`

### 4.4 特征提取器（test_extractors.cpp）

合成输入（黑色背景白矩形，已知几何）：

| 提取器 | 构造 | 断言 |
|--------|------|------|
| TinyTarget | 50×50 白方块 | 提取 4 角点，顺序 TL/TR/BR/BL，误差 <2px |
| BinaryCorner | 10 角点星形多边形 | 提取角点数 == 10，重排序与模板 1:1 对应 |
| AKAZE | 纹理模板合成图 | 匹配数 ≥ min_pts，模板 3D 对应正确 |

### 4.5 输入系统（test_input_system.cpp）

- `FileStereoSource`: 给定左右路径 → `getNextPacket()` 返回有效图对，二次调用返回空
- `DirectoryStereoSource`: 按 `pattern` 排序读取编号序列，逐帧递增
- `SequenceSource`: 单目序列，`right_image` 为空
- `InputProvider`: 首帧 `getNextPacket()` 正确组装 `SensorPacket`

### 4.6 集成流水线（integration/）

合成双目立体视觉环境：
- 几何：方形/多边形目标放置于已知深度（如 1000mm），基线 120mm，fx=fy=1000
- 用真实 R,t 渲染左右图 → 跑 `StereoTracker::process()` → 断言位姿误差 <10%

| 集成 | 通道 | 路径 |
|------|------|------|
| test_stereo_pipeline | 双目 | State 2/3 策略 + GPnP warm-start + MAD |
| test_mono_pipeline | 单目 | `processMono()` → MonoPnP EPnP |
| test_dual_roi | 双目 | 双 ROI（class0+class1）合并路径 | 

## 5. 构建与运行

### 5.1 方式一：根工程开关

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPNP_BUILD_TESTS=ON
cmake --build . --config Release --target gpnp_tests
ctest --test-dir build/tests -C Release --output-on-failure
```

### 5.2 方式二：独立构建（推荐 CI 用，完全避开 main.cpp/ONNX）

```bash
cd tests
mkdir build && cd build
cmake .. -DOpenCV_DIR=<opencv_dir>          # 仅需 OpenCV + Eigen
cmake --build . --config Release
ctest --output-on-failure
```

### 5.3 L2 系统冒烟（可选，需要 best.onnx + 真实数据）

```bash
python l2_system_test.py --bin build/Steretracker --data data/ --model best.onnx
```

## 6. 风险与规避

| 风险 | 规避 |
|------|------|
| 主工程 ONNX 路径 Linux 硬编码导致 WIN 编译失败 | 测试构建独立，不编译 `main.cpp`/`YoloRoiProvider.cpp` |
| 提取器对真实图敏感 | 全部用脚本合成图（矩形/多边形/纹理），几何真值已知 |
| 阈值占位值未标定 | 测试使用与配置一致的默认值，且用例断言状态而非绝对面积常量 |
| GPnP 数值敏感 | 对双目位姿断言用宽松容差（旋转<5°、平移<10%） |
| 模板数据依赖 | 合成模板在 fixtures/gen_fixtures.py 中按需生成 |

## 7. 已完成 / 待办

- [x] 方案设计（本文档）
- [ ] TestAssert.hpp 断言框架
- [ ] L0 单元测试（config/roi/pose/feature/input）
- [ ] fixtures 合成资产生成脚本
- [ ] L1 集成测试（stereo/mono/dual_roi）
- [ ] tests/CMakeLists.txt + `GPNP_BUILD_TESTS` 开关 + `.vscode/tasks.json`