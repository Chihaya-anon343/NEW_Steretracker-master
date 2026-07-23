# Steretracker — GPNP 双目视觉跟踪器

基于 **YOLO 检测 → 手动/自动 ROI → 面积策略分发 → 特征提取 → 视差 → 位姿解算** 流水线的 C++17 双目视觉定位系统，面向无人机视觉定位场景。

**技术栈**: C++17 · OpenCV 4.x · Eigen 3.x · ONNX Runtime


---

## 1. 整体流程

```
                          ┌─────────────────┐
                          │ tracker_config.json │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  input_system 节存在？            │
                    └──────┬──────────────┬───────────┘
                      true │              │ false
                           ▼              ▼
                    ┌──────────────┐  ┌──────────────────────┐
                    │ InputProvider │  │ cv::imread 加载       │
                    │ (File / Dir   │  │ 旧版 input.left /     │
                    │  / Sequence)  │  │ input.right 路径      │
                    └──────┬───────┘  └──────────┬───────────┘
                           │                     │
                           └──────┬──────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │      逐 帧 循 环             │
                    │  (InputProvider 或 旧版循环)  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  manual_roi.enabled ?            │
                    └──────┬──────────────┬───────────┘
                      true │              │ false
                           ▼              ▼
                    ┌──────────────┐  ┌──────────────────────┐
                    │ 手动 RoiGroup │  │ YOLO ONNX 目标检测    │
                    │ (primary 固定)│  │ 输出左右 RoiGroup      │
                    └──────┬───────┘  │ (可含双 ROI)           │
                           │          └──────────┬───────────┘
                           └──────┬───────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │  mono_mode ?                  │
                    └──────┬──────────────┬─────────┘
                      true │              │ false
                           ▼              ▼
                    ┌──────────────────────────────────┐
                    │ StereoTracker::processMono()      │
                    │  ├─ is_dual? → processDualRoiMono │
                    │  ├─ 否则 → configureStrategy      │
                    │  └─ extractMono() + MonoPnP       │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │ StereoTracker::process()         │
                    │  ├─ is_dual? → processDualRoi    │
                    │  ├─ 否则 → configureStrategy     │
                    │  └─ 特征提取 + 退化 + GPNP       │
                    │     (全部内部自动完成)            │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 输出位姿 [R|t] + 日志 + 可视化 │
                    └─────────────────────────────┘
```

帧循环中 `main.cpp` 仅负责获取 ROI（手动配置或 YOLO 检测），根据 `mono_mode` 配置调用 `tracker.processMono()` 或 `tracker.process()`，所有策略选择和特征提取由 `StereoTracker` 内部自动完成。ROI 优先级：**手动 ROI > YOLO 检测**。

图像加载通过**输入系统**（见[§2](#2-输入系统-input-system)）统一管理：若配置文件中存在 `input_system` 节，优先使用 `InputProvider` 多源抽象；否则回退到旧版 `input.left` / `input.right` 直接 `cv::imread` 加载。

---

## 2. 输入系统 (Input System)

**适用场景**：统一管理多源图像输入（文件/目录/序列/摄像头），为后续多传感器融合（IMU + 高度计）提供基础架构。

**模块映射**：`InputProvider` + `IStereoImageSource` + 三种具体源实现 + `RingBuffer<T>`

### 2.1 概述

输入系统是 Phase 1 新增模块，提供可配置、可扩展的图像源抽象层。配置文件中的 `input_system` 节**优先于**旧版 `input.left` / `input.right` 路径，实现新旧路径无缝兼容。

```cpp
// main.cpp 中的双路径逻辑
if (use_input_system) {
    // 新版：InputProvider 数据驱动的帧循环
    input_provider.initialize(input_sys_cfg);
    while (input_provider.getNextPacket(packet) && frame < max_frames) {
        processFrame(frame, packet.left_image, packet.right_image);
    }
} else {
    // 旧版：cv::imread 加载 + 固定帧数循环（向后兼容）
    for (int frame = 1; frame <= max_frames; ++frame)
        processFrame(frame, left_img, right_img);
}
```

### 2.2 架构

```
┌──────────────────────────────────────────────────────────────┐
│                       InputProvider                           │
│  统一协调器：创建图像源 → 组装 SensorPacket                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────┐  ┌────────────────────┐              │
│  │ IStereoImageSource  │  │ TimeSyncUnit       │ (Phase 2)   │
│  │ (图像源抽象接口)     │  │ (IMU 插值 + 融合)   │              │
│  └────────┬───────────┘  └────────────────────┘              │
│           │                                                   │
│     ┌─────┼─────────────┬──────────────┐                     │
│     ▼     ▼             ▼              ▼                      │
│  ┌────┐ ┌──────────┐ ┌──────────┐ ┌────────┐                │
│  │File│ │Directory │ │Sequence  │ │Camera  │                │
│  │文件│ │双目编号序│ │单目序列  │ │实时摄像│ (Phase 3)       │
│  │对  │ │列        │ │          │ │头      │                │
│  └────┘ └──────────┘ └──────────┘ └────────┘                │
│                                                               │
│  输出: SensorPacket { left, right, timestamp, IMU, 高度 }     │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 三种图像源模式

| type | 类 | 用途 | nextFrame() 行为 |
|------|----|------|-----------------|
| `"file"` | `FileStereoSource` | 静态双目文件对 | 每次返回同一帧图像 |
| `"directory"` | `DirectoryStereoSource` | 双目编号图像序列 | 扫描 `left_*.png` / `right_*.png` 配对，按文件名排序逐帧返回 |
| `"sequence"` | `SequenceSource` | 单目图像序列 | 扫描文件作为 left 输出，right = left.clone() |
| `"camera"` | *未实现* | 实时摄像头 | Phase 3 待开发 |

`FileStereoSource` 等价于旧版 static-file 行为，保持 warm-start 兼容：`max_frames=2` 时同一帧跑两次，第二次携带前一帧位姿做 GPNP warm-start。

### 2.4 SensorPacket 统一数据包

```cpp
struct SensorPacket {
    int64_t timestamp_us;              // 图像时间戳 (微秒)
    cv::Mat left_image;                // 左图
    cv::Mat right_image;               // 右图

    // Phase 2 占位
    std::optional<ImuPacket> imu;      // 时间对齐后的 IMU 插值数据
    std::optional<HeightPacket> height;// 高度计读数

    bool valid = false;
};
```

`InputProvider::getNextPacket()` 负责从图像源拉取下一帧并填充 `timestamp_us`、`left_image`、`right_image`。Phase 2 的 IMU/高度计字段当前置为 `reset()`（空值）。

### 2.5 RingBuffer（线程安全环形缓冲区）

位于 `include/input/RingBuffer.hpp`，是 header-only 模板类，为 Phase 2 多传感器时间对齐提供基础设施：

| 特性 | 说明 |
|------|------|
| 线程安全 | `std::shared_mutex`：生产者独占写（`unique_lock`），消费者共享读（`shared_lock`） |
| 容量上限 | 满时丢弃最老元素，防止内存无限增长 |
| 丢弃计数 | `dropped()` 监控因满溢而丢弃的元素数 |
| 批量消费 | `drainAll()` 一次性排空所有元素（如 ESKF 预测） |
| 快照 | `snapshot()` 读取所有元素但不删除 |

### 2.6 Phase 规划

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1 | ✅ 已完成 | 图像源抽象 (file/directory/sequence) + RingBuffer |
| Phase 2 | 占位 | IMU (`IImuSource`) + 高度计 (`IAltimeterSource`) + `TimeSyncUnit` 时间对齐 |
| Phase 3 | 未开始 | 实时摄像头源 (`CameraSource`) |

Phase 2/3 的接口已在 `InputProvider::createImuSource()` / `createAltimeterSource()` 中预留，当前返回 `false` 并输出 `"尚未实现"` 提示。

### 2.7 配置

见 [§11 配置文件](#11-配置文件) 中的 `input_system` 节说明。

---

## 3. 策略分发与退化机制

### 3.1 主策略选择

系统根据 ROI 面积自动选择**主**特征提取策略：

| ROI 面积 | 主策略 | 提取器 | 位姿解算 |
|----------|------|--------|----------|
| ≤ 800 px² | TinyTarget | `TinyTargetExtractor` | `cv::solvePnP(ITERATIVE)` |
| 801 ~ 40000 px² | BinaryCorner | `BinaryCornerExtractor` | `GPnPSolver` (Eigen LM) |
| ≥ 40001 px² / 无检测 | AKAZE_GPNP | `AkazeGpnpExtractor` | `GPnPSolver` (Eigen LM) |

> **双 ROI 模式**（见[§5](#5-双-roi-策略-dual-roi)）是一种正交于面积阈值的特殊路径：当 YOLO 同时检测到 class 0 和 class 1 目标时，`process()` 直接走 `processDualRoi()`，不走上述策略链。

三种提取器在 `StereoTracker` **构造时一次性预创建并加载全部模板**（包括双 ROI 专用的第二个 AKAZE 提取器）。每帧 `process()` 内部调用 `configureStrategyChain()` 仅通过**指针切换**来选择活跃的主策略和退化链，零堆分配开销。

### 3.2 退化后备链

当主策略（或更高优先级后备）特征提取失败时，系统自动退化到低一级精度策略。退化链按精度从高到低排列：

```
AKAZE_GPNP  ──失败──→  BinaryCorner  ──失败──→  TinyTarget  （终止）
```

| 主策略 | 退化链 | 触发条件 |
|--------|--------|----------|
| AKAZE_GPNP | AKAZE → BinaryCorner → TinyTarget | AKAZE 关键点 < 4 个；BinaryCorner 角点为空 |
| BinaryCorner | BinaryCorner → TinyTarget | BinaryCorner 提取角点为空 |
| TinyTarget | 无进一步后备 | — |

**退化触发**在 `StereoTracker::process()` 内部自动完成：

1. 尝试主策略提取 → 成功则走对应 PnP 解算
2. 主策略特征提取失败 → 记录 `[Degradation]` 日志 → 尝试下一个后备
3. 后备策略提取成功 → 走**该后备策略对应的** PnP 路径（如 BinaryCorner 走 GPNP，TinyTarget 走 solvePnP）
4. 所有策略失败 → 输出 `All strategies failed`

PnP 分发由 `dispatchPnP()` 根据每个 extractor 的 `name()` 自动路由到 `runAkazePnP()` / `runBinaryCornerPnP()` / `runTinyTargetPnP()`，确保退化后的位姿解算与策略匹配。

### 3.3 API

```cpp
// StereoTracker 构造函数 — 预创建全部 3 种提取器并加载模板（含双 ROI AKAZE）
StereoTracker tracker(K, R_rl, t_rl, template_path, tracker_cfg,
                      binary_cfg, binary_template_dir,
                      tiny_cfg, tiny_template_dir);

// 每帧处理 — 策略链选择 + ROI padding + 退化全部内部自动完成
auto result = tracker.process(left_img, right_img, visualize, &left_group, &right_group);
```

旧 API `setExtractor()` / `addFallbackExtractor()` / `clearFallbackExtractors()` 已移除。策略链配置由私有方法 `configureStrategyChain(int roi_area)` 完成，对外不可见。

### 3.4 FeatureExtractor 基类

```cpp
enum class StrategyType { Akaze, BinaryCorner, TinyTarget };

class FeatureExtractor {
public:
    virtual std::string name() const = 0;          // 用于 PnP 路由和退化链去重
    virtual PipelineResult extract(...) = 0;       // 核心提取逻辑
    // ...
};
```

**ROI 外扩**：在送入提取器前，BinaryCorner 和 TinyTarget 策略会由 `applyRoiPadding()` 自动外扩 ROI 边距（`roi_pad_pixels`），为角点提取提供周围上下文像素。

---

## 4. 单目模式 (Mono Mode)

**适用场景**：仅左图可用（如单目相机），右图数据不存在或无效。

**模块映射**：`StereoTracker::processMono()` / `processDualRoiMono()` + `MonoPnPSolver` + 各提取器的 `extractMono()`

### 4.1 概述

单目模式通过配置文件中的 `mono_mode: true` 启用。与双目路径相比，单目路径的核心差异：

| 维度 | 双目 | 单目 |
|------|------|------|
| 输入图像 | 左图 + 右图 | 仅左图 |
| 光流追踪 | LK 金字塔 L→R + FB 校验 | **无** |
| 立体投影 | 视差 → 深度 → 右图重投影 | **无** |
| 视差滤波 | MAD 中值滤波 | **无** |
| 位姿解算 | `GPnPSolver` (Eigen LM, 7 维) | `MonoPnPSolver` (EPnP) |
| 优化约束 | 重投影 + 立体射线 | 仅重投影 |
| Warm-start | 前一帧位姿缓存 | **无** |
| ROI 输入 | `RoiGroup*` (含 primary + secondary) | `RoiGroup*`（同双目，支持双 ROI） |

单目配置结构体 `MonoConfig`（`include/tracker/StereoTracker.hpp:72-76`）：

```cpp
struct MonoConfig {
    bool enabled = false;             ///< true → processMono() 生效
    int akaze_min_area = 40001;
    int tiny_max_area = 800;
};
```

### 4.2 入口与分发

`processMono()` 是单目帧处理的唯一入口，内部根据 ROI 类型分发：

```
processMono(left_img, visualize, left_group)
  │
  ├── left_group->is_dual == true
  │   └── processDualRoiMono(left_img, left_group, visualize)
  │       (单目双 ROI：BC class 0 + AK class 1 → MonoPnP)
  │
  └── left_group->is_dual == false 或 nullptr
      └── 单 ROI 策略链
          ├── configureStrategyChain(roi_area)  ← 同双目
          ├── 尝试 extractMono() → 退化链
          └── mono_pnp_.solve(pts_2d, pts_3d, K)
```

### 4.3 单 ROI 策略链

与双目相同的面积分级逻辑（见 §3.1），但使用各提取器的 `extractMono()` 方法：

**`AkazeGpnpExtractor::extractMono(gray, color)`**
- AKAZE 特征检测（同双目）
- **模板匹配**：Ratio Test → Cross-Check → Homography RANSAC
- **跳过**：光流追踪、立体投影、MAD 滤波
- 输出：`kp_left`, `desc_left`, `good_matches`, `pts_left_match`, `n_template_match`

**`BinaryCornerExtractor::extractMono(gray, color)`**
- Otsu 二值化 → 保留最大连通域 → 形态学平滑
- 模板匹配 (IoU) → 旋转回正 → approxPolyDP 角点提取 → 逆旋转
- 模板角点重排序 (matchCorners)
- **跳过**：右图提取、立体配对
- 输出：`kp_left`, `pts_left_match`, `pts_template_match`, `good_matches`, `template_data_.pts_3d`

**`TinyTargetExtractor::extractMono(gray, color)`**
- Otsu 二值化 → 模板匹配 → 超分辨率放大 → 连通域评分 → minAreaRect → 角点提取
- **跳过**：右图提取、角度对齐
- 输出：`kp_left`, `pts_left_match`, `pts_template_match`, `good_matches`, `template_data_.pts_3d`

> 如果某个提取器未重写 `extractMono()`，基类提供默认实现：直接调用 `extract()` 并将右图参数置空。

### 4.4 单目 Dual-ROI 策略

与双目 Dual-ROI（§5）触发条件相同（YOLO 检测到 class 0 + class 1），但仅使用左图。

**完整流水线**：

```
┌──────────────────────────────────────────────────────────────┐
│                  单目 Dual-ROI 流水线                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─ class 0 ROI (边缘) ─┐    ┌─ class 1 ROI (中心) ────────┐ │
│  │ BinaryCorner 提取     │    │ AKAZE 提取                   │ │
│  │ extractMono()         │    │ extractMono()                │ │
│  │ • Otsu 二值化          │    │ • AKAZE 特征检测             │ │
│  │ • 模板匹配 (IoU)      │    │ • 模板匹配 (三阶段)          │ │
│  │ • 旋转回正             │    │                              │ │
│  │ • approxPolyDP 角点   │    │                              │ │
│  │ • matchCorners 重排序  │    │                              │ │
│  └────────┬──────────────┘    └──────────┬──────────────────┘ │
│           │                              │                    │
│           │  坐标变换:                    │                    │
│           │  class-1-local → class-0-local                    │
│           │                              │                    │
│           ▼                              ▼                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 合并两组特征点:                                          │ │
│  │  - BC 角点[i] ↔ dual_bc_tmpl_pts3d_[i]  (10 个轮廓角点) │ │
│  │  - AK 匹配点[i] ↔ ak_pts3d[good_matches[i].trainIdx]    │ │
│  │     (2~N 个 AKAZE 特征点)                                │ │
│  └────────────────────┬────────────────────────────────────┘ │
│                       │                                       │
│                       ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 恢复全图坐标: offset = (primary.x, primary.y)             │ │
│  └────────────────────┬────────────────────────────────────┘ │
│                       │                                       │
│                       ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ MonoPnPSolver::solve(pts_2d, pts_3d, K)                  │ │
│  │  ┌─ RANSAC EPnP (300 iter, 8.0px, 0.99)                │ │
│  │  │   成功 → ITERATIVE 内点精化                           │ │
│  │  └─ 有效性校验: t.z>0, 10<|t|<20000, 有限值             │ │
│  └────────────────────┬────────────────────────────────────┘ │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

**合并阶段的 3D 对应关系**：

- **BC 贡献**：`result_bc.pts_left_match[i]` ↔ `dual_bc_tmpl_pts3d_[i]`。若 BC 匹配到的模板角度 ≠ 0°，先对 `dual_bc_tmpl_pts3d_` 用 `reorderByGeometry(ref_angle=bc_matched->angle)` 重排，确保角点与 3D 点的物理对应一致。
- **AK 贡献**：`result_ak.pts_left_match[i]` ↔ `ak_pts3d[good_matches[i].trainIdx]`。通过 `trainIdx` 在模板关键点列表中查找对应的 3D 物点。

**与双目 Dual-ROI 的关键差异**：

| 步骤 | 双目 `processDualRoi()` | 单目 `processDualRoiMono()` |
|------|------------------------|---------------------------|
| BC 提取 | `extract()` — 左右图双路提取 + 立体配对 | `extractMono()` — 仅左图 |
| AK 提取 | `extract()` — AKAZE + 光流 + 投影 | `extractMono()` — AKAZE + 模板匹配 |
| BC n_bc_use | `min(n_bc, n_bc_right)` — 受右图约束 | `min(n_bc, n_bc_3d)` — 受 3D 点数约束 |
| AK 合并 | 有立体验证分支 + 回退分支 | 直接使用 `trainIdx` 查找 3D 点 |
| PnP 求解 | `InitialPnP` → `GPnPSolver` (立体射线约束) | `MonoPnPSolver` (EPnP, 无立体) |
| Warm-start | 前一帧位姿 | 无 |
| 坐标恢复 | `offsetResultToOriginal()` (左右图) | 手动 offset (仅左图) |
| 可视化 | 5 面板 (overview/corners/axes/reproj/right/correspondence) | 3 面板 (overview/corners/axes/reproj) |

### 4.5 MonoPnPSolver

`MonoPnPSolver`（`src/pose/MonoPnPSolver.cpp`）是单目位姿估计的核心，流程如下：

```
MonoPnPSolver::solve(pts_2d, pts_3d, K)
  │
  ├─ 1. 校验: pts_2d.size() == pts_3d.size() && ≥ 4
  │    失败 → 返回空 PoseEstimate
  │
  ├─ 2. cv::solvePnPRansac(SOLVEPNP_EPNP)
  │    参数: 300 iter, 8.0px reproj, 0.99 confidence
  │    useExtrinsicGuess = false
  │    失败/内点<4 → 返回失败
  │
  ├─ 3. ITERATIVE 精化 (仅内点子集)
  │    cv::solvePnP(SOLVEPNP_ITERATIVE, useExtrinsicGuess=true)
  │    失败 → 继续使用 RANSAC 结果（非致命）
  │
  └─ 4. 有效性校验
       • t.z > 0（相机在模板前方）
       • 10 < |t| < 20000 mm
       • R, t 各分量有限（无 NaN/Inf）
```

与 `GPnPSolver` 的关键区别：

| 特性 | `MonoPnPSolver` | `GPnPSolver` |
|------|----------------|-------------|
| 算法 | OpenCV EPnP + ITERATIVE | Eigen Levenberg-Marquardt |
| 约束 | 2D↔3D 重投影误差 | 重投影 + 立体射线交叉残差 |
| 参数空间 | 无初始值 | 7 维 [q, t]，依赖 warm-start |
| 帧间状态 | 无缓存 | 缓存上一帧 [R\|t] |
| 适用模式 | 单目 | 双目 |

---

## 5. 双 ROI 策略 (Dual-ROI)

**适用场景**：大尺寸目标（class 0 检测框面积 > 700×700 px²），可利用目标的边缘 (class 0) 与中心纹理 (class 1) 两部分信息联合定位。

**模块映射**：`StereoTracker::processDualRoi()` + `AkazeGpnpExtractor` + `BinaryCornerExtractor`

### 5.1 触发条件

当 `RoiGroup::is_dual == true` 时，`process()` 直接调用 `processDualRoi()`，不走常规单策略链。双 ROI 由 `RoiGenerator::generateGroup()` 自动创建：

- **primary ROI**（class 0）：目标整体边界框 → BinaryCorner 提取边缘角点
- **secondary ROI**（class 1）：目标中心区域 → AKAZE 提取纹理特征

### 5.2 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                    双 ROI 流水线                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─ class 0 ROI (边缘) ─┐    ┌─ class 1 ROI (中心) ────────┐ │
│  │ BinaryCorner 提取     │    │ AKAZE 提取                   │ │
│  │ • Otsu 二值化          │    │ • resize(dual_roi_akaze_    │ │
│  │ • 保留最大连通域       │    │   scale)                    │ │
│  │ • 模板匹配 (IoU)      │    │ • detectAndCompute          │ │
│  │ • approxPolyDP 角点   │    │ • 坐标还原 ÷scale           │ │
│  │ • 模板角点重排序       │    │ • 光流 L→R + FB 校验        │ │
│  └────────┬──────────────┘    │ • 立体投影                   │ │
│           │                   └──────────┬──────────────────┘ │
│           │                              │                    │
│           ▼                              ▼                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 合并两组特征点:                                          │ │
│  │  - 左侧: BinaryCorner N 角点 + AKAZE K 关键点            │ │
│  │  - 右侧: 同上（各自坐标系的对应点）                       │ │
│  │  - 3D 物点: BinaryCorner 模板角点 + AKAZE 模板关键点      │ │
│  │    (AKAZE 物点由 prepareDualBcTemplate() 预计算)          │ │
│  └────────────────────┬────────────────────────────────────┘ │
│                       │                                       │
│                       ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 位姿解算                                                  │ │
│  │  ┌─ 首帧 ─────────────────────────────────────────────┐ │ │
│  │  │ InitialPnP (RANSAC PnP + ITERATIVE 精化)           │ │ │
│  │  │ 成功 → GPNP warm-start                             │ │ │
│  │  │ 失败 → GPNP 默认 depth=5000mm                       │ │ │
│  │  │ 跳过 → 直接 GPNP (若 use_initial_pnp=false)         │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  │  ┌─ 后续帧 ───────────────────────────────────────────┐ │ │
│  │  │ GPNP (Eigen LM) warm-start = 上一帧位姿             │ │ │
│  │  │ 优化 = 合并后的全部 (N+K) 条射线                     │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  └────────────────────┬────────────────────────────────────┘ │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 关键成员与配置

| 成员/配置 | 类型 | 用途 |
|-----------|------|------|
| `dual_akaze_extractor_` | `unique_ptr<AkazeGpnpExtractor>` | 双 ROI 专用的 AKAZE 提取器，缩放因子独立于主 AKAZE |
| `dual_bc_template_ready_` | `bool` | BinaryCorner 模板角点是否已从 AKAZE 模板预计算完成 |
| `dual_bc_tmpl_corners_` | `vector<Point2f>` | 从 AKAZE 模板图像中提取的 BinaryCorner 角点（模板坐标系） |
| `dual_bc_tmpl_pts3d_` | `vector<Vector3d>` | 对应 3D 物点坐标（mm） |
| `dual_roi_secondary_expand` | `int` (默认 10) | secondary ROI 的额外拓展像素数 |
| `dual_roi_akaze_scale` | `double` (默认 0.5) | 双 ROI 中 AKAZE 提取的缩放因子 |

### 5.4 模板预处理

`prepareDualBcTemplate()` 在首次使用双 ROI 时**一次性**完成：对 AKAZE 模板图像调用 BinaryCorner 提取器，获得 N 个角点坐标及其对应的 3D 物点。此后每帧合并时直接复用，无需重复提取。

---

## 6. 策略一：TinyTarget（ROI ≤ 800 px²）

**适用场景**：远距离微小矩形目标（如 4 角点标定板）。

**模块映射**：`TinyTargetExtractor` (`src/feature/TinyTargetExtractor.cpp`)

### 6.1 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                    TinyTarget 流水线                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  左 ROI 灰度图                   右 ROI 灰度图                 │
│       │                              │                        │
│       ▼                              ▼                        │
│  ┌─────────────┐              ┌─────────────┐                │
│  │ Otsu 二值化  │              │ Otsu 二值化  │               │
│  └──────┬──────┘              └──────┬──────┘                │
│         │                            │                        │
│         ▼                            ▼                        │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 模板匹配 (IoU)      │    │ 模板匹配 (IoU)      │            │
│  │ 确定最佳模板角度      │    │ 确定最佳模板角度      │            │
│  │ (遍历所有模板角)      │    │ (遍历所有模板角)      │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 超分辨率放大 (×4)    │    │ 超分辨率放大 (×4)    │            │
│  │ → GaussianBlur     │    │ → GaussianBlur     │            │
│  │ → Otsu → 形态学     │    │ → Otsu → 形态学     │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 连通域评分           │    │ 连通域评分           │            │
│  │ (面积/矩形度/        │    │ (面积/矩形度/        │            │
│  │  中心距/长宽比)      │    │  中心距/长宽比)      │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ minAreaRect → 4顶点│    │ minAreaRect → 4顶点│            │
│  │ cornerSubPix 亚像素 │    │ cornerSubPix 亚像素 │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 角度对齐 (4象限量化)  │    │ 角度对齐 (4象限量化)  │            │
│  │ rotate(0/90/180/270)│    │ rotate(0/90/180/270)│            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           │    ┌────────────────────┘                         │
│           │    │                                              │
│           ▼    ▼                                              │
│  ┌───────────────────────────────────────────┐               │
│  │ 视差计算: disparity = -(left.x - right.x)  │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│  ┌───────────────────────────────────────────┐               │
│  │ cv::solvePnP(ITERATIVE)                    │               │
│  │ 4 个图像角点 ↔ 4 个正方形物点                │               │
│  │ (物点间距 = square_size_m)                 │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 位姿解算

TinyTarget **不走 GPNP 优化**，而是直接用 OpenCV 的 `cv::solvePnP(cv::SOLVEPNP_ITERATIVE)` 求解 PnP：

- 4 个图像角点 ←→ 4 个正方形物点（边长 = `square_size_m` 米）
- 物点坐标固定：`(0,0,0)`, `(s,0,0)`, `(s,s,0)`, `(0,s,0)`
- 无初值要求，直接迭代求解

---

## 7. 策略二：BinaryCorner（ROI 801 ~ 40000 px²）

**适用场景**：中等尺寸多边形目标（如 10 角点标识板）。

**模块映射**：`BinaryCornerExtractor` → `InitialPnPSolver` → `GPnPSolver`

### 7.1 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                   BinaryCorner 流水线                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  左 ROI 灰度图                   右 ROI 灰度图                 │
│       │                              │                        │
│       ▼                              ▼                        │
│  ┌─────────────┐              ┌─────────────┐                │
│  │ Otsu 二值化  │              │ Otsu 二值化  │               │
│  └──────┬──────┘              └──────┬──────┘                │
│         │                            │                        │
│         ▼                            ▼                        │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 保留最大连通域       │    │ 保留最大连通域       │            │
│  │ → 填充孔洞          │    │ → 填充孔洞          │            │
│  │ → 形态学平滑         │    │ → 形态学平滑         │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 模板匹配 (IoU)      │    │ 模板匹配 (IoU)      │            │
│  │ 确定旋转角度         │    │ 确定旋转角度         │            │
│  │ (从文件名解析)       │    │ (从文件名解析)       │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 旋转回正             │    │ 旋转回正             │            │
│  │ → 提取最大轮廓       │    │ → 提取最大轮廓       │            │
│  │ → approxPolyDP      │    │ → approxPolyDP      │            │
│  │   二分搜索 N 角点    │    │   二分搜索 N 角点    │            │
│  │ → 逆旋转回原角度     │    │ → 逆旋转回原角度     │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 模板角点重排序       │    │ 模板角点重排序       │            │
│  │ (reorderByGeometry  │    │ (reorderByGeometry  │            │
│  │  极角对齐)           │    │  极角对齐)           │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           │    ┌────────────────────┘                         │
│           │    │                                              │
│           ▼    ▼                                              │
│  ┌───────────────────────────────────────────┐               │
│  │ 视差计算: disparity = -(left.x - right.x)  │               │
│  │ (全图坐标下直接计算)                        │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│  ┌───────────────────────────────────────────┐               │
│  │ 位姿解算                                    │               │
│  │                                             │               │
│  │  ┌─ 首帧 ──────────────────────────────┐   │               │
│  │  │ ① InitialPnP (RANSAC PnP + 精化)    │   │               │
│  │  │   成功 → GPNP warm-start             │   │               │
│  │  │   失败 → 视差估算深度 (500mm 默认)    │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  │  ┌─ 后续帧 ────────────────────────────┐   │               │
│  │  │ GPNP 优化 (Eigen Levenberg-Marquardt)│   │               │
│  │  │ 初值 = 上一帧位姿                      │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 位姿解算

**首帧**：
1. 若 `use_initial_pnp=true`，运行 `InitialPnPSolver`（RANSAC PnP 300 次迭代 + ITERATIVE 精化），成功后作为 GPNP 的 warm-start
2. 若 InitialPnP 失败或 `use_initial_pnp=false`，则由视差中位数估算深度 `depth = f·b/median_disp`，clamp 至 [50, 5000] mm 作为初值 `(0,0,depth)`

**后续帧**：直接用上一帧位姿作为 GPNP 初值

**GPNP 求解器** (`GPnPSolver`)：基于 **Eigen Levenberg-Marquardt** 非线性优化，最小化双目交叉射线（cross-product）残差：

```
残差 = cross(P_c1 - origin, direction)   // 左射线部分
     + cross(P_c1 - t_rl_cam, direction) // 右射线部分 (变换到左相机坐标系)
```

优化参数 `[qx, qy, qz, qw, tx, ty, tz]`（7 维），Jacobian 由 Eigen 自动数值差分计算。

---

## 8. 策略三：AKAZE_GPNP（ROI ≥ 40001 px² / 无检测）

**适用场景**：大尺寸 / 全图纹理丰富的目标（传统 AKAZE 特征匹配方案）。

**模块映射**：`AkazeGpnpExtractor` → `OpticalFlowTracker` → `StereoProjector` → `TemplateMatcher` → `MadDisparityFilter` → `InitialPnPSolver` → `GPnPSolver`

### 8.1 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                    AKAZE_GPNP 流水线                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  左灰度图 (resize×scale)          右灰度图 (resize×scale)     │
│       │                              │                        │
│       ▼                              │                        │
│  ┌────────────────────┐              │                        │
│  │ AKAZE 特征提取      │              │                        │
│  │ • cv::AKAZE::create │              │                        │
│  │ • detectAndCompute  │              │                        │
│  │ • N×61 二值描述子   │              │                        │
│  │ • 坐标还原 ÷scale   │              │                        │
│  └────────┬───────────┘              │                        │
│           │                          │                        │
│           ▼                          ▼                        │
│  ┌──────────────────────────────────────────┐                │
│  │ LK 金字塔光流 L→R (OpticalFlowTracker)    │                │
│  │ 左图 AKAZE 关键点 → 追踪到右图              │                │
│  │ • winSize=21×21, maxLevel=3               │                │
│  │ • Forward-Backward 一致性校验 (fb<1.0px)  │                │
│  │ • MAD 视差异常点滤波                        │                │
│  │   (|dx - median| < 3 × max(MAD, 1.0))     │                │
│  │ • 劣化处理：匹配点 < 3 → 回退到未滤波数据  │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ 立体投影 (StereoProjector)                │                │
│  │ • 视差 → 深度: depth = f·b / |disp|       │                │
│  │ • 左相机射线 → 3D 点 → 右相机投影          │                │
│  │ • 有效性检查: disparity > 0, Pz > 0       │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ 模板匹配 (TemplateMatcher)                │                │
│  │ ┌─ Stage 1: Ratio Test ────────────────┐ │                │
│  │ │ KNN k=2, threshold=0.75              │ │                │
│  │ │ <4 matches → return                  │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  │ ┌─ Stage 2: Cross-Check ───────────────┐ │                │
│  │ │ 模板→图像方向 Ratio Test + 对称性验证  │ │                │
│  │ │ <4 matches → return                  │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  │ ┌─ Stage 3: Homography RANSAC ─────────┐ │                │
│  │ │ cv::findHomography(RANSAC, 5.0px)     │ │                │
│  │ │ 保留 inlier 内点                       │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ MAD 视差滤波 (MadDisparityFilter)         │                │
│  │ 将投影步骤的 valid_mask 与 MAD 过滤结果同步 │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ 位姿解算                                   │                │
│  │ ┌─ 首帧 ───────────────────────────────┐ │                │
│  │ │ InitialPnP (RANSAC PnP 300+精化)      │ │                │
│  │ │ 成功 → GPNP warm-start                │ │                │
│  │ │ 跳过 → 原始 GPNP (t=[0,0,5000]mm)     │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  │ ┌─ 后续帧 ─────────────────────────────┐ │                │
│  │ │ GPNP (Eigen LM) warm-start = 上一帧   │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 关键模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| `AkazeGpnpExtractor` | `src/feature/AkazeGpnpExtractor.cpp` | AKAZE 提取 + 光流 + 投影 + 模板匹配一次性完成 |
| `OpticalFlowTracker` | `src/feature/OpticalFlowTracker.cpp` | LK 金字塔光流 + FB 一致性检查，`fb_err < 1.0px` |
| `MadDisparityFilter` | `src/feature/MadDisparityFilter.cpp` | 中值绝对偏差（MAD）异常值剔除，`dx_thresh = 3×max(MAD,1.0)` |
| `StereoProjector` | `src/stereo/StereoProjector.cpp` | 视差 → 深度 → 右图投影，输出 valid_mask |
| `TemplateMatcher` | `src/matching/TemplateMatcher.cpp` | Ratio Test → Cross-Check → Homography RANSAC 三阶段级联 |
| `InitialPnPSolver` | `src/pose/InitialPnPSolver.cpp` | RANSAC PnP (300 iter, 8.0px) → ITERATIVE 精化 |
| `GPnPSolver` | `src/pose/GPnPSolver.cpp` | Eigen LM 优化，交叉射线残差，7 维参数空间 |

### 8.3 位姿有效性检查

GPNP 输出位姿需通过以下全部校验：

1. `t[2] > 0` — 相机在目标平面前方
2. `10 < |t| < 20000` mm — 深度在合理范围
3. 所有分量有限（无 NaN/Inf）
4. 旋转矩阵有限

---

## 9. 五策略对比总结

### 9.1 双目策略

| 维度 | Dual-ROI | TinyTarget | BinaryCorner | AKAZE_GPNP |
|------|----------|-----------|-------------|------------|
| **触发条件** | class 0 > 700×700 px² 且有 class 1 | ≤ 800 px² | 801 ~ 40000 px² | ≥ 40001 px² |
| **特征提取器** | BinaryCorner + AKAZE（并行） | `TinyTargetExtractor` | `BinaryCornerExtractor` | `AkazeGpnpExtractor` |
| **特征类型** | N 角点 + M AKAZE 关键点 | 4 角点 (minAreaRect) | N 角点 (approxPolyDP) | AKAZE 关键点 + 61 维二值描述子 |
| **左右匹配** | 模板匹配 + 光流 | 模板匹配 L↔R | 模板匹配 L↔R | LK 光流 L→R + FB 校验 |
| **视差滤波** | 无 (角点) + MAD (AKAZE) | 无 | 无 | MAD (median ± 3σ) |
| **位姿解算** | `GPnPSolver` (Eigen LM) | `cv::solvePnP(ITERATIVE)` | `GPnPSolver` (Eigen LM) | `GPnPSolver` (Eigen LM) |
| **物点来源** | BC 模板角点 + AKAZE 模板关键点 | 4 正方形顶点 (固定) | 模板角点 (像素→实际尺寸) | 模板 AKAZE 关键点 (像素→mm) |
| **首帧初值** | InitialPnP / 默认 5000mm | 无需 | InitialPnP / 深度估算 | InitialPnP / 默认 5000mm |
| **退化角色** | 独立路径（不退化为单策略） | 链末端（无后备） | AKAZE 的后备；可退化为 TinyTarget | 最优先策略；可退化为 BinaryCorner |
| **可视化** | 5 面板（双 ROI 合并） | 标准 solvePnP 输出 | 5 面板 (二值/轴系/模板/立体/重投影) | 4 面板 (特征/立体/模板/坐标轴) |

### 9.2 单目策略 (mono_mode = true)

| 维度 | Mono Dual-ROI | Mono TinyTarget | Mono BinaryCorner | Mono AKAZE |
|------|--------------|-----------------|-------------------|------------|
| **触发条件** | class 0 > 700×700 + class 1 | ≤ 800 px² | 801 ~ 40000 px² | ≥ 40001 px² |
| **提取方法** | `extractMono()` × 2 | `extractMono()` | `extractMono()` | `extractMono()` |
| **输入图像** | 仅左图 (class 0 + class 1 ROI) | 仅左图 | 仅左图 | 仅左图 |
| **光流 / 立体** | 无 | 无 | 无 | 无 |
| **3D 来源** | BC: dual_bc_tmpl_pts3d_, AK: trainIdx | 4 正方形顶点 | 0° 模板角点 × pixel_to_meter | AKAZE 模板 pts_3d[trainIdx] |
| **位姿解算** | `MonoPnPSolver` (EPnP) | `MonoPnPSolver` (EPnP) | `MonoPnPSolver` (EPnP) | `MonoPnPSolver` (EPnP) |
| **帧间缓存** | 无 warm-start | 无 warm-start | 无 warm-start | 无 warm-start |
| **可视化** | 3 面板 (overview/corners/axes/reproj) | 单图关键点+坐标轴 | 单图关键点+坐标轴 | 单图关键点+坐标轴 |

### 9.3 单/双目关键差异

| 维度 | 双目 | 单目 |
|------|------|------|
| 启用方式 | 默认 | `mono_mode: true` |
| API 入口 | `process(left, right, ...)` | `processMono(left, ...)` |
| ROI 参数 | `RoiGroup*` (含 primary + secondary) | `RoiGroup*`（同双目） |
| 去畸变 | 双目极线校正 (可选) | 无 |
| PnP 求解 | `GPnPSolver` (LM 7维) | `MonoPnPSolver` (EPnP + ITERATIVE) |
| 退化约束 | 立体射线 + 重投影 | 仅重投影 |

---

## 10. 核心数据结构

所有模块间数据传递使用强类型结构体，定义于 [`include/common/Types.hpp`](include/common/Types.hpp)：

| 结构体 | 用途 |
|--------|------|
| `StereoCameraParams` | 双目相机内外参 (K, R_rl, t_rl, focal_length, baseline) |
| `TrackerConfig` | 跟踪器配置 (scale, gpnp_min_pts, use_initial_pnp, LK 参数, akaze_min_area, tiny_max_area, dual_roi_secondary_expand, dual_roi_akaze_scale) |
| `PipelineResult` | 单帧完整输出: 特征、光流、投影、匹配、位姿、计时、valid_mask、ROI 偏移量 |
| `TrackResult` | 光流跟踪结果: 左右匹配点、视差、FB 统计 |
| `ProjectionResult` | 立体投影结果: 投影点、valid_mask |
| `MatchResult` | 模板匹配结果: good_matches、匹配统计 |
| `PoseEstimate` | 位姿估计: R, t, success, num_points |
| `GPNPMonitor` | GPNP 优化诊断: 成本、迭代、失败原因 |
| `LogEntry` | 单帧日志: 特征数、匹配数、视差中位数、耗时 |
| `TrackingState` | 帧间状态: 上帧位姿缓存、帧计数、日志列表 |
| `RoiRect` | ROI 矩形: (x, y, width, height) |
| `RoiGroup` | 双 ROI 组: primary (class 0) + secondary (class 1) + is_dual 标志 |
| `MonoConfig` | 单目模式配置: enabled, akaze_min_area, tiny_max_area (定义于 `StereoTracker.hpp`) |
| `MadFilterResult` | MAD 滤波输出: 过滤后点集、filter_mask、劣化标志 |
| `YoloConfig` | YOLO 检测器配置: 模型路径、设备类型、置信度阈值等 |
| `Detection` | YOLO 单次检测结果: class_id、confidence、bbox |

输入系统新增数据结构，定义于 `include/input/`：

| 结构体 | 文件 | 用途 |
|--------|------|------|
| `InputSystemConfig` | `InputConfig.hpp` | 输入系统顶层配置 (图像源 + IMU + 高度计) |
| `SensorPacket` | `InputTypes.hpp` | 统一帧数据包 (left/right 图像 + timestamp + 可选 IMU/高度) |

`MonoPnPSolver`（`include/pose/MonoPnPSolver.hpp`）是单目位姿估计类：

| 方法 | 说明 |
|------|------|
| `solve(pts_2d, pts_3d, K)` | RANSAC EPnP (300 iter, 8.0px) → ITERATIVE 精化 → 有效性校验 |

---

## 11. 配置文件

配置文件位于 [`config/tracker_config.json`](config/tracker_config.json)，结构如下：

```jsonc
{
  "input": {
    "_comment": "===== 旧版输入配置（input_system 不存在时使用）=====",
    "left": "data/delivery_area_2l/im0.png",
    "right": "data/delivery_area_2l/im1.png"
  },

  "input_system": {
    "_comment": "===== 新版输入系统（优先于 input.*）=====",
    "_comment2": "type: file(静态双图), directory(双目序列), sequence(单目序列), camera(实时摄像头-未来)",
    "max_frames": 0,
    "image": {
      "type": "sequence",
      "directory_path": "data/output_frames",
      "left_pattern": "left",
      "right_pattern": "right",
      "sequence_pattern": "frame"
    },
    "imu": {
      "_comment": "IMU 传感器配置 (Phase 2, Linux only)",
      "enabled": false,
      "port": "/dev/ttyUSB0",
      "baud_rate": 921600
    },
    "altimeter": {
      "_comment": "高度计/雷达配置 (Phase 2, Linux only)",
      "enabled": false,
      "can_interface": "can0"
    }
  },

  "mono_mode": false,
  "output": {
    "visualize": true,
    "verbose_console": true
  },
  "camera": {
    "fx": 541.764,
    "fy": 541.764,
    "cx": 553.682,
    "cy": 232.397,
    "baseline_mm": 60.0
  },
  "yolo": {
    "model_path": "best.onnx",
    "conf_threshold": 0.5,
    "target_class_id": 0,
    "roi_expand_ratio": 0.0,
    "roi_min_size": 0
  },
  "manual_roi": {
    "_comment": "设为 true 则跳过 YOLO，直接使用下方 ROI",
    "enabled": true,
    "left":  { "x": 416, "y": 113, "width": 300, "height": 300 },
    "right": { "x": 410, "y": 113, "width": 300, "height": 300 }
  },
  "strategies": {
    "_comment": "ROI 面积阈值: ≤ tiny_max → TinyTarget, ≥ akaze_min → AKAZE, else BinaryCorner",
    "akaze_min_area": 40001,
    "tiny_max_area": 800,
    "dual_trigger_area": 490000,
    "akaze_gpnp": {
      "template_path": "data/delivery_area_2l/im0 - 副本.png",
      "template_real_width_mm": 500.0,
      "template_real_height_mm": 500.0,
      "scale": 0.5,
      "gpnp_min_pts": 4,
      "use_initial_pnp": true
    },
    "binary_corner": {
      "template_dir": "data/NewMuBan(reordered)",
      "corners": 10,
      "kernel_size": 3,
      "corner_scale": 3.0,
      "target_width": 100,
      "target_height": 100,
      "pixel_to_meter_scale": 0.004,
      "roi_pad_pixels": 3,
      "otsu_ratio": 1
    },
    "tiny_target": {
      "template_dir": "data/NewMuBan(reordered)",
      "target_width": 50,
      "target_height": 50,
      "scale_factor": 4,
      "square_size_m": 0.2,
      "roi_pad_pixels": 5
    },
    "dual_roi": {
      "_comment": "class 1 (center) ROI expand pixels + AKAZE extraction scale",
      "secondary_expand_pixels": 10,
      "akaze": {
        "scale": 0.5
      }
    }
  }
}
```

关键配置项说明：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mono_mode` | `false` | 设为 `true` 启用单目模式（仅左图，EPnP 位姿解算） |
| `strategies.akaze_min_area` | 40001 | ROI ≥ 此值时选择 AKAZE_GPNP 策略 |
| `strategies.tiny_max_area` | 800 | ROI ≤ 此值时选择 TinyTarget 策略 |
| `manual_roi.enabled` | `true` | 设为 `true` 跳过 YOLO 检测，直接使用硬编码 ROI |
| `strategies.dual_roi.secondary_expand_pixels` | 10 | 双 ROI 模式下 class 1 ROI 的额外拓展像素 |
| `strategies.dual_roi.akaze.scale` | 0.5 | 双 ROI 模式下 AKAZE 特征提取的缩放因子 |

### 11.1 输入系统配置

`input_system` 节用于替代旧版 `input.left` / `input.right`，提供多源输入抽象。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `input_system.max_frames` | 2 | 最大处理帧数，`0` 表示无限 |
| `input_system.image.type` | — | 图像源类型：`"file"` / `"directory"` / `"sequence"` |
| `input_system.image.left_path` | — | (type=file) 左图文件路径 |
| `input_system.image.right_path` | — | (type=file) 右图文件路径 |
| `input_system.image.directory_path` | — | (type=directory/sequence) 图像目录路径 |
| `input_system.image.left_pattern` | `"left"` | (type=directory) 左图文件名模式 |
| `input_system.image.right_pattern` | `"right"` | (type=directory) 右图文件名模式 |
| `input_system.image.sequence_pattern` | `"frame"` | (type=sequence) 图像文件名模式 |
| `input_system.imu.enabled` | `false` | Phase 2：启用 IMU 传感器输入 |
| `input_system.altimeter.enabled` | `false` | Phase 2：启用高度计传感器输入 |

**向后兼容**：若配置文件中不存在 `input_system` 或 `input_system.image` 为空，则自动回退到 `input.left` / `input.right` 路径方式。旧版路径默认 `max_frames=2`（兼容 warm-start 需要同一帧运行两次的行为）。

---

## 12. 工厂函数

提供三个带参数校验的工厂函数，定义于 [`include/common/Config.hpp`](include/common/Config.hpp)：

| 函数 | 用途 | 校验内容 |
|------|------|---------|
| `makeStereoCameraParams(K, R_rl, t_rl)` | 构造双目相机参数，预计算 K⁻¹、focal_length、baseline | K 为标准内参形式；R_rl det=1.0；无 NaN/Inf |
| `makeTrackerConfig(scale, gpnp_min_pts, ...)` | 构造跟踪器配置（含双 ROI 参数） | `0 < scale ≤ 1.0`；`gpnp_min_pts ≥ 3`；模板尺寸为正；面积阈值为正 |
| `makeYoloConfig(model_path, device, ...)` | 构造 YOLO 检测器配置 | 模型路径非空；`conf_threshold`/`iou_threshold` 在 (0,1]；输入尺寸为正；线程数 ≥ 1 |

### 使用示例

```cpp
using namespace gpnp;

// 相机参数
Eigen::Matrix3d K = ...;
Eigen::Matrix3d R_rl = Eigen::Matrix3d::Identity();
Eigen::Vector3d t_rl(baseline_mm, 0.0, 0.0);

// 创建配置（校验嵌入式在工厂函数内）
auto tracker_cfg = makeTrackerConfig(0.5, 4, true, 500.0, 500.0,
                                      40001, 800, 10, 0.5);
auto yolo_cfg = makeYoloConfig("best.onnx", DeviceType::CPU, 0.5f);

// 初始化 tracker
StereoTracker tracker(K, R_rl, t_rl, template_path, tracker_cfg,
                      binary_cfg, binary_template_dir,
                      tiny_cfg, tiny_template_dir);
```

---

## 13. 构建与运行

### 依赖

| 库 | 用途 |
|----|------|
| OpenCV 4.x (`opencv2/core`, `features2d`, `calib3d`, `imgproc`, `imgcodecs`) | AKAZE、光流、PnP、图像处理 |
| Eigen 3.x | 线性代数、GPNP LM 优化 |
| ONNX Runtime | YOLO 模型推理 |

### 编译

```bash
git clone https://github.com/Chihaya-anon343/NEW_Steretracker.git
cd NEW_Steretracker
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 运行

```bash
# 使用默认配置文件 config/tracker_config.json
./build/Steretracker

# 指定配置文件
./build/Steretracker path/to/config.json
```

### 输出

- **控制台**: 每帧处理摘要（特征点、匹配、投影、视差中位数、GPNP 状态、耗时）
- **日志表**: 所有帧的详细统计（`tracker.printLogs()`）
- **output/ 目录**: 可视化调试图像（若 `visualize=true`）
  - AKAZE: `akaze_***.png`
  - BinaryCorner: `binary_corner_***.png`（5 面板）
  - TinyTarget: `tiny_target_***.png`

---

## 14. 项目目录结构

```
Steretracker/
├── README.md
├── PORTING_REPORT.md
├── FEATURE_EXTRACTION_SPEC.md
├── CMakeLists.txt
├── main.cpp                    # 程序入口（手动/YOLO ROI + InputProvider + StereoTracker）
├── best.onnx                   # YOLO ONNX 模型
│
├── config/
│   └── tracker_config.json     # 默认配置文件（含 input_system）
│
├── data/
│   ├── 大图/                   # AKAZE 场景
│   ├── 中图/                   # BinaryCorner 场景
│   ├── 小图/                   # TinyTarget 场景
│   ├── NewMuBan/               # BinaryCorner 模板库
│   ├── NewMuBan(reordered)/
│   └── rotated/
│
├── include/
│   ├── common/
│   │   ├── Types.hpp           # 14 个强类型结构体（含 RoiGroup, MadFilterResult）
│   │   ├── Config.hpp          # 配置校验与工厂函数
│   │   └── GeometryUtils.hpp   # 几何工具函数 (inline)
│   ├── detection/
│   │   ├── YoloDetector.hpp    # ONNX YOLO 推理
│   │   ├── YoloRoiProvider.hpp # 检测→RoiGroup 转换
│   │   └── RoiGenerator.hpp    # ROI 生成器（支持双 ROI）
│   ├── feature/
│   │   ├── FeatureExtractor.hpp     # 特征提取器抽象基类
│   │   ├── AkazeExtractor.hpp       # 纯 AKAZE 提取（无光流/投影）
│   │   ├── AkazeGpnpExtractor.hpp   # AKAZE 提取 + 光流 + 投影 + 匹配
│   │   ├── BinaryCornerExtractor.hpp# 二值轮廓角点提取
│   │   ├── TinyTargetExtractor.hpp  # 微小矩形目标角点提取
│   │   ├── OpticalFlowTracker.hpp   # LK 光流 + FB 校验
│   │   └── MadDisparityFilter.hpp   # MAD 视差滤波
│   ├── input/                      # ★ 新增：输入系统
│   │   ├── InputProvider.hpp       # 统一协调器
│   │   ├── InputConfig.hpp         # 配置结构体 (InputSystemConfig)
│   │   ├── InputTypes.hpp          # 统一数据包 (SensorPacket)
│   │   ├── IStereoImageSource.hpp   # 图像源抽象接口
│   │   ├── FileStereoSource.hpp    # 静态文件对实现
│   │   ├── DirectoryStereoSource.hpp # 双目编号序列实现
│   │   ├── SequenceSource.hpp      # 单目序列实现
│   │   └── RingBuffer.hpp          # 线程安全环形缓冲区
│   ├── matching/
│   │   └── TemplateMatcher.hpp      # 三阶段模板匹配
│   ├── pose/
│   │   ├── InitialPnPSolver.hpp     # RANSAC+ITERATIVE 初始 PnP
│   │   ├── GPnPSolver.hpp           # Eigen LM GPNP 非线性优化
│   │   └── MonoPnPSolver.hpp        # 单目 EPnP 位姿估计（仅左图）
│   ├── stereo/
│   │   └── StereoProjector.hpp      # 视差→深度→投影
│   ├── tracker/
│   │   └── StereoTracker.hpp        # 核心协调器（策略链 + 退化 + PnP 分发 + 双 ROI）
│   ├── visualization/
│   │   └── Visualizer.hpp           # 多面板调试图像
│   └── utils/
│       └── PoseUtils.hpp            # 位姿工具函数
│
├── src/                             # 对应 .cpp 实现
│   ├── detection/  (YoloDetector + YoloRoiProvider + RoiGenerator)
│   ├── feature/    (AkazeExtractor + AkazeGpnpExtractor + BinaryCornerExtractor
│   │                + TinyTargetExtractor + OpticalFlowTracker + MadDisparityFilter)
│   ├── input/                       # ★ 新增：输入系统实现
│   │   ├── InputProvider.cpp
│   │   ├── FileStereoSource.cpp
│   │   ├── DirectoryStereoSource.cpp
│   │   └── SequenceSource.cpp
│   ├── matching/   (TemplateMatcher)
│   ├── pose/       (InitialPnPSolver + GPnPSolver + MonoPnPSolver)
│   ├── stereo/     (StereoProjector)
│   ├── tracker/    (StereoTracker)
│   ├── visualization/ (Visualizer)
│   └── utils/      (PoseUtils)
│
└── test/                             # 单元测试（待添加）