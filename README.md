# Steretracker — GPNP 双目视觉跟踪器

基于 **YOLO 检测 → 手动/自动 ROI → 面积策略分发 → 特征提取 → 视差 → 位姿解算** 流水线的 C++17 双目视觉定位系统，面向无人机视觉定位场景。

**技术栈**: C++17 · OpenCV 4.x · Eigen 3.x · ONNX Runtime

> 项目由 Python 单文件 (~1250 行) 重构为模块化 C++ 架构 (~23 个源文件)。算法逻辑与 Python 版**完全等价**，详见 [PORTING_REPORT.md](./PORTING_REPORT.md)。

---

## 1. 整体流程

```
                          ┌─────────────────┐
                          │ tracker_config.json │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  加载双目图像     │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      逐 帧 循 环             │
                    └──────────────┬──────────────┘
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
                    │ StereoTracker::process()      │
                    │  ├─ is_dual? → processDualRoi │
                    │  ├─ 否则 → configureStrategy  │
                    │  └─ 特征提取 + 退化 + 位姿     │
                    │     (全部内部自动完成)         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 输出位姿 [R|t] + 日志 + 可视化 │
                    └─────────────────────────────┘
```

帧循环中 `main.cpp` 仅负责获取 ROI（手动配置或 YOLO 检测）并调用 `tracker.process()`，所有策略选择和特征提取由 `StereoTracker` 内部自动完成。ROI 优先级：**手动 ROI > YOLO 检测**。

---

## 2. 策略分发与退化机制

### 2.1 主策略选择

系统根据 ROI 面积自动选择**主**特征提取策略：

| ROI 面积 | 主策略 | 提取器 | 位姿解算 |
|----------|------|--------|----------|
| ≤ 800 px² | TinyTarget | `TinyTargetExtractor` | `cv::solvePnP(ITERATIVE)` |
| 801 ~ 40000 px² | BinaryCorner | `BinaryCornerExtractor` | `GPnPSolver` (Eigen LM) |
| ≥ 40001 px² / 无检测 | AKAZE_GPNP | `AkazeGpnpExtractor` | `GPnPSolver` (Eigen LM) |

> **双 ROI 模式**（见[§3](#3-双-roi-策略-dual-roi)）是一种正交于面积阈值的特殊路径：当 YOLO 同时检测到 class 0 和 class 1 目标时，`process()` 直接走 `processDualRoi()`，不走上述策略链。

三种提取器在 `StereoTracker` **构造时一次性预创建并加载全部模板**（包括双 ROI 专用的第二个 AKAZE 提取器）。每帧 `process()` 内部调用 `configureStrategyChain()` 仅通过**指针切换**来选择活跃的主策略和退化链，零堆分配开销。

### 2.2 退化后备链

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

### 2.3 API

```cpp
// StereoTracker 构造函数 — 预创建全部 3 种提取器并加载模板（含双 ROI AKAZE）
StereoTracker tracker(K, R_rl, t_rl, template_path, tracker_cfg,
                      binary_cfg, binary_template_dir,
                      tiny_cfg, tiny_template_dir);

// 每帧处理 — 策略链选择 + ROI padding + 退化全部内部自动完成
auto result = tracker.process(left_img, right_img, visualize, &left_group, &right_group);
```

旧 API `setExtractor()` / `addFallbackExtractor()` / `clearFallbackExtractors()` 已移除。策略链配置由私有方法 `configureStrategyChain(int roi_area)` 完成，对外不可见。

### 2.4 FeatureExtractor 基类

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

## 3. 双 ROI 策略 (Dual-ROI)

**适用场景**：大尺寸目标（class 0 检测框面积 > 700×700 px²），可利用目标的边缘 (class 0) 与中心纹理 (class 1) 两部分信息联合定位。

**模块映射**：`StereoTracker::processDualRoi()` + `AkazeGpnpExtractor` + `BinaryCornerExtractor`

### 3.1 触发条件

当 `RoiGroup::is_dual == true` 时，`process()` 直接调用 `processDualRoi()`，不走常规单策略链。双 ROI 由 `RoiGenerator::generateGroup()` 自动创建：

- **primary ROI**（class 0）：目标整体边界框 → BinaryCorner 提取边缘角点
- **secondary ROI**（class 1）：目标中心区域 → AKAZE 提取纹理特征

### 3.2 完整流水线

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
│  │ GPnPSolver (Eigen LM)                                    │ │
│  │ 初值 = 上一帧位姿 (或默认 500mm 深度)                     │ │
│  │ 优化 = 合并后的全部 (N+K) 条射线                          │ │
│  └────────────────────┬────────────────────────────────────┘ │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 关键成员与配置

| 成员/配置 | 类型 | 用途 |
|-----------|------|------|
| `dual_akaze_extractor_` | `unique_ptr<AkazeGpnpExtractor>` | 双 ROI 专用的 AKAZE 提取器，缩放因子独立于主 AKAZE |
| `dual_bc_template_ready_` | `bool` | BinaryCorner 模板角点是否已从 AKAZE 模板预计算完成 |
| `dual_bc_tmpl_corners_` | `vector<Point2f>` | 从 AKAZE 模板图像中提取的 BinaryCorner 角点（模板坐标系） |
| `dual_bc_tmpl_pts3d_` | `vector<Vector3d>` | 对应 3D 物点坐标（mm） |
| `dual_roi_secondary_expand` | `int` (默认 10) | secondary ROI 的额外拓展像素数 |
| `dual_roi_akaze_scale` | `double` (默认 0.5) | 双 ROI 中 AKAZE 提取的缩放因子 |

### 3.4 模板预处理

`prepareDualBcTemplate()` 在首次使用双 ROI 时**一次性**完成：对 AKAZE 模板图像调用 BinaryCorner 提取器，获得 N 个角点坐标及其对应的 3D 物点。此后每帧合并时直接复用，无需重复提取。

---

## 4. 策略一：TinyTarget（ROI ≤ 800 px²）

**适用场景**：远距离微小矩形目标（如 4 角点标定板）。

**模块映射**：`TinyTargetExtractor` (`src/feature/TinyTargetExtractor.cpp`)

### 4.1 完整流水线

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

### 4.2 位姿解算

TinyTarget **不走 GPNP 优化**，而是直接用 OpenCV 的 `cv::solvePnP(cv::SOLVEPNP_ITERATIVE)` 求解 PnP：

- 4 个图像角点 ←→ 4 个正方形物点（边长 = `square_size_m` 米）
- 物点坐标固定：`(0,0,0)`, `(s,0,0)`, `(s,s,0)`, `(0,s,0)`
- 无初值要求，直接迭代求解

---

## 5. 策略二：BinaryCorner（ROI 801 ~ 40000 px²）

**适用场景**：中等尺寸多边形目标（如 10 角点标识板）。

**模块映射**：`BinaryCornerExtractor` → `InitialPnPSolver` → `GPnPSolver`

### 5.1 完整流水线

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

### 5.2 位姿解算

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

## 6. 策略三：AKAZE_GPNP（ROI ≥ 40001 px² / 无检测）

**适用场景**：大尺寸 / 全图纹理丰富的目标（传统 AKAZE 特征匹配方案）。

**模块映射**：`AkazeGpnpExtractor` → `OpticalFlowTracker` → `StereoProjector` → `TemplateMatcher` → `MadDisparityFilter` → `InitialPnPSolver` → `GPnPSolver`

### 6.1 完整流水线

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

### 6.2 关键模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| `AkazeGpnpExtractor` | `src/feature/AkazeGpnpExtractor.cpp` | AKAZE 提取 + 光流 + 投影 + 模板匹配一次性完成 |
| `OpticalFlowTracker` | `src/feature/OpticalFlowTracker.cpp` | LK 金字塔光流 + FB 一致性检查，`fb_err < 1.0px` |
| `MadDisparityFilter` | `src/feature/MadDisparityFilter.cpp` | 中值绝对偏差（MAD）异常值剔除，`dx_thresh = 3×max(MAD,1.0)` |
| `StereoProjector` | `src/stereo/StereoProjector.cpp` | 视差 → 深度 → 右图投影，输出 valid_mask |
| `TemplateMatcher` | `src/matching/TemplateMatcher.cpp` | Ratio Test → Cross-Check → Homography RANSAC 三阶段级联 |
| `InitialPnPSolver` | `src/pose/InitialPnPSolver.cpp` | RANSAC PnP (300 iter, 8.0px) → ITERATIVE 精化 |
| `GPnPSolver` | `src/pose/GPnPSolver.cpp` | Eigen LM 优化，交叉射线残差，7 维参数空间 |

### 6.3 位姿有效性检查

GPNP 输出位姿需通过以下全部校验：

1. `t[2] > 0` — 相机在目标平面前方
2. `10 < |t| < 20000` mm — 深度在合理范围
3. 所有分量有限（无 NaN/Inf）
4. 旋转矩阵有限

---

## 7. 四策略对比总结

| 维度 | Dual-ROI | TinyTarget | BinaryCorner | AKAZE_GPNP |
|------|----------|-----------|-------------|------------|
| **触发条件** | class 0 > 700×700 px² 且有 class 1 | ≤ 800 px² | 801 ~ 40000 px² | ≥ 40001 px² |
| **特征提取器** | BinaryCorner + AKAZE（并行） | `TinyTargetExtractor` | `BinaryCornerExtractor` | `AkazeGpnpExtractor` |
| **特征类型** | N 角点 + M AKAZE 关键点 | 4 角点 (minAreaRect) | N 角点 (approxPolyDP) | AKAZE 关键点 + 61 维二值描述子 |
| **左右匹配** | 模板匹配 + 光流 | 模板匹配 L↔R | 模板匹配 L↔R | LK 光流 L→R + FB 校验 |
| **视差滤波** | 无 (角点) + MAD (AKAZE) | 无 | 无 | MAD (median ± 3σ) |
| **位姿解算** | `GPnPSolver` (Eigen LM) | `cv::solvePnP(ITERATIVE)` | `GPnPSolver` (Eigen LM) | `GPnPSolver` (Eigen LM) |
| **物点来源** | BC 模板角点 + AKAZE 模板关键点 | 4 正方形顶点 (固定) | 模板角点 (像素→实际尺寸) | 模板 AKAZE 关键点 (像素→mm) |
| **首帧初值** | 上一帧 / 默认 500mm | 无需 | InitialPnP / 深度估算 | InitialPnP / 默认 5000mm |
| **退化角色** | 独立路径（不退化为单策略） | 链末端（无后备） | AKAZE 的后备；可退化为 TinyTarget | 最优先策略；可退化为 BinaryCorner |
| **可视化** | 5 面板（双 ROI 合并） | 标准 solvePnP 输出 | 5 面板 (二值/轴系/模板/立体/重投影) | 4 面板 (特征/立体/模板/坐标轴) |

---

## 8. 核心数据结构

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
| `MadFilterResult` | MAD 滤波输出: 过滤后点集、filter_mask、劣化标志 |
| `YoloConfig` | YOLO 检测器配置: 模型路径、设备类型、置信度阈值等 |
| `Detection` | YOLO 单次检测结果: class_id、confidence、bbox |

---

## 9. 配置文件

配置文件位于 [`config/tracker_config.json`](config/tracker_config.json)，结构如下：

```jsonc
{
  "input": {
    "left": "data/delivery_area_2l/im0.png",
    "right": "data/delivery_area_2l/im1.png"
  },
  "output": {
    "visualize": true
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
| `strategies.akaze_min_area` | 40001 | ROI ≥ 此值时选择 AKAZE_GPNP 策略 |
| `strategies.tiny_max_area` | 800 | ROI ≤ 此值时选择 TinyTarget 策略 |
| `manual_roi.enabled` | `true` | 设为 `true` 跳过 YOLO 检测，直接使用硬编码 ROI |
| `strategies.dual_roi.secondary_expand_pixels` | 10 | 双 ROI 模式下 class 1 ROI 的额外拓展像素 |
| `strategies.dual_roi.akaze.scale` | 0.5 | 双 ROI 模式下 AKAZE 特征提取的缩放因子 |

---

## 10. 工厂函数

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

## 11. 构建与运行

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

## 12. 项目目录结构

```
Steretracker/
├── README.md
├── PORTING_REPORT.md
├── FEATURE_EXTRACTION_SPEC.md
├── CMakeLists.txt
├── main.cpp                    # 程序入口（手动/YOLO ROI + 调用 StereoTracker）
├── best.onnx                   # YOLO ONNX 模型
│
├── config/
│   └── tracker_config.json     # 默认配置文件
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
│   ├── matching/
│   │   └── TemplateMatcher.hpp      # 三阶段模板匹配
│   ├── pose/
│   │   ├── InitialPnPSolver.hpp     # RANSAC+ITERATIVE 初始 PnP
│   │   └── GPnPSolver.hpp           # Eigen LM GPNP 非线性优化
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
│   ├── matching/   (TemplateMatcher)
│   ├── pose/       (InitialPnPSolver + GPnPSolver)
│   ├── stereo/     (StereoProjector)
│   ├── tracker/    (StereoTracker)
│   ├── visualization/ (Visualizer)
│   └── utils/      (PoseUtils)
│
└── test/                             # 单元测试（待添加）