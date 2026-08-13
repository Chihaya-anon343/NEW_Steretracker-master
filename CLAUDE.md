# CLAUDE.md — Steretracker 项目 AI Agent 说明书

> **目标受众**：后续 AI Agent / Vibe Coding 会话
> **设计原则**：系统级理解优先，按数据流链路组织，关键常量/阈值集中列出，陷阱与边界条件醒目标注
>
> 基于对 `main.cpp`、全部 `include/` 头文件、全部 `src/` 源文件的深度阅读编写。阅读本文后，AI Agent 不再需要遍历代码即可操作和修改本项目。

---

## 目录

1. [项目定义与顶层架构](#1-项目定义与顶层架构)
2. [主入口: main.cpp 数据流](#2-主入口-maincpp-数据流)
3. [核心类型系统](#3-核心类型系统)
4. [五状态策略系统](#4-五状态策略系统)
5. [特征提取器详解](#5-特征提取器详解)
6. [位姿解算器](#6-位姿解算器)
7. [输入系统](#7-输入系统)
8. [配置文件完全指南](#8-配置文件完全指南)
9. [关键实现细节与陷阱](#9-关键实现细节与陷阱)
10. [文件索引速查](#10-文件索引速查)

---

## 1. 项目定义与顶层架构

### 1.1 一句话定义

基于 **YOLO ONNX 检测 → 五状态面积分级 → 策略选择 → 特征提取 → 位姿解算** 流水线的 C++17 双目/单目视觉定位系统，面向无人机视觉定位场景。

### 1.2 技术栈

| 库 | 版本 | 用途 |
|----|------|------|
| C++ | 17 | 语言标准 |
| OpenCV | 4.x | AKAZE、光流、PnP、图像处理、GUI |
| Eigen | 3.x | 线性代数、GPNP LM 优化 |
| ONNX Runtime | - | YOLO 模型推理 |

### 1.3 命名空间

所有代码位于 `namespace gpnp` 内。

### 1.4 顶层数据流

```
main.cpp main()
  ├─ [1] 读取 tracker_config.json
  ├─ [2] 初始化相机参数 K, R_rl, t_rl
  ├─ [3] 构造 TrackerConfig + YoloConfig
  ├─ [4] 创建 InputProvider (normal模式) 或手动加载图像 (debug模式)
  ├─ [5] 创建 StereoTracker (或 MonoTracker) 实例
  ├─ [6] 逐帧循环:
  │   ├─ 获取 SensorPacket (normal) 或 cv::imread (debug)
  │   ├─ YOLO 检测 → RoiGroup (normal始终YOLO; debug可选手动ROI或YOLO)
  │   ├─ mono_mode?
  │   │   ├─ true  → tracker.processMono(left_img, visualize, left_group)
  │   │   └─ false → tracker.process(left_img, right_img, visualize, left_group, right_group)
  │   └─ 输出位姿 + 可视化 + 日志
  └─ [7] 打印摘要统计
```

### 1.5 类继承关系

```
TrackerBase                              (include/tracker/TrackerBase.hpp)
├── MonoTracker                           (include/tracker/MonoTracker.hpp)
│   └── 单目追踪器 (EPnP, 无立体)
│
├── StereoTracker                        (include/tracker/StereoTracker.hpp)
│   └── 双目追踪器 (GPnP + 光流 + MAD)

FeatureExtractor                          (include/feature/FeatureExtractor.hpp)
├── AkazeGpnpExtractor                   (include/feature/AkazeGpnpExtractor.hpp)
├── BinaryCornerExtractor                (include/feature/BinaryCornerExtractor.hpp)
└── TinyTargetExtractor                  (include/feature/TinyTargetExtractor.hpp)
```

> **注意**：单目模式使用独立的 `MonoTracker` 类（继承自 `TrackerBase`）。`MonoTracker` 和 `StereoTracker` 是并列关系，各自持有独立的 PnP 求解器和可视化逻辑。

---

## 2. 主入口: main.cpp 数据流

### 2.1 完整调用链

```
main() main.cpp:22-272
│
├─ 阶段1: 配置解析 (lines 26-127)
│   cv::FileStorage 读取 config/tracker_config.json
│   所有字段提取为局部变量
│   mode = "normal" | "debug"
│   mono_mode = true | false
│
├─ 阶段2: 输出目录 (lines 132-140)
│   normal: output/<input_system.source_dir名>/
│   debug:  output/<左图文件名(去 " - " 后缀)>/
│
├─ 阶段3: 相机参数 + 配置构造 (lines 145-155)
│   K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
│   R_rl = I (默认) 或自定义 rotation_matrix
│   t_rl = [-baseline_mm, 0, 0]  (baseline_mm 需转换为米→毫米取决于配置)
│   TrackerConfig 通过 makeTrackerConfig() 构造 (include/common/Config.hpp)
│
├─ 阶段4: 输入源初始化 (lines 157-176)
│   normal模式:
│     InputProvider provider(input_system_cfg)
│     首次 getNextPacket() 获取 SensorPacket
│   debug模式:
│     cv::imread(input.left_path), cv::imread(input.right_path)
│
├─ 阶段5: YOLO + Tracker 创建 (lines 157-187)
│   YoloConfig → YoloRoiProvider (内部持有 YoloDetector + RoiGenerator)
│   StereoTracker 构造 (内部预创建3个提取器 + dual_akaze + 3个PnP求解器 + MAD滤波器)
│   mono_mode时设置 MonoConfig
│
├─ 阶段6: 帧循环 (lines 192-261)
│
│   ┌─ 单目路径 (mono_mode==true, lines 192-217)
│   │   for each frame:
│   │     if (use_manual_roi && debug):
│   │       left_group = RoiGroup{manual_left_roi, {}, false}
│   │     elif (yolo可用):
│   │       left_group = yolo_provider.detectMono(left_img)
│   │     else:
│   │       left_group = RoiGroup{}
│   │     plg = left_group.valid() ? &left_group : nullptr
│   │     result = tracker.processMono(left_img, visualize, plg)
│   │
│   └─ 双目路径 (默认, lines 218-261)
│       for each frame:
│         if (use_manual_roi && debug):
│           lg = RoiGroup{manual_left_roi, {}, false}
│           rg = RoiGroup{manual_right_roi, {}, false}
│         elif (yolo可用):
│           std::tie(lg, rg) = yolo_provider.detect(left, right)
│         else:
│           lg = rg = RoiGroup{}
│         result = tracker.process(left, right, visualize, &lg, &rg)
│
└─ 阶段7: 统计输出 (line 268)
```

### 2.2 模式对照速查

| 维度 | Normal | Debug |
|------|--------|-------|
| 图像源 | InputProvider (input_system 节) | cv::imread (input 节) |
| ROI 来源 | 始终 YOLO | 手动 ROI 或 YOLO |
| verbose_console | false (简洁) | true (详细) |
| 可视化 | 仅坐标轴叠加 (`mono_f{N}.png`，使用实际帧号) | 各策略多面板中间结果 (使用实际帧号) |
| 日志文件 | 可选 (`output.log_file: true` → `tracking_log.txt`) | 终端打印 ASCII 表格 |
| input_system 配置 | **必填** | 不需要 |
| input 配置 | 不需要 | **必填** |
| 帧循环 | 多帧 (InputProvider 驱动) | 固定2帧 |

---

## 3. 核心类型系统

> 所有类型定义于 `include/common/Types.hpp`，工厂函数定义于 `include/common/Config.hpp`
> 命名空间: `namespace gpnp`

### 3.1 PipelineResult — 贯穿全链路的单一数据载体

这是系统中最重要的结构体，所有提取器的 `extract()`/`extractMono()` 都返回此类型，PnP求解器从此读取输入并回填位姿。

```cpp
struct PipelineResult {
    // === 特征点 ===
    std::vector<cv::KeyPoint> kp_left;              // 左图关键点 (ROI坐标)
    std::vector<cv::KeyPoint> kp_right;             // 右图关键点 (仅双目填充, 单目为空)

    // === 匹配结果 (左右立体匹配) ===
    std::vector<cv::Point2f> pts_left_match;        // 立体匹配成功的左图点 (ROI坐标)
    std::vector<cv::Point2f> pts_right_match;       // 立体匹配成功的右图点 (ROI坐标)

    // === 模板匹配结果 (图像点↔模板关键点/角点的对应关系) ===
    std::vector<cv::Point2f> pts_template_match;    // 与ptm_left_match对应的模板点坐标
    std::vector<cv::DMatch> good_matches;           // DMatch: queryIdx=左图点索引, trainIdx=模板点索引

    // === 光流追踪 ===
    std::vector<cv::Point2f> pts_left_good;         // 光流追踪成功的左图点
    std::vector<cv::Point2f> pts_right_good;        // 光流追踪成功的右图点
    std::vector<float> disparity;                   // 各匹配点的视差值

    // === 状态标志 ===
    bool success = false;                           // 特征提取是否成功
    int n_kp_left = 0;                              // 左图关键点数量
    int n_matched = 0;                              // 立体匹配成功数量
    int n_projected = 0;                            // 立体投影成功数量
    int n_template_match = 0;                       // 模板匹配成功数量 (即 good_matches.size())

    // === 位姿 (PnP求解后回填) ===
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    bool gpnp_success = false;

    // === 计时 (毫秒) ===
    double total_time_ms = 0.0;

    // === 额外标志 (用于Dual-ROI合并) ===
    bool is_class1 = false;                         // class1-only 回退帧标志 (同时用于 LogEntry)
};
```

**坐标约定**:
- `kp_left`、`pts_left_match`、`pts_left_good` 存储**ROI裁剪后的局部坐标**
- 调用方在提取后负责 `+= cv::Point2f(roi.x, roi.y)` 恢复全图坐标
- `pts_template_match` 为模板图像坐标，由 `TemplateData::pts_3d` 查找3D对应

### 3.2 TemplateData — 模板的中央数据存储

```cpp
struct TemplateData {
    // AKAZE 模板 (描述子匹配用)
    std::vector<cv::KeyPoint> keypoints;            // 模板AKAZE关键点
    cv::Mat descriptors;                            // 模板AKAZE描述子 (N×61, CV_8U)
    std::vector<Eigen::Vector3d> pts_3d;            // 对应3D点 (Z=0平面)
    cv::Mat gray_image;                             // 模板灰度图 (用于prepareDualBcTemplate)

    // BinaryCorner / TinyTarget 模板 (形状匹配用)
    std::vector<std::vector<cv::Point2f>> corners;  // 24个角度的角点列表
    std::vector<float> angles;                      // 24个角度值 (0°, 15°, ..., 345°)
    std::vector<cv::Mat> image_bool;                // 24个角度的二值mask (50×50)

    // 几何信息
    float real_width_mm = 0.0f;                     // 模板物理宽度 mm
    float real_height_mm = 0.0f;                    // 模板物理高度 mm
};
```

**三种使用场景**:

| 提取器 | 使用的模板字段 | 3D点来源 |
|--------|---------------|---------|
| AKAZE | keypoints + descriptors + pts_3d | pts_3d[trainIdx] |
| BinaryCorner | corners + angles + image_bool | 0°模板角点 × pixel_to_meter_scale × 1000 |
| TinyTarget | image_bool + angles (IoU匹配) | 硬编码4个3D点: (±half, ±half, 0) |
| Dual-ROI BC | — | prepareDualBcTemplate() 从AKAZE模板灰度图提取 |

### 3.3 其他关键类型

| 类型 | 文件:行 (Types.hpp) | 用途 |
|------|---------------------|------|
| `StereoCameraParams` | 28-47 | K, K_inv, R_rl, t_rl, focal_length, baseline |
| `TrackerConfig` | 50-68 | 策略配置: scale, gpnp_min_pts, use_initial_pnp, 面积阈值等 |
| `FeatureSet` | 66-71 | AKAZE提取输出: keypoints, descriptors, points |
| `TrackResult` | 163-175 | 光流追踪输出: pts_left_good, pts_right_good, disparity, fb_error |
| `ProjectionResult` | 178-188 | 立体投影输出: pts_right_projected, valid_mask |
| `MatchResult` | 191-200 | 模板匹配输出: good_matches, pts_left_match, pts_template_match |
| `PoseEstimate` | 203-210 | PnP输出: R, t, success, num_points |
| `GPNPMonitor` | 213-225 | GPNP优化监控: initial_cost, final_cost, iterations |
| `LogEntry` | 228-242 | 单帧日志: frame, timestamp, is_first, fallback, n_kp, n_match, n_proj, n_tmpl, disp_median, gpnp, total_ms, timing, strategy_name, is_class1, t/rvec |
| `TrackingState` | 245-258 | 帧间缓存: R_prev, t_prev, has_cache, frame_count, logs |
| `RoiRect` | 291-302 | ROI矩形: x, y, width, height; valid() = width>0 && height>0 |
| `RoiGroup` | 305-315 | 双ROI组: primary, secondary, is_dual |
| `Detection` | 374-378 | YOLO检测框: class_id, confidence, bbox (Rect2f) |
| `YoloConfig` | 327-342 | YOLO配置: model_path, device_type, conf_threshold |
| `MadFilterResult` | 318-324 | MAD滤波输出: filtered_points, filter_mask, degraded |

### 3.4 Status 枚举

```cpp
enum class Status {
    Success = 0,
    EmptyInput,
    ModelLoadFailed,
    InferenceFailed,
    UnknownError
};
```

### 3.5 DeviceType 枚举

```cpp
enum class DeviceType {
    Auto,    // 优先CUDA，不可用时CPU
    CPU,
    CUDA
};
```

### 3.6 StrategyType 枚举 (include/feature/FeatureExtractor.hpp)

```cpp
enum class StrategyType {
    Akaze,
    BinaryCorner,
    TinyTarget
};
```

用于 PnP 路由 (`dispatchPnP`) 和退化去重。

---

## 4. 五状态策略系统

### 4.1 五状态判定树 (RoiGenerator::generateGroup)

```
输入: YOLO检测结果 (class 0 = 整体, class 1 = 中心)
       ↓
  ┌─ 有 class0 ─────────────────────────────────────┐
  │                                                  │
  │  面积 ≤ tiny_max_area (800)     → State 1 远     │
  │  801 ≤ 面积 ≤ 40000             → State 2 中     │
  │  40001 ≤ 面积 < 490000          → State 3 中近    │
  │  面积 ≥ 490000 && 无 class1     → State 3 中近    │
  │  面积 ≥ 490000 && 有 class1     → State 4 近     │
  │                                                  │
  ├─ 无 class0 && 有 class1 ─────────────────────────┤
  │  面积 ≥ class1_min_area        → State 5 极近    │
  │     (用 class1 生成 ROI → 按面积重分类)           │
  │                                                  │
  └─ 无 class0 && 无 class1 ─────────────────────────┤
      → SKIP 帧 (不触发任何策略，不进入退化链)         │
```

### 4.2 关键面积阈值常量

| 常量 | 类型 | 默认值 | 位置 (main.cpp读取) | 含义 |
|------|------|--------|---------------------|------|
| `tiny_max_area` | int | 800 | strategies.tiny_max_area | State 1/2 分界 |
| `akaze_min_area` | int | 40001 | strategies.akaze_min_area | State 2/3 分界 |
| `dual_trigger_area` | int | 490000 | strategies.dual_roi | State 3/4 分界 (700×700) |
| `class1_min_area` | int | — | strategies.close_range | State 5 class1最小面积 |
| `akaze_min_area_class1` | int | 0 | strategies.close_range.akaze_min_area | class1-only 专用 AKAZE 阈值 (0=回退到通用值) |
| `tiny_max_area_class1` | int | 0 | strategies.close_range.tiny_max_area | class1-only 专用 TinyTarget 阈值 |

> ⚠️ **这些值为早期测试占位值**。在实际部署场景中必须重新标定。修改方法：
> 1. `config/tracker_config.json` 中修改对应值
> 2. `configureStrategyChain()` 中使用的阈值来自 `akaze_min_area_` 和 `tiny_max_area_` 成员变量

### 4.3 策略链配置: configureStrategyChain()

```cpp
// include/tracker/TrackerBase.hpp protected方法
// StereoTracker构造中调用，每帧根据ROI面积重新配置

void configureStrategyChain(int roi_area, bool is_class1 = false) {
    // 选择阈值：class1 专用阈值 > 0 时使用，否则回退到通用阈值
    int akaze_thresh = is_class1 && akaze_min_area_class1_ > 0
        ? akaze_min_area_class1_ : akaze_min_area_;
    int tiny_thresh  = is_class1 && tiny_max_area_class1_ > 0
        ? tiny_max_area_class1_  : tiny_max_area_;

    if (roi_area >= akaze_thresh || roi_area == 0) {
        extractor_ = akaze_extractor_.get();           // 主策略: AKAZE
        fallback_extractors_ = {binary_extractor_.get(), tiny_extractor_.get()};
    } else if (roi_area > tiny_thresh) {
        extractor_ = binary_extractor_.get();           // 主策略: BC
        fallback_extractors_ = {tiny_extractor_.get()};
    } else {
        extractor_ = tiny_extractor_.get();             // 主策略: TT
        fallback_extractors_ = {};
    }
}
```

**调用时机**: 每帧在 `process()`/`processMono()` 中 ROI 裁剪后、特征提取前调用。

**特殊处理**: `roi_area == 0` (传入nullptr ROI→全图回退) 走 AKAZE 链。

### 4.4 五状态总览表

| State | 名称 | 条件 | 主提取器 | 双目Pose | 单目Pose | Warm-start | 退化链 |
|-------|------|------|----------|----------|----------|-----------|--------|
| 1 | 远 | ≤800 px² | TinyTarget | solvePnP ITERATIVE | MonoPnP EPnP | ❌ | 无→终止 |
| 2 | 中 | 801~40000 | BinaryCorner | InitialPnP→GPnP | MonoPnP EPnP | ✅ 双目 | TT |
| 3 | 中近 | 40001~489999 | AKAZE | InitialPnP→GPnP | MonoPnP EPnP | ✅ 双目 | BC→TT |
| 4 | 近 | ≥490000+class1 | Dual-ROI (BC+AK) | InitialPnP→GPnP | MonoPnP EPnP | ✅ 双目 | 独立路径 |
| 5 | 极近 | 无class0+有class1 | 按面积重分类 | 按重分类 | 按重分类 | 按重分类 | 按重分类 |

### 4.5 退化后备链

```
AKAZE_GPNP  ──失败──→  BinaryCorner  ──失败──→  TinyTarget  ──失败──→ 终止
```

| 规则 | 说明 |
|------|------|
| Dual-ROI (State 4) | 独立并行路径，不进入退化链 |
| YOLO 无检测 | 直接 skip 帧，不触发退化链 |
| State 5 回退 | class 1 重分类后重新走 State 1~4 策略链 |
| 退化链由 `configureStrategyChain()` 一次性配置 | 调用方只需遍历 `fallback_extractors_` 列表 |
| 特征提取 "失败" 的定义 | `PipelineResult::success == false` 或 `n_kp_left < 3` |

---

## 5. 特征提取器详解

### 5.1 基类 FeatureExtractor

```cpp
// include/feature/FeatureExtractor.hpp
class FeatureExtractor {
public:
    virtual std::string name() const = 0;
    virtual PipelineResult extract(
        const cv::Mat& left_gray, const cv::Mat& right_gray,
        const cv::Mat& left_color, const cv::Mat& right_color) = 0;
    virtual PipelineResult extractMono(
        const cv::Mat& gray, const cv::Mat& color);  // 默认调用 extract(gray,empty,color,empty)
};
```

`name()` 返回值: "AKAZE_GPNP", "BinaryCorner", "TinyTarget" — 用于 PnP 路由和退化去重。

### 5.2 State 3: AKAZE_GPNP 提取器

**类**: `AkazeGpnpExtractor` (include/feature/AkazeGpnpExtractor.hpp, src/feature/AkazeGpnpExtractor.cpp)

**内部组件**:
- `AkazeExtractor` — 纯 AKAZE 检测与描述子计算
- `OpticalFlowTracker` — Lucas-Kanade 光流 + Forward-Backward 校验
- `StereoProjector` — 视差→深度→右图重投影
- `TemplateMatcher` — 三阶段模板匹配 (Ratio Test → Cross-Check → Homography RANSAC)

**双目 extract() 全流水线**:

```
输入: left_gray, right_gray (ROI裁剪后的灰度图)
      left_color, right_color (ROI裁剪后的彩色图，仅可视化用)

[Step 1] AkazeExtractor::extract()
  ├─ 若 scale < 1.0: resize(INTER_LINEAR) 降采样
  ├─ detectAndCompute(AKAZE, NORM_HAMMING) → 61字节二值描述子
  └─ 坐标 × (1.0/scale) 还原到原始ROI尺寸
  → PipelineResult.kp_left, PipelineResult.kp_right (=kp_left初始值)

[Step 2] OpticalFlowTracker::track()
  输入: left_gray, right_gray, kp_left
  ├─ calcOpticalFlowPyrLK(L→R)
  ├─ calcOpticalFlowPyrLK(R→L) (反向)
  └─ Forward-Backward 误差校验: FB_error < 1.0px
  → TrackResult{pts_left_good, pts_right_good, disparity, fb_error, num_valid}

[Step 3] StereoProjector::project()
  输入: pts_left_good, pts_right_good, disparity, K, baseline
  ├─ 深度 = focal_length * baseline / disparity
  ├─ 3D点反投影: X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy, Z = depth
  ├─ 3D点变换到右相机: P_right = R_rl * P_left + t_rl
  └─ 右图投影: (u', v') = project(P_right, K)
  → ProjectionResult{pts_right_projected, valid_mask}

[Step 4] TemplateMatcher::match() 三阶段
  输入: kp_left, descriptors_left, template_kp, template_desc

  Stage 1 — Ratio Test (TemplateMatcher.cpp:48-76):
    knnMatch(L→T, k=2, NORM_HAMMING)
    Lowe's ratio: dist[0] < 0.75 × dist[1]
    <4 matches → 提前返回

  Stage 2 — Cross-Check (cpp:78-112):
    knnMatch(T→L, k=2) + ratio test
    对称验证: T→L映射回L→T同一索引
    <4 → 提前返回

  Stage 3 — Homography RANSAC (cpp:114-148):
    findHomography(L_pts, T_pts, RANSAC, 5.0px)
    H为空 → 返回Stage2结果
    仅保留inlier

  → MatchResult{good_matches, pts_left_match, pts_template_match, num_matches}

[Step 5] 填充结果
  ├─ kp_left, kp_right
  ├─ pts_left_match, pts_right_match (filtered by stereo matching + MAD)
  ├─ pts_template_match (from template matching)
  ├─ good_matches
  ├─ pts_left_good, pts_right_good (光流成功点)
  ├─ disparity
  ├─ success = (num_matches >= gpnp_min_pts 默认3)
  └─ n_kp_left, n_matched, n_projected, n_template_match
```

**单目 extractMono() 简化流程**:

```
同双目但跳过:
  ✗ OpticalFlowTracker (无光流)
  ✗ StereoProjector (无立体投影)
  ✗ MAD 视差滤波
  ✗ pts_right 系列字段为空

保留:
  ✓ AKAZE 检测 + 描述子
  ✓ 三阶段模板匹配
  ✓ PipelineResult 中仅填充左侧字段
```

**模板预计算** `setTemplateData()` (cpp:33-45):

```
AkazeExtractor(scale=1.0)  // 全分辨率
→ detectAndCompute(template_gray)
→ 每个关键点: 3D = (kp.x/tw * real_w_mm, kp.y/th * real_h_mm, 0)
→ TemplateData.keypoints[i] ↔ TemplateData.pts_3d[i]
```

**PnP时3D点查找** (processMono/process中):

```cpp
// 对每个 good_match:
matched_pts_3d.push_back(template_.pts_3d[m.trainIdx]);
// trainIdx 直接索引到模板关键点列表
```

### 5.3 State 2: BinaryCorner 提取器

**类**: `BinaryCornerExtractor` (include/feature/BinaryCornerExtractor.hpp, src/feature/BinaryCornerExtractor.cpp)

**模板**: 24个角度 (.txt 角点坐标 + .png 二值mask `image>127`)，角度步长 15° (0°, 15°, ..., 345°)

**Config 结构**:
```cpp
struct Config {
    int corners = 10;              // 目标角点数
    int kernel_size = 3;           // 形态学核大小
    float scale = 1.0;             // 提取前放大尺度
    int target_size = 100;         // 模板匹配标准化尺寸
    float pixel_to_meter_scale;    // 像素→毫米换算
    int roi_pad_pixels = 0;        // ROI外扩像素
    float otsu_ratio = 1.3;        // Otsu阈值乘数
};
```

**双目 extract() 全流水线**:

```
[Step 1] 左右独立 extractFromBinary():
  对左右灰度图分别执行:

  1.1 Otsu二值化: threshold(gray, binary, 0, 255, THRESH_BINARY|THRESH_OTSU)

  1.2 keepLargestRegion():
      connectedComponentsWithStats(8)
      排除背景(label=0)，取最大面积连通域
      num_labels ≤ 1 → 返回原二值图

  1.3 fillHoles():
      findContours(RETR_EXTERNAL) → drawContours(FILLED)

  1.4 smoothBoundary():
      MORPH_CLOSE(MORPH_RECT, kernel_size) → MORPH_OPEN(MORPH_RECT, kernel_size)

  1.5 模板匹配 (IoU):
      resize二值图到 target_size×target_size
      findBestMatch():
        extractAndNormalizeRoi(binary, 50) → 裁剪包围盒→正方形→resize 50×50
        遍历24个模板: calculateOverlap(norm_binary, template.image_bool)
        IoU = countNonZero(A & B) / countNonZero(A | B)
      取最高IoU角度 → last_matched_template_

  1.6 旋转回正:
      优选: warpAffine(INTER_CUBIC)旋转灰度图
        → Otsu × otsu_ratio
        → keepRegionFromCenter(): 从中心螺旋搜索→floodFill
        → fillHoles + smoothBoundary
      回退: warpAffine(INTER_NEAREST)旋转二值图 + 形态学

  1.7 extractLargestContour():
      findContours(RETR_EXTERNAL) → 取最大面积

  1.8 extractCornersFromContour():
      可选 scale up (config_.scale)
      perimeter = arcLength(contour, closed=true)
      二分搜索 epsilon ∈ [0.001, 0.05]×perimeter, 8次迭代
        目标 n == config_.corners
        n < target → hi = mid; n > target → lo = mid
      无精确命中 → 取 best_diff 最近似
      scale down 回原始

  1.9 逆旋转 + 重排序:
      逆旋转角点坐标回原始方向
      matchCorners(tmpl, corners): reorderByGeometry → 1:1 索引配对

  → 左图 corners, 右图 corners

[Step 2] 左右模板匹配 (IoU):
  左图best_match_angle附近搜索右图匹配 (15°容差)
  左右结果几何对齐

[Step 3] 3D点构造:
  使用0°模板角点坐标
  X = (tmpl_corners[i].x) × pixel_to_meter_scale × 1000 → mm
  Y = (tmpl_corners[i].y) × pixel_to_meter_scale × 1000 → mm
  Z = 0

[Step 4] 填充 PipelineResult:
  kp_left, kp_right = 角点转KeyPoint
  pts_left_match, pts_right_match = 角点坐标
  good_matches = [DMatch(i, i, 0) for i in range(n)]  (1:1 一一对应)
  pts_template_match = 0°模板角点坐标
  success = !corners.empty()
```

**reorderByGeometry() 算法** (cpp:735-791):

```
零轴 = 正上方 (-y方向), CCW = 正方向
每个角点: angle = atan2(-dx, -dy), 归一化到 [0, 2π)
找最接近 ref_angle 的角点为起点
若 ref_dist > 0: 距离中心最近的优先
从起点 CCW 遍历: [start, start+1, ..., n-1, 0, ..., start-1]
```

### 5.4 State 1: TinyTarget 提取器

**类**: `TinyTargetExtractor` (include/feature/TinyTargetExtractor.hpp, src/feature/TinyTargetExtractor.cpp)

**模板**: 同 BinaryCorner，24个角度 (0°~345°)，标准化尺寸 50×50

**Config 结构**:
```cpp
struct Config {
    int target_size = 50;          // 模板标准化尺寸
    float scale_factor = 4.0;      // 超分辨率放大倍数
    float square_size_m;           // 目标正方形边长 (米)
    int roi_pad_pixels = 0;        // ROI外扩
};
```

**extractMono() / extract() 核心 — extract4Corners()**:

```
[Step 1] Otsu + 模板匹配 (IoU):
  threshold(gray, binary, 0, 255, THRESH_BINARY|THRESH_OTSU)
  matchTemplate():
    connectedComponentsWithStats(8) → 取最大连通域
    裁剪包围盒 → 扩展正方形 → resize 50×50 (INTER_NEAREST)
    遍历24个模板: calculateOverlap(IoU)
    → best_angle, best_overlap

[Step 2] 超分辨率放大 (×4):
  resize(INTER_CUBIC) × scale_factor (默认4)
  GaussianBlur(3×3, σ=0.3)
  Otsu →
  MORPH_OPEN(3×3) → MORPH_CLOSE(5×5)

[Step 3] 连通域评分 selectBestComponent():
  最小面积过滤: area < 200 (×4空间) → 跳过
  4维评分:
    ┌──────────────┬──────────────────────────────┬────────┐
    │ 指标         │ 公式                         │ 权重   │
    ├──────────────┼──────────────────────────────┼────────┤
    │ 矩形度       │ area / (w * h)               │ 0.25   │
    │ 面积合理性   │ area_ratio ∈ [0.15, 0.6]     │ 0.30   │
    │ 中心距离     │ 1 - dist/roi_radius          │ 0.30   │
    │ 长宽比       │ 1 / (max(w,h)/min(w,h))      │ 0.15   │
    └──────────────┴──────────────────────────────┴────────┘
  总分 = Σ(score × weight)，取最高分

[Step 4] minAreaRect + 角点排序:
  minAreaRect(best_contour) → 4角点
  orderPoints(): TL=min(sum), BR=max(sum), 余下: 大diff=TR, 小diff=BL

[Step 5] 亚像素精化:
  copyMakeBorder(6)
  cornerSubPix(winSize=5×5, zeroZone=-1×-1, TermCriteria(EPS+MAX_ITER,50,0.001))

[Step 6] 象限旋转对齐:
  quadrant = round((360 - best_angle) / 90.0) % 4
  std::rotate(corners, corners + quadrant, corners + 4)
  0°→不转, 90°→转3次, 180°→转2次, 270°→转1次

[Step 7] 缩放回原始: x_out = x / sf, y_out = y / sf

[Step 8] 3D点:
  half_mm = square_size_m * 1000 / 2
  TL = (-half_mm, -half_mm, 0)
  TR = (+half_mm, -half_mm, 0)
  BR = (+half_mm, +half_mm, 0)
  BL = (-half_mm, +half_mm, 0)
```

**双目与单目差异**: 双目额外做左右模板匹配(IoU)确保左右提取的角点对应同一物理目标。单目仅左图提取。

### 5.5 State 4: Dual-ROI 混合

**触发**: class0面积 ≥ 490000 && class1存在

**组件**: 同时使用两个提取器:
- `binary_extractor_` → class0 的边缘角点 (primary)
- `dual_akaze_extractor_` → class1 的中心纹理 (secondary, 不同 scale)

**双目 processDualRoi() 流程**:

```
[Step 0] prepareDualBcTemplate() (一次性, 首次调用执行):
  从 AKAZE 模板灰度图提取 BinaryCorner 角点
  ├─ Otsu 二值化模板灰度图
  ├─ findContours → 最大面积
  ├─ approxPolyDP 二分搜索 → N角点
  ├─ reorderByGeometry → dual_bc_tmpl_corners_
  └─ 3D点: (x/tw)*real_w_mm, (y/th)*real_h_mm, Z=0
  → dual_bc_tmpl_corners_ + dual_bc_tmpl_pts3d_

[Step 1] 裁剪4个子图:
  left_class0 = left_img(primary_roi)
  right_class0 = right_img(primary_roi)
  left_class1 = left_img(secondary_roi_expanded)
  right_class1 = right_img(secondary_roi_expanded)

[Step 2] 并行提取:
  result_bc = binary_extractor_->extract(left_class0, right_class0, ...)
  result_ak = dual_akaze_extractor_->extract(left_class1, right_class1, ...)
  (左右独立的BC + AKAZE含光流/投影/匹配)

[Step 3] 合并:
  // AK的所有点 += secondary→primary的偏移量
  result_ak 所有坐标 += (sec.x - pri.x, sec.y - pri.y)

  // 角度检查 + BC重排 (cpp:1584-1611)
  若 |matched_angle| > 0.5°:
    用 reorderByGeometry 重排 bc_pts3d 副本

  // BC贡献 (cpp:1613-1620):
  for i in result_bc.pts_left_match:
    merged_2d += result_bc.pts_left_match[i]
    merged_3d += dual_bc_tmpl_pts3d_[i]

  // AK贡献 (cpp:1622-1631):
  for each good_match in result_ak:
    merged_2d += result_ak.pts_left_match[i]
    merged_3d += ak_template.pts_3d[trainIdx]

  total_use < 4 → 返回失败

[Step 4] 恢复全图坐标:
  merged_pts_2d += (primary_roi.x, primary_roi.y)

[Step 5] PnP:
  GPnP (双目) 或 MonoPnP (单目)
```

### 5.6 光流追踪器 OpticalFlowTracker

**类**: `OpticalFlowTracker` (include/feature/OpticalFlowTracker.hpp, src/feature/OpticalFlowTracker.cpp)

```
track(left_gray, right_gray, pts_left):
  ├─ calcOpticalFlowPyrLK(L→R)
  ├─ calcOpticalFlowPyrLK(R→L) (反向验证)
  └─ 对每个点:
      FB_error = norm(pt_original - pt_back)
      if FB_error < 1.0px: 保留
  → TrackResult{pts_left_good, pts_right_good, disparity, fb_error, num_valid}
```

**仅用于双目 AKAZE 策略**。单目模式下不调用。

### 5.7 MAD 视差滤波器 MadDisparityFilter

**类**: `MadDisparityFilter` (include/feature/MadDisparityFilter.hpp, src/feature/MadDisparityFilter.cpp)

```
filter(disparity, inlier_mask):
  ├─ median_disparity = median(disparities)
  ├─ MAD = median(|di - median|)
  └─ 保留: |di - median| ≤ 3.0 × MAD
  → MadFilterResult{filtered_points, filter_mask, degraded}
```

**仅用于双目 AKAZE 策略**，在 GPnP 优化前对立体匹配点做离群值剔除。

### 5.8 模板匹配器 TemplateMatcher

**类**: `TemplateMatcher` (include/matching/TemplateMatcher.hpp, src/matching/TemplateMatcher.cpp)

三阶段匹配流水线已在 §5.2 中详细描述。关键参数:

| 参数 | 值 | 含义 |
|------|-----|------|
| Lowe's ratio | 0.75 | dist[0] < 0.75 × dist[1] |
| Homography RANSAC threshold | 5.0 px | 内点判定距离阈值 |
| 描述子距离度量 | NORM_HAMMING | AKAZE 二值描述子 |
| 最低匹配数 | 4 | 低于此值提前返回 |

---

## 6. 位姿解算器

### 6.1 求解器对比

| 特性 | GPnPSolver | InitialPnPSolver | MonoPnPSolver |
|------|-----------|-----------------|---------------|
| 文件 | src/pose/GPnPSolver.cpp | src/pose/InitialPnPSolver.cpp | src/pose/MonoPnPSolver.cpp |
| 算法 | Eigen LM 非线性优化 | OpenCV RANSAC+ITERATIVE | OpenCV EPnP+ITERATIVE |
| 优化变量 | [qx,qy,qz,qw,tx,ty,tz] (7维) | 无初值 | 无初值 |
| 约束 | 重投影 + 立体交叉射线 | 仅重投影 | 仅重投影 |
| 帧间缓存 | ✅ 上帧位姿 warm-start | ❌ | ❌ |
| 适用 | 双目所有策略 (除 TinyTarget) | 双目首帧初值 | 单目全部策略 |
| 最低点数 | gpnp_min_pts (默认3) | — | 4 |

### 6.2 GPnPSolver 详细

**类**: `GPnPSolver` (include/pose/GPnPSolver.hpp)

**核心思想**: 最小化双目交叉射线残差，而非仅最小化重投影误差。

```
残差定义:
  for each matched point pair i:
    P_3d = R * P_model[i] + t           // 模型点到世界坐标
    ray_left = K⁻¹ * [u_left, v_left, 1]  // 左射线方向
    ray_right = K⁻¹ * [u_right, v_right, 1] // 右射线方向

    残差_left = cross(P_3d - origin_left, ray_left)
    残差_right = cross(P_3d - (R_rl⁻¹ * origin_right - R_rl⁻¹ * t_rl), ray_right_rightcam)

  total_residual = [残差_left, 残差_right]  (6×N 维向量)
```

**优化**: Eigen Levenberg-Marquardt (`Eigen::LevenbergMarquardt<Functor>`)

**warm-start**: 上帧位姿作为初值注入 `previous_R_`, `previous_t_`。首帧或缓存失效时用 InitialPnP 初值。

**监控**: 输出 `GPNPMonitor` 包含 initial_cost, final_cost, iterations, failure_reason。

### 6.3 InitialPnPSolver

**流程**:
```
solve(pts_2d, pts_3d, K):
  1. cv::solvePnPRansac(ITERATIVE, 300 iter, 8.0px reproj, 0.99 confidence, no extrinsic guess)
     → inliers < 4 → 失败

  2. cv::solvePnP(ITERATIVE, useExtrinsicGuess=true) on RANSAC inliers
     → 精化

  3. 有效性检查 (见 §6.5)
```

### 6.4 MonoPnPSolver

**流程**:
```
solve(pts_2d, pts_3d, K):
  1. 校验: size >= 4 && 2d.size() == 3d.size()

  2. 转换: Eigen → cv (Point3f, Mat)

  3. cv::solvePnPRansac(EPNP, 300 iter, 8.0px, 0.99, no extrinsic guess)
     → inliers < 4 → 返回 PoseEstimate{fail}

  4. 保存 RANSAC 结果 rvec_ransac / tvec_ransac

  5. cv::solvePnP(ITERATIVE, useExtrinsicGuess=true) on inliers
     抛出异常 → 非致命, 继续用RANSAC结果

  6. ITERATIVE 发散检测：若精化后 |t| > 100000 或 < 10，
     自动回退到 RANSAC EPnP 初值
     → 输出 "[MonoPnP] ITERATIVE 发散，回退到 RANSAC EPnP 结果"

  7. 有效性检查 (见 §6.5)
     → PoseEstimate{R, t, success, num_points}
```

> **ITERATIVE 发散回退**（新增）：极小 ROI（~30px²）下 2D 点高度聚集而 3D 点跨度大，深度估计病态化，ITERATIVE 精化可能发散到 `|t| → ∞`。此机制在发散时自动回退到 RANSAC 的有效结果，避免整帧被标记为 FAILED。

### 6.5 位姿有效性校验 (所有求解器共用)

```cpp
// 通过才标记 success=true
1. t.z > 0                    // 相机必须在目标前方
2. 10 < norm(t) < 100000       // 深度在合理范围 (mm)；MonoPnP 上限 100m，双目 GPnP 上限 20m
3. R, t 各分量 isfinite       // 无 NaN/Inf
4. num_points >= min_pts      // 有效点数足够
```

### 6.6 PnP 路由 (dispatchPnP)

```cpp
// StereoTracker.cpp
std::pair<bool, PoseEstimate> dispatchPnP(FeatureExtractor* ext, PipelineResult& result, bool is_first) {
    std::string n = ext->name();
    if (n == "AKAZE_GPNP")     return runAkazePnP(result, is_first);
    if (n == "BinaryCorner")   return runBinaryCornerPnP(result, is_first);
    if (n == "TinyTarget")     return runTinyTargetPnP(result);
}
```

**TinyTarget 特殊处理**: 仅用 `solvePnP(ITERATIVE)` 不经过 RANSAC 也不做 warm-start。

### 6.7 MAD 滤波在 PnP 中的应用

仅在 `runAkazePnP()` 中：PnP 前对立体匹配点做 MAD 视差滤波，剔除异常值后再传入 GPnP:

```cpp
auto mad_result = mad_filter_.filter(disparity, inlier_mask);
// 仅用 mad_result.filtered_points 做 GPnP 优化
```

### 6.8 ESKF 多源信息融合 (适配层)

> **库**: `eskf/eskf_vio.hpp` (header-only, 第三方, **不改动**) — `eskf::ESKF_VIO` 误差状态卡尔曼 (16维名义态 `[p,v,q,b_a,b_g]` + 15×15 误差协方差, SI 单位), `eskf::RadarAltimeter` (跳变+NIS 检验), `eskf::GravityEstimator` (在线重力, 第一版未启用)
> **适配层**: `fusion::EskfFusionManager` (include/fusion/EskfFusionManager.hpp + src/fusion/EskfFusionManager.cpp)
> **合成源**: `input::SimulatedImu/SimulatedRadar` (include/input/SimulatedSensors.hpp, main.cpp 驱动, 不改 InputProvider)

**数据流** (main.cpp normal 模式, `eskf.enabled=true` 时):

```
getNextPacket → t_cam = timestamp_us×1e-6
  ├─ sim_imu.generate(t_prev, t_cam, R_cam_w)   → fusion.feedImu(t, acc, gyro)
  ├─ sim_radar.generate(t_prev, t_cam, height)  → fusion.feedRadar(t, h)
  ├─ fusion.propagateTo(t_cam)     // 排干 IMU 逐样本 predict + 雷达 validate/update
  ├─ processFrame → PipelineResult
  └─ fusion.feedCameraPose(t_cam, R, t_mm, success)
      └─ 内部: propagateTo → 位姿转换 → lazy init 或 update_camera_pose_hybrid
输出: [Frame N] ESKF p=[..]m v=[..] q=[..] | PnP t(mm)=[..]  (融合为主输出)
```

**位姿转换** (世界系=模板系, Z 向上): PnP 输出 `R(模板→相机), t(mm)` →
`R_cam_w = R_template_world·Rᵀ`, `p_cam_w = R_cam_w·(-t/1000)` (m), 姿态四元数 `[w,x,y,z]`。

**关键接口**:
- `feedImu(t, acc, gyro)` / `feedRadar(t, height)` — 仅入缓冲 (IMU ~2s / 雷达 ~1s 上限, 溢出丢最旧)
- `propagateTo(t)` — 排干式对齐: 逐样本 `predict_adaptive(acc, gyro, dt)` 传播; 雷达逐样本 `RadarAltimeter::validate` + `update_altitude`
- `feedCameraPose(t, R, t_mm, valid)` — 内部先 `propagateTo(t)`; `valid=false` (视觉失败) 仅惯性传播; 间隔 > `max_cam_gap_s` 自动 `reset()` 重新 lazy init
- 输出: `position()/velocity()/rotation()/quaternion()/stats()` (`Stats{imu_samples, cam_updates, cam_ignored, cam_rot_skipped, radar_accepted, radar_rejected}`)

**陷阱**:
1. ⚠️ 世界系=模板系 Z 向上假设; 实际朝向不同用 `eskf.R_template_world` 修正
2. ⚠️ 单位: PnP `t` 为 mm, 适配层内部 ÷1000 转 m; 雷达高度 m
3. ⚠️ debug 模式 (固定2帧) 无时间序列, ESKF 自动禁用 (main.cpp 打警告)
4. ⚠️ `lazyInit` 后 `last_prop_t_ = -1` 重新锚定: 初始化时刻之前的 IMU 样本已排干, 下一个样本作积分起点
5. ⚠️ 合成源仅悬停近似 (比力 = -g 旋转到相机系 + 白噪声 + 偏置), 用于链路验证, 非物理仿真
6. ⚠️ 相机更新走 `update_camera_pose_hybrid` (生产路径): 位置永远更新 + 姿态 NIS 自洽 (×rot_detect_scale=4.0 < χ²₉₅) 才追加 — 坏帧被 FDI 拒掉是预期行为

### 6.9 延迟测量反向传播与退化监控 (方案B 核心)

> 相机位姿是 **t0 曝光时刻** 拍的, 但 PnP 处理完送达时已是 **t1**。反向传播把延迟测量贴回真实时刻, 避免"拿 t0 位姿纠 t1 状态"的延迟误差。

**反向传播 (IMU 重放)**:
- `feedCameraPose(const CameraObservation&)` — 显式区分 `t_exposure`(t0) / `t_arrival`(t1)
- 内部: 若 `latency > 1ms`, 回退到 ≤ t0 的最近状态快照 (`StateSnap`, **ESKF_VIO 整体拷贝**), 重放 IMU 到 t0 → 在 t0 应用 `update_camera_pose_hybrid` → 重放 IMU 到当前 → 重建历史
- 快照按 `state_hist_hz`(默认 100Hz) 记录, 窗口 = `backprop_window_s`(默认 0.2s)

**协方差膨胀兜底** (延迟超窗):
- `latency_fallback = Inflate`(默认): `cam_pos_noise² += σ_pos_rw²·Δt`, `cam_rot_noise² += σ_gyro²·Δt`, 按到达时刻应用 (一阶有界近似)
- `latency_fallback = Reject`: 直接丢弃该观测

**退化监控**:
- `getLatestState() → FusionState{position, velocity, quaternion, cov_trace, quality, last_cam_t}`
- `FusionQuality`: `Normal` → `Degraded`(相机丢失 > `max_output_age_s`) → `Stale`(> `max_cam_gap_s`)
- `cov_trace` = 位置协方差迹, 作为可信度信号

**线程化 (Phase 4, `eskf.threaded: true`)**:
- `start()` 启动内部融合工作线程; `feedImu/feedRadar/feedCameraPose` 异步入队 (mutex + condition_variable)
- 融合线程独立: 按到达序处理相机观测 (内部 `propagateInternal` + lazyInit/反向传播), 无相机时把已缓冲 IMU/雷达传播到最新 (相机缺席状态继续跑)
- `getLatestState()`/`stats()` 用 `mtx_` 加锁 (线程安全); 旧 `position()/velocity()/rotation()/quaternion()/initialized()` 为同步 API, 线程化模式下勿用
- `stop()` 停线程并 join (析构自动); 未 `start()` 时保持单线程同步语义 (向后兼容)

**陷阱**:
1. ⚠️ 反向传播仅重放 IMU, **不重放雷达** — 回退窗口内已应用的雷达更新在回退后丢失 (20Hz、Z 轴单维, 影响小, 已知简化)
2. ⚠️ 无延迟 (`latency ≤ 1ms`) 时**不走**反向传播, 按到达时刻直接更新, 与旧同步路径一致
3. ⚠️ 线程化输出有一拍异步 (worker 尚未消费完当帧相机观测时, `getLatestState()` 返回前一状态)
4. ⚠️ 反向传播历史内存 ≈ 100Hz × 0.2s × ~470 double ≈ 150KB, 可忽略

---

## 7. 输入系统

> **输入分类**：输入按数据来源明确分为两大类：
> - **实时相机输入** — `input_system.image.type: "camera"`（CameraSource，Phase 3 已实现）
> - **离线图像输入** — `input_system.image.type: "file" | "directory" | "sequence"`，以及 Debug 模式的 `cv::imread` 固定图像（此时 `input_system` 配置不需要）

### 7.1 架构

```
InputProvider (统一协调器)
├── 持有 IStereoImageSource 实例 (图像源抽象)
├── 可选 TimeSyncUnit (IMU + 高度计融合, Phase 2 预留)
└── 输出: SensorPacket { left, right, timestamp, IMU(opt), height(opt), valid }
```

### 7.2 四种图像源

| 分类 | type | 类 | 文件 | 用途 | 配置 |
|------|------|-----|------|------|------|
| 离线 | `"file"` | `FileStereoSource` | src/input/FileStereoSource.cpp | 静态双目文件对 | `input_system.image.left_path/right_path` |
| 离线 | `"directory"` | `DirectoryStereoSource` | src/input/DirectoryStereoSource.cpp | 双目编号图像序列 | `input_system.image.directory_path` + `left_pattern/right_pattern` |
| 离线 | `"sequence"` | `SequenceSource` | src/input/SequenceSource.cpp | 单目图像序列 | `input_system.image.directory_path` + `sequence_pattern` |
| 实时 | `"camera"` | `CameraSource` | src/input/CameraSource.cpp | **实时摄像头 (Phase 3, 已实现)** | `input_system.image.camera_devices` + `target_fps` |

> **CameraSource 要点** (src/input/CameraSource.cpp):
> - **运行方式**: `./build/Steretracker config/tracker_config_webcam.json`（type=camera + mono_mode=true 的完整示例配置，见 §8.2）
> - OpenCV `VideoCapture` 打开设备: 纯数字 `"0"` → 索引模式; `/dev/video0` → 路径模式
> - `nextFrame()` 阻塞在 `cap_->grab()` 上 (天然按摄像头帧率节流); `target_fps>0` 时额外 sleep 限速; **时间戳在 grab 返回后立即打戳** (接近曝光时刻)
> - 单目语义: 右图 = 左图 `clone()` (与 SequenceSource 一致); `totalFrames() = -1`
> - 打开后 `grab()` 预热 5 帧 (等自动曝光/白平衡收敛)
> - `camera_devices` 支持 `"0;1"` 多设备格式, 当前只取第一个 (USB 双目预留)
> - **线程化采集 (Phase 3.1)**: Camera 类型自动启用 (`InputProvider::initialize`), 采集线程产帧入 `RingBuffer` (容量 `ring_capacity`, 默认4), 主循环 `getNextPacket()` 用 **take-latest** 语义取最新帧——处理快时阻塞等帧 (等效同步), 处理慢时跳过旧帧 (延迟有界、丢帧可统计)。统计接口 `stats()` 返回 captured/consumed/dropped; `shutdown()` 停止并 join 采集线程
> - **优雅退出 (Phase 3.2)**: main.cpp 注册 SIGINT/SIGTERM → 帧循环收尾 → 打印运行汇总 (帧数/成功率/实际FPS/输入统计) → `log_file` 可正常落盘

### 7.3 SensorPacket

```cpp
struct SensorPacket {
    int64_t timestamp_us;
    cv::Mat left_image;
    cv::Mat right_image;
    std::optional<ImuPacket> imu;        // Phase 2 预留
    std::optional<HeightPacket> height;  // Phase 2 预留
    bool valid = false;
};
```

### 7.4 Phase 规划

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1 | ✅ 完成 | 图像源抽象 + RingBuffer |
| Phase 2 | 占位 | IMU + 高度计 + TimeSyncUnit (extracted_input_system/ 提供 CanSocket/TimeSyncUnit 参考实现, 未接入) |
| Phase 3 | ✅ 完成 | 实时摄像头源 (CameraSource) |
| Phase 3.x | ✅ 完成 | 线程化采集 (采集线程 + RingBuffer take-latest) + 丢帧/FPS 统计 + Ctrl+C 优雅退出 |

### 7.5 独立输入系统模块

`extracted_input_system/` 目录包含独立可复用的输入系统代码，有独立的 CMakeLists.txt 和 ARCHITECTURE.md。

---

## 8. 配置文件完全指南

### 8.1 配置文件路径

默认: `config/tracker_config.json` (运行时可通过命令行参数指定)

### 8.2 完整配置项

```jsonc
{
    // ========== 顶层 ==========
    "mode": "normal",              // "normal" | "debug"
    "mono_mode": false,            // true = 单目, false = 双目

    // ========== 相机参数 ==========
    "camera": {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 640.0,
        "cy": 512.0,
        "baseline_mm": 120.0,
        // 可选: rotation_matrix (3×3, 默认I)
    },

    // ========== Normal 模式: 输入系统 ==========
    "input_system": {
        "max_frames": 0,              // <=0 = 无限 (实时摄像头用)
        "image": {
            "type": "directory",      // "file" | "directory" | "sequence" | "camera"
            "left_path": "",          // type=file: 左图路径
            "right_path": "",         // type=file: 右图路径
            "directory_path": "data/大图/",   // type=directory/sequence: 目录
            "left_pattern": "left",   // type=directory: 左图前缀
            "right_pattern": "right", // type=directory: 右图前缀
            "sequence_pattern": "frame",      // type=sequence: 单目序列前缀
            "camera_devices": "0",    // type=camera: 设备索引 "0" 或路径 "/dev/video0"; 支持 "0;1" (取第一个)
            "target_fps": 10          // type=camera: 目标帧率, 0=不限制
        },
        "imu": { "enabled": false, "port": "/dev/ttyUSB0", "baud_rate": 921600, "protocol": "custom" },
        "altimeter": { "enabled": false, "can_interface": "can0", "type": "can" }   // Phase 2, Linux only
    },
    // 实时摄像头完整示例: config/tracker_config_webcam.json (type=camera + mono_mode=true)

    // ========== Debug 模式: 手动图像路径 ==========
    "input": {
        "left_path": "data/大图/im0.png",
        "right_path": "data/大图/im1.png"
    },

    // ========== 手动 ROI (Debug模式) ==========
    "manual_roi": {
        "enabled": false,
        "left":  {"x": 100, "y": 100, "width": 400, "height": 400},
        "right": {"x": 100, "y": 100, "width": 400, "height": 400}
    },

    // ========== YOLO 配置 ==========
    "yolo": {
        "model_path": "best.onnx",
        "device_type": "Auto",         // "Auto" | "CPU" | "CUDA"
        "conf_threshold": 0.5,
        "target_class_id": 0,          // class 0 = 整体
        "roi_expand_ratio": 0.1,       // ROI外扩比例
        "roi_min_size": 50,            // ROI最小尺寸 (px)
        "intra_op_threads": 4          // ONNX线程数
    },

    // ========== 策略参数 ==========
    "strategies": {
        // 面积阈值 (⚠️ 占位值, 需重新标定)
        "akaze_min_area": 40001,       // State 2/3 分界
        "tiny_max_area": 800,          // State 1/2 分界

        // AKAZE 策略
        "akaze_gpnp": {
            "template_path": "data/NewMuBan(reordered)/",
            "real_w": 200.0,           // 模板物理宽度 mm
            "real_h": 150.0,           // 模板物理高度 mm
            "scale": 0.5,              // 检测前降采样
            "min_pts": 3,              // GPnP最低点数
            "use_initial_pnp": true,   // 首帧使用InitialPnP
            "mad_sigma": 3.0           // MAD滤波σ系数
        },

        // BinaryCorner 策略
        "binary_corner": {
            "corners": 10,             // 目标角点数
            "kernel_size": 3,          // 形态学核
            "scale": 1.0,              // 提取前放大
            "target_size": 100,        // 标准化尺寸
            "pixel_to_meter_scale": 0.5,
            "roi_pad_pixels": 0,
            "otsu_ratio": 1.3
        },

        // TinyTarget 策略
        "tiny_target": {
            "target_size": 50,         // 标准化尺寸
            "scale_factor": 4.0,       // 超分倍数
            "square_size_m": 0.05,     // 目标边长 m
            "roi_pad_pixels": 0
        },

        // Dual-ROI 策略
        "dual_roi": {
            "trigger_area": 490000,    // 触发面积 (700×700)
            "secondary_expand_pixels": 10,  // class1 ROI外扩
            "akaze_scale": 0.5         // class1 AKAZE scale
        },

        // State 5 回退
        "close_range": {
            "enabled": true,
            "class1_min_area": 100,
            "roi_expand_ratio": 0.1,
            "min_expand_pixels": 10,
            "akaze_min_area": 40001,     // class1-only 专用 AKAZE 阈值 (0=使用默认)
            "tiny_max_area": 400          // class1-only 专用 TinyTarget 阈值 (0=使用默认)
        }
    },

    // ========== 可视化 ==========
    "output": {
        "visualize": true,              // 是否保存可视化图像
        "log_file": false               // Normal 模式是否输出 TXT 日志文件
    },
}
```

### 8.3 模式与配置必填字段对照

| 字段 | Normal 单目 | Normal 双目 | Debug 单目 | Debug 双目 |
|------|:-----------:|:-----------:|:----------:|:----------:|
| mode | normal | normal | debug | debug |
| mono_mode | true | false | true | false |
| camera | ✅ | ✅ | ✅ | ✅ |
| input_system | ✅ | ✅ | ❌ | ❌ |
| input | ❌ | ❌ | ✅ | ✅ |
| manual_roi | ❌ | ❌ | 可选 | 可选 |
| yolo | ✅ | ✅ | 可选 | 可选 |
| strategies | ✅ | ✅ | ✅ | ✅ |

---

## 9. 关键实现细节与陷阱

### 9.1 ⚠️ 陷阱1: Dual-ROI 完全脱离策略链 (EARLY RETURN)

```cpp
// processMono() 中 (cpp:1796-1800)
if (left_group && left_group->is_dual) {
    result = processDualRoiMono(left_img, *left_group, visualize);
    state_.frame_count++;
    return result;  // <-- EARLY RETURN, 不执行 configureStrategyChain!
}
```

**影响**:
- Dual-ROI 不调用 `configureStrategyChain()`
- 不进入退化后备链
- 独立使用 `binary_extractor_` + `dual_akaze_extractor_` (不是 `akaze_extractor_`)
- BC 提取失败直接返回失败，不降级

### 9.2 ⚠️ 陷阱2: 单目模式无 warm-start

单目全部使用 `MonoPnPSolver`，该求解器**每帧独立**，不缓存上帧位姿。

```cpp
// 双目有: state_.R_prev, state_.t_prev → GPnP warm-start
// 单目无: pose = mono_pnp_.solve(pts_2d, pts_3d, K)  // 无初始猜测
```

### 9.3 ⚠️ 陷阱3: 单目模式无光流、无立体投影、无MAD

单目 `processMono()` 调用各提取器的 `extractMono()`，其内部：
- ❌ 不执行光流追踪
- ❌ 不执行立体投影
- ❌ 不执行 MAD 视差滤波
- ❌ `PipelineResult` 中 `pts_right_*` 全部为空

### 9.4 ⚠️ 陷阱4: YOLO 无检测 → skip 帧 (不触发退化)

```cpp
// main.cpp 帧循环中
RoiGroup lg, rg;
if (yolo_ok) {
    std::tie(lg, rg) = yolo_provider.detect(left, right);
} else {
    // YOLO 失败
}
// ...
auto* plg = lg.valid() ? &lg : nullptr;
auto* prg = rg.valid() ? &rg : nullptr;
result = tracker.process(left, right, visualize, plg, prg);
```

当 `lg` 和 `rg` 都无效时，`process()` 内部可能使用 `nullptr` → 全图回退或空结果。**YOLO 无检测不会触发策略退化链**——退化链仅在特征提取失败时触发。

### 9.5 ⚠️ 陷阱5: Class 1 的语义不同于 Class 0

| class_id | 语义 | 用途 |
|----------|------|------|
| 0 | 目标整体 | primary ROI (所有策略) |
| 1 | 目标中心 | secondary ROI (Dual-ROI/State 5) |

- State 1~3 忽略 class 1
- 仅当 class 0 面积 ≥ 490000 且存在 class 1 时进入 State 4
- class 1 回退 (State 5) 仅在 class 0 消失时触发

### 9.6 ⚠️ 陷阱6: Dual-ROI 中 BC 与 AK 的偏移处理

```cpp
// BC 在 primary ROI (class 0) 中提取 → 坐标相对于 primary_roi
// AK 在 secondary ROI (class 1) 中提取 → 坐标相对于 secondary_roi
// 合并前需要将 AK 坐标偏移:
ak_pts += (secondary_roi.x - primary_roi.x, secondary_roi.y - primary_roi.y)

// 合并后再统一偏移到全图:
merged_pts += (primary_roi.x, primary_roi.y)
```

### 9.7 ⚠️ 陷阱7: prepareDualBcTemplate() 仅执行一次

```cpp
void StereoTracker::prepareDualBcTemplate() {
    if (dual_bc_template_ready_) return;  // <-- 一次性
    // ... 从 AKAZE 模板灰度图提取 BC 角点
    dual_bc_template_ready_ = true;
}
```

修改 AKAZE 模板后需手动重置 `dual_bc_template_ready_`。

### 9.8 ⚠️ 陷阱8: 坐标空间约定

全系统有两个坐标空间:

| 空间 | 范围 | 何时使用 |
|------|------|---------|
| ROI 局部坐标 | [0, roi_width) × [0, roi_height) | 特征提取器内部 |
| 全图坐标 | [0, img_width) × [0, img_height) | PnP 输入、可视化 |

**转换发生在**:
- `process()`/`processMono()` 中提取后: `pts_left_match += cv::Point2f(roi.x, roi.y)`
- `processDualRoi()` 中合并后: `merged_pts += (primary_roi.x, primary_roi.y)`

**注意**: `PipelineResult` 产出时各字段是 ROI 局部坐标！调用方负责转换。

### 9.9 ⚠️ 陷阱9: ROI padding 影响策略判定

`binary_roi_pad_` 和 `tiny_roi_pad_` 会影响提取器的 ROI 尺寸，但 **策略链选择使用原始 ROI 面积**，不是 padding 后的。

### 9.10 ⚠️ 陷阱10: BC/TT 的 is_class1 标志

```cpp
// 当 State 5 回退时，BC/TT 提取器被告知 is_class1=true
// 3D 点尺寸计算使用 class 1 的物理尺寸而非 class 0
// 见 TinyTargetExtractor: square_size_m 可能不同
```

### 9.11 提取器 vs 策略对应关系

| 成员变量 | 类型 | 用途 |
|----------|------|------|
| `akaze_extractor_` | `unique_ptr<AkazeGpnpExtractor>` | State 3 主策略 (含光流+投影) |
| `dual_akaze_extractor_` | `unique_ptr<AkazeGpnpExtractor>` | State 4 class1 部分 (不同 scale) |
| `binary_extractor_` | `unique_ptr<BinaryCornerExtractor>` | State 2 主策略 + State 4 class0 部分 |
| `tiny_extractor_` | `unique_ptr<TinyTargetExtractor>` | State 1 主策略 + 退化后备 |

> `akaze_extractor_` 和 `dual_akaze_extractor_` 是**两个不同的实例**，参数不同（scale, min_pts 等）。Dual-ROI 用的是 `dual_akaze_extractor_`。

### 9.12 退化全景 ⚠️

系统中共有 **13 类退化/fallback 机制**，按触发层级从高到低排列：

#### 一、策略主退化链（最高层，跨策略降级）

由 `configureStrategyChain()` 每帧配置，`process()`/`processMono()` 中的 fallback 循环驱动。退化方向始终**单向**，不可逆。

```
AKAZE_GPNP (State 3) ──失败──→ BinaryCorner (State 2) ──失败──→ TinyTarget (State 1) ──失败──→ 终止（无位姿输出）
```

| 退化步骤 | 触发条件 | 降级目标 |
|---------|---------|---------|
| AKAZE → BC | `success == false` 或 `n_kp_left < 3` | BinaryCorner |
| BC → TT | 同上 | TinyTarget |
| TT → 终止 | 同上 | 帧输出失败，无位姿 |

> 退化不跨面积区间折返（如 TT 失败不会回到 BC）。

#### 二、Dual-ROI (State 4) — 独立路径，不参与退化链

```
Dual-ROI (BC + AK 并行提取) ──任一失败──→ 直接返回失败（不降级）
```

| 退化情况 | 触发条件 | 行为 |
|---------|---------|------|
| BC 提取失败 | `result_bc.success == false` | 整体失败，不 fallback |
| AK 提取失败 | `result_ak.success == false` | 整体失败，不 fallback |
| 合并点数不足 | `total_use < 4` | 整体失败，不 fallback |
| BC 模板角度偏差 | `|matched_angle| > 0.5°` | 用 `reorderByGeometry` 重排 3D 点（修正，不算退化） |

> 代码证据：`processMono()` 中对 Dual-ROI 走 **EARLY RETURN**，不调用 `configureStrategyChain()`，也不进入 fallback 循环。

#### 三、State 5 极近 — class1 回退重分类

```
class0 丢失 + class1 存在 + 面积 ≥ class1_min_area
    → 用 class1 生成 ROI → 按面积重新判定 State 1~4 → 走对应策略链
```

| 退化情况 | 触发条件 | 行为 |
|---------|---------|------|
| class0→class1 回退 | 无 class0，有 class1，面积 ≥ `class1_min_area` | 重分类后走标准策略链 |
| class1 面积不足 | 面积 < `class1_min_area` | 不回退，直接 skip 帧 |

> 回退时 BC/TT 的 `is_class1=true` 影响 3D 点物理尺寸计算。

#### 四、YOLO 检测层 — 帧级 skip（非退化，不触发策略链）

| 情况 | 触发条件 | 行为 |
|------|---------|------|
| 无任何检测 | 无 class0 且无 class1 | **skip 帧** — 不触发任何策略 |
| 模型加载失败 | ONNX 初始化失败 | 整个跟踪不可用 (`Status::ModelLoadFailed`) |
| 推理失败 | 单帧 ONNX 异常 | 该帧无 ROI (`Status::InferenceFailed`) |

> YOLO 无检测 **不是退化**，不进入策略退化链。退化链仅在 "有 ROI 但特征提取失败" 时触发。

#### 五、AKAZE 模板匹配 — 三阶段子退化

`TemplateMatcher::match()` 内部三阶段：

```
Stage 1 (Ratio Test, Lowe=0.75) ──<4 matches──→ 直接失败
Stage 2 (Cross-Check)           ──<4 matches──→ 直接失败
Stage 3 (Homography RANSAC, 5.0px) ──H为空──→ 回退到 Stage 2 结果（不算完全失败）
```

#### 六、BinaryCorner — 内部子退化

| 子退化 | 触发条件 | 降级行为 |
|--------|---------|---------|
| 连通域过少 | `connectedComponentsWithStats` → `num_labels ≤ 1` | 保留原始二值图，不做连通域筛选 |
| 旋转回退 | `warpAffine(INTER_CUBIC)` 旋转灰度图 + Otsu + floodFill 失败 | 降级到 `warpAffine(INTER_NEAREST)` 旋转二值图 + 纯形态学 |
| 角点数不精确 | `approxPolyDP` 8 次二分搜索未命中目标角点数 | 取 `best_diff` 最近似的多边形 |
| 左右模板匹配容差 | 左图匹配角度在右图搜索 | 15° 容差范围搜索 |

#### 七、TinyTarget — 内部子退化（退化链终点，无后备）

| 子退化 | 触发条件 | 行为 |
|--------|---------|------|
| 连通域面积过滤 | `area < 200`（×4 超分空间） | 跳过该连通域 |
| 连通域评分 | 4 维加权评分（矩形度 0.25 + 面积 0.30 + 中心距 0.30 + 长宽比 0.15）仅取最高分 | 无合格 → `success=false` → 触发策略链终止 |
| 策略链终点 | TT 失败 | **无 further fallback**，输出空位姿 |

#### 八、光流追踪 — 点级退化（仅双目 AKAZE）

| 退化 | 触发条件 | 行为 |
|------|---------|------|
| FB 校验失败 | `Forward-Backward error ≥ 1.0px` | 该点被丢弃 |
| 全部点丢弃 | 所有点 FB 失败 | `num_valid = 0`，AKAZE 立体匹配失败 |

#### 九、MAD 视差滤波 — 点集退化（仅双目 AKAZE）

| 退化 | 触发条件 | 行为 |
|------|---------|------|
| 离群值剔除 | `|disp_i - median| > 3.0 × MAD` | 该点从 GPnP 输入中移除 |
| 点集退化标记 | 大量点被剔除 | `MadFilterResult::degraded = true` |

> 调用位置：`runAkazePnP()` 中 GPnP 优化前。

#### 十、立体投影 — 点级退化

| 退化 | 触发条件 | 行为 |
|------|---------|------|
| 深度异常 | `disparity ≤ 0` 或投影后超出图像边界 | `valid_mask[i] = false`，该点丢弃 |

#### 十一、PnP 求解器退化

| 求解器 | 退化步骤 | 触发条件 | 降级行为 |
|--------|---------|---------|---------|
| InitialPnPSolver | RANSAC 失败 | `inliers < 4` | GPnP 用深度估算初值 `depth = f·b/median_disp, clamp[50,5000]` |
| InitialPnPSolver | ITERATIVE 精化失败 | 精化后位姿不合法 | 仍用 RANSAC 结果 |
| GPnPSolver | warm-start 失效 | 首帧或 `has_cache == false` | 用 InitialPnP 初值 |
| GPnPSolver | LM 不收敛 | Eigen LM 未收敛 | 可能仍输出位姿（取决于监控信息） |
| GPnPSolver/MonoPnPSolver | 位姿校验失败 | `t.z ≤ 0` / `|t| ∉ [10, 20000]` / NaN/Inf | `success = false` |
| MonoPnPSolver | EPnP RANSAC 失败 | `inliers < 4` | 返回 `PoseEstimate{success=false}` |
| MonoPnPSolver | ITERATIVE 精化异常 | `solvePnP` 抛出异常 | **非致命**，继续用 RANSAC 结果 |
| MonoPnPSolver | ITERATIVE 发散 | `\|t\| > 100000` mm | 自动回退到 RANSAC EPnP 结果（不标记 FAILED） |
| TinyTarget solvePnP | 无 RANSAC，无 warm-start | ITERATIVE 失败 | 直接 `gpnp_success = false` |

#### 十二、ROI 输入层退化

| 情况 | 行为 |
|------|------|
| ROI 为 nullptr（`RoiGroup::valid() == false`） | `roi_area = 0` → 走 AKAZE 链（全图回退） |
| Debug 手动 ROI 无效（尺寸 ≤ 0） | 回退到 YOLO 检测（如果可用） |

#### 十三、Dual-ROI BC 模板退化

| 退化 | 触发条件 | 行为 |
|------|---------|------|
| 模板未就绪 | 首次调用 `prepareDualBcTemplate()` | 现场从 AKAZE 模板灰度图提取 BC 角点（一次性） |
| 灰度图二值化无连通域 | Otsu 后无有效区域 | `dual_bc_template_ready_` 为 true 但角点为空的边缘情况 |

#### 退化全景图

```
                    YOLO 无检测 → SKIP 帧 (非退化)
                         │
                    YOLO 有检测
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    有 class0       无 class0       无 class0
    有/无 class1    有 class1       无 class1
         │               │               │
    按面积分级    State 5 回退    SKIP 帧 (非退化)
         │         (重分类→State1~4)
         │
   ┌──┬──┼──┬────┐
   │  │  │  │    │
  St1 St2 St3  St4 (BC∥AK 独立路径)
   │  │  │  │    │
   TT BC AKAZE  BC∥AK ──任一失败→终止
   │  │  │  (不参与退化链)
   │  │  │
   │  │  ├─RatioTest<4→失败
   │  │  ├─CrossCheck<4→失败
   │  │  ├─Homography空→回退Stage2
   │  │  ├─光流FB≥1px→丢弃点
   │  │  ├─MAD滤波→剔除离群点
   │  │  └─立体投影异常→丢弃点
   │  │
   │  ├─连通域≤1→保留原图
   │  ├─旋转回退→INTER_NEAREST
   │  └─approxPolyDP→best_diff
   │
   ├─连通域<200→跳过
   └─评分无合格→失败(终止)
         │
    ──→ PnP 求解器层退化 (RANSAC/LM/校验)
```

#### 退化设计关键特征总结

| 特征 | 说明 |
|------|------|
| **退化方向** | 始终单向：AKAZE → BC → TT，不可逆 |
| **Dual-ROI 隔离** | State 4 完全独立，不参与退化链，失败直接终止 |
| **YOLO vs 特征退化分离** | YOLO 无检测 = skip 帧（不触发策略链）；特征提取失败 = 策略退化；两者互不触发 |
| **单目退化面窄** | 单目无光流、无立体投影、无 MAD、无 warm-start — 比双目退化面缩窄约 5 类 |
| **子退化不跨模块** | BC 内部旋转回退不影响策略链选择；AKAZE 匹配阶段回退不改变策略 |
| **终止即终止** | TT 失败后无进一步 fallback，输出空位姿 |
| **永不成环** | 任何退化路径都不会形成循环——退化图是有向无环图 (DAG) |

---

## 10. 文件索引速查

### 10.1 入口与构建

| 文件 | 行数 | 关键内容 |
|------|------|---------|
| `main.cpp` | ~272 | 完整程序入口：配置解析、模式分发、帧循环 |
| `CMakeLists.txt` | — | 构建配置，依赖 OpenCV/Eigen/ONNX Runtime |

### 10.2 核心类型与配置

| 文件 | 关键类型/函数 |
|------|-------------|
| `include/common/Types.hpp` | 全部强类型: PipelineResult, TemplateData, StereoCameraParams, TrackerConfig, RoiRect, RoiGroup, PoseEstimate, LogEntry, TrackingState, Detection, YoloConfig... |
| `include/common/Config.hpp` | 工厂函数: makeTrackerConfig(), makeYoloConfig(), makeStereoCameraParams() |
| `include/common/GeometryUtils.hpp` | 内联几何工具函数 |
| `include/common/LogConfig.hpp` | 全局日志开关 g_verbose_console |

### 10.3 Tracker 核心

| 文件 | 关键方法 | 行号参考 |
|------|---------|---------|
| `include/tracker/TrackerBase.hpp` | initExtractors(), configureStrategyChain(), validateRoi(), finalizePose() | — |
| `src/tracker/TrackerBase.cpp` | 上述方法的实现 | — |
| `include/tracker/StereoTracker.hpp` | process(), processMono(), processDualRoi(), processDualRoiMono(), dispatchPnP() | — |
| `src/tracker/StereoTracker.cpp` | **全部核心逻辑** (~2000行) | ctor:~30-90, configureStrategyChain:94-122, process:~480, processDualRoi:~1300, processDualRoiMono:1513-1771, processMono:1779-1954, prepareDualBcTemplate:954-1038, dispatchPnP:~200, runAkazePnP:~230, runBinaryCornerPnP:~350, runTinyTargetPnP:~430 |

### 10.4 特征提取

| 文件 | 关键方法 |
|------|---------|
| `include/feature/FeatureExtractor.hpp` | 基类, StrategyType 枚举 |
| `include/feature/AkazeExtractor.hpp` + `.cpp` | extract(), extractTemplate() — 纯AKAZE检测 |
| `include/feature/AkazeGpnpExtractor.hpp` + `.cpp` | extract() (光流+投影+匹配), extractMono(), setTemplateData() |
| `include/feature/BinaryCornerExtractor.hpp` + `.cpp` | extract(), extractMono(), extractFromBinary(), extractCornersFromContour(), extractLargestContour(), keepLargestRegion(), fillHoles(), smoothBoundary(), findBestMatch(), matchCorners(), reorderByGeometry() |
| `include/feature/TinyTargetExtractor.hpp` + `.cpp` | extract(), extractMono(), extract4Corners(), matchTemplate(), selectBestComponent(), refineCorners(), orderPoints() |
| `include/feature/OpticalFlowTracker.hpp` + `.cpp` | track() — LK光流+FB校验 |
| `include/feature/MadDisparityFilter.hpp` + `.cpp` | filter() — MAD离群值剔除 |

### 10.5 位姿解算

| 文件 | 关键方法 |
|------|---------|
| `include/pose/GPnPSolver.hpp` + `.cpp` | solve() — Eigen LM 7维优化，交叉射线残差 |
| `include/pose/InitialPnPSolver.hpp` + `.cpp` | solve() — RANSAC PnP + ITERATIVE精化 |
| `include/pose/MonoPnPSolver.hpp` + `.cpp` | solve() — EPnP RANSAC + ITERATIVE精化 + 有效性校验 |

### 10.6 立体视觉

| 文件 | 关键方法 |
|------|---------|
| `include/stereo/StereoProjector.hpp` + `.cpp` | project() — 视差→深度→右图重投影 |

### 10.7 模板匹配

| 文件 | 关键方法 |
|------|---------|
| `include/matching/TemplateMatcher.hpp` + `.cpp` | match() — 三阶段: RatioTest→CrossCheck→HomographyRANSAC |

### 10.8 检测

| 文件 | 关键方法 |
|------|---------|
| `include/detection/YoloDetector.hpp` | detect() — ONNX推理 (header-only, ~424行) |
| `include/detection/YoloRoiProvider.hpp` + `.cpp` | detect() (双目), detectMono() (单目) |
| `include/detection/RoiGenerator.hpp` + `.cpp` | generate(), generateGroup(), generateStereoGroup(), detectionToRoi(), normalizeStereoPair() |

### 10.9 可视化

| 文件 | 关键方法 |
|------|---------|
| `include/visualization/Visualizer.hpp` + `.cpp` | 多面板调试渲染 (Debug模式) |

### 10.10 工具

| 文件 | 关键函数 |
|------|---------|
| `include/utils/PoseUtils.hpp` + `.cpp` | loadTemplates(), readCorners(), calculateOverlap(), extractAndNormalizeRoi(), orderPoints() |

### 10.11 输入系统

| 文件/目录 | 关键内容 |
|-----------|---------|
| `include/input/InputProvider.hpp` + `src/input/InputProvider.cpp` | 统一协调器 (图像源工厂 + getNextPacket) |
| `include/input/IStereoImageSource.hpp` | 图像源抽象接口 |
| `include/input/SensorTypes.hpp` | SensorPacket / ImuData / AltimeterData |
| `include/input/InputConfig.hpp` | InputSystemConfig / ImageSourceType 枚举 (含 Camera) |
| `include/input/CameraSource.hpp` + `src/input/CameraSource.cpp` | **实时摄像头源 (Phase 3)** |
| `include/input/RingBuffer.hpp` | 环形缓冲区 (线程采集预留, 已单测) |
| `src/input/` | FileStereoSource, DirectoryStereoSource, SequenceSource 实现 |
| `extracted_input_system/` | 独立可复用输入系统模块 (CanSocket/TimeSyncUnit, Phase 2 参考) |
| `include/fusion/FusionTypes.hpp` | ImuSample/RadarSample/CameraObservation (延迟测量样本/观测类型) |
| `include/fusion/EskfFusionManager.hpp` + `src/fusion/EskfFusionManager.cpp` | **ESKF 多源融合适配层** (缓冲/排干对齐/lazy init/反向传播/退化监控/位姿转换/统计) |
| `include/input/SimulatedSensors.hpp` + `src/input/SimulatedSensors.cpp` | 合成 IMU/雷达源 (离线链路验证) |
| `eskf/eskf_vio.hpp` | ESKF 库 (header-only, 第三方, 不改动) |
| `config/tracker_config_eskf.json` | ESKF 融合示例配置 |

### 10.12 配置与模型

| 文件 | 用途 |
|------|------|
| `config/tracker_config.json` | 默认配置文件 |
| `best.onnx` | YOLO ONNX 模型 |
| `data/` | 测试图像与模板数据 |
| `sysml/` | SysML 需求模型 (sysrequire.puml, softwarerequire.puml, flow.puml) |

### 10.13 附加文档

| 文件 | 内容 |
|------|------|
| `README.md` | 项目完整文档 (面向人类) |
| `CLAUDE.md` | 本文档 (面向 AI Agent) |
| `FEATURE_EXTRACTION_SPEC.md` | 特征提取规格说明 |
| `dockerfile_0620` | Docker 构建文件 |

---

## 附录 A: 关键常量速查

| 常量 | 值 | 位置 | 说明 |
|------|-----|------|------|
| `tiny_max_area` | 800 | config/代码 | State 1/2 分界 |
| `akaze_min_area` | 40001 | config/代码 | State 2/3 分界 |
| `dual_trigger_area` | 490000 | config/代码 | State 3/4 分界 (700×700) |
| 最低特征点数 (PnP) | 4 (单目), 3 (双目) | 代码 | — |
| Lowe's ratio | 0.75 | TemplateMatcher | AKAZE 匹配 |
| Homography 阈值 | 5.0 px | TemplateMatcher | RANSAC 内点 |
| LK FB 误差阈值 | 1.0 px | OpticalFlowTracker | 光流验证 |
| MAD σ 系数 | 3.0 | MadDisparityFilter | 视差滤波 |
| 超分倍数 (TT) | 4.0 | TinyTarget | 小目标放大 |
| Template target_size (BC) | 100 | BinaryCorner | 标准化尺寸 |
| Template target_size (TT) | 50 | TinyTarget | 标准化尺寸 |
| RANSAC 迭代 (InitialPnP) | 300 | InitialPnPSolver | — |
| RANSAC 置信度 | 0.99 | PnP Solvers | — |
| RANSAC 重投影阈值 | 8.0 px | PnP Solvers | — |
| 深度有效范围 | [10, 100000] mm (MonoPnP), [10, 20000] mm (GPnP) | 位姿校验 | — |
| ITERATIVE 发散阈值 | \|t\| > 100000 mm | MonoPnPSolver | 检测到发散时回退到 RANSAC |
| approxPolyDP 二分搜索 | 0.001~0.05×perimeter, 8次 | BC | — |
| max_imu_gap_s | 0.1 s | eskf 配置 | IMU 间隙超限丢弃样本 |
| max_cam_gap_s | 1.0 s | eskf 配置 | 相机间隔超限重置滤波器 |
| backprop_window_s | 0.2 s | eskf 配置 | 反向传播回退窗口 |
| state_hist_hz | 100 | eskf 配置 | 状态快照记录频率 |
| max_output_age_s | 0.5 s | eskf 配置 | 相机更新间隔超限 → DEGRADED |
| 相机 FDI 门限 | χ²₉₅(3)=7.8147 / χ²₉₉₉(3)=16.266 | eskf_vio.hpp | 位置/姿态 NIS 分级 |
| 姿态追加门限 | nis_rot×4.0 < 7.8147 | eskf_vio.hpp | hybrid 路径姿态自洽 |
| 雷达跳变/NIS 门限 | 30.0 m / 7.879 | RadarAltimeter | 默认值 |

---

## 附录 B: 常见修改场景指南

### B.1 调整面积阈值

修改 `config/tracker_config.json` 中 `strategies.akaze_min_area`、`strategies.tiny_max_area`、`strategies.dual_roi.trigger_area`。

### B.2 添加新的特征提取器

1. 继承 `FeatureExtractor`，实现 `extract()` 和 `extractMono()`
2. 在 `StrategyType` 枚举添加新类型
3. 在 `StereoTracker` 构造函数中创建实例
4. 修改 `configureStrategyChain()` 添加策略分发逻辑
5. 修改 `dispatchPnP()` 添加 PnP 路由

### B.3 添加实时摄像头支持

Phase 3 已完成（`CameraSource`）。扩展新视频源（如 USB 双目、RTSP）时：在 `src/input/` 中实现新的 `IStereoImageSource` 子类，注册到 `InputProvider::createImageSource()`；若需同步双摄像头，参考线程化采集模式（独立采集线程 + RingBuffer，见 §7.2 与 `InputProvider::captureLoop`）。

### B.4 调优 YOLO 模型

替换 `best.onnx` 文件，相应调整 `yolo.conf_threshold` 和 `yolo.target_class_id`。模型输入尺寸自动从 ONNX 读取。

### B.5 启用/调优 ESKF 融合

1. 配置 `eskf.enabled: true` + `input_system.imu/altimeter.type: "simulated"` (见 config/tracker_config_eskf.json)
2. 运行验证: 终端输出 `[Frame N] ESKF p=[..]m ... | PnP t(mm)=[..]`; 汇总含 ESKF 统计
3. 调优方向: `noise.cam_pos_noise/cam_rot_noise` (相机观测信任度), `noise.sigma_acc/sigma_gyro` (IMU 信任度), `noise.radar_alt_noise` (雷达), `init_std.*` (首帧协方差), `max_cam_gap_s` (视觉丢失容忍)
4. 接入真实硬件 (Phase 2): 串口/CAN 源以"带时间戳样本流"模式产出, 直接喂 `feedImu/feedRadar` 即可, 对齐逻辑在适配层内
5. ⚠️ 融合输出为主输出; 需保留原始 PnP 时看 verbose 日志 `PnP t(mm)=[...]`