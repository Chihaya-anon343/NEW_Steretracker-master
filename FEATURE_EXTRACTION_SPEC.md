# 特征提取模块功能说明书

> **目标读者**：另一个 AI / 开发者，需要在其他项目中迁移或重新实现这两个特征提取模块。
> **关注重点**：输入/输出接口、内部流水线、依赖项、模板文件格式。

---

## 目录

1. [共享基础设施](#1-共享基础设施)
2. [BinaryCornerExtractor — 二值图像角点提取器](#2-binarycornerextractor)
3. [TinyTargetPoseDetector — 微小目标角点检测器](#3-tinytargetposedetector)
4. [集成方式与调用链路](#4-集成方式与调用链路)
5. [模板文件格式](#5-模板文件格式)

---

## 1. 共享基础设施

### 1.1 数据类型（定义于 `include/Types.hpp`）

两个模块都依赖以下共享类型：

```cpp
// 状态码枚举
enum class Status {
    Success = 0,
    EmptyInput,
    InvalidSize,
    FileNotFound,
    // ...
    InsufficientFeatures,
    NoSuitableComponent,
    UnknownError,
};

// 模板数据结构
struct TemplateData {
    int angle = 0;                        // 模板角度（度），从文件名解析
    cv::Mat image;                        // 灰度模板图像
    cv::Mat image_bool;                   // 预计算的布尔掩码（image > 127），用于加速 IoU
    std::vector<cv::Point2f> corners;     // 模板角点坐标
};

// 角点提取结果
struct CornerResult {
    std::vector<cv::Point2f> corners;                  // 提取的角点
    std::vector<cv::Point2f> matched_template_corners; // 对应模板角点
    double match_overlap = 0.0;
    int match_angle = -1;
    PoseResult pose;              // PnP 解算结果（仅调用方做 PnP 时填充）
};

// 微小目标结果
struct TinyTargetResult {
    int match_angle = -1;
    double match_overlap = 0.0;
    Eigen::Matrix<double, 4, 2> corners_in_image;   // 完整图像坐标
    Eigen::Matrix<double, 4, 2> corners_in_roi;     // ROI 相对坐标
    PoseResult pose;                                  // PnP 解算结果
};

// 位姿结果
struct PoseResult {
    Eigen::Vector3d rvec;
    Eigen::Vector3d tvec;
    Eigen::Matrix3d rotation_matrix;
    Eigen::Vector3d camera_position_in_obj;
    double reprojection_error_mean / std / max;
    int inlier_count;
    int correspondence_count;
};
```

### 1.2 共享工具函数（定义于 `include/PoseUtils.hpp`, `src/PoseUtils.cpp`）

| 函数 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `loadTemplates(dir, reshape_to_1x2)` | 从目录加载所有角点模板 | 目录路径 | `std::vector<TemplateData>`（按角度排序，image_bool 已预计算） |
| `calculateOverlap(a, b)` | 计算两张二值图的 IoU（交并比） | 两张 `cv::Mat`（uint8 或 bool） | `double` [0,1] |
| `extractAndNormalizeRoi(binary_img, target_size)` | 提取白色区域的外接正方形并归一化 | 二值图 + 目标尺寸 | `(cv::Mat, bool)` |
| `readCorners(txt_path)` | 解析 `_degrees.txt` 角点文件 | 文件路径 | `std::vector<cv::Point2f>` |
| `orderPoints(pts)` | 将 4 角点排序为 TL→TR→BR→BL | 4 个点 | 排序后的 4 个点 |
| `reorderCorners(ordered, angle_deg)` | 按模板角度旋转 4 角点的起始索引 | 4 个有序点 + 角度 | 重排后的 4 个点 |
| `imsave(img, title, filename, out_dir)` | 保存图像文件 | Mat + 元数据 | bool |

---

## 2. BinaryCornerExtractor

**文件**：`include/BinaryCorner.hpp` + `src/BinaryCorner.cpp`

**目的**：从二值图像中提取多边形目标的 N 个角点。适用于中等尺寸目标（像素面积在 800~40000 之间）。使用模板匹配确定旋转角度并指导角点排序。

### 2.1 配置结构

```cpp
struct Config {
    int corners = 10;                  // 期望提取的角点数量
    int kernel_size = 3;               // 形态学核大小（强制奇数）
    double scale = 1.0;                // 内部缩放系数，>1 可提供亚像素精度
    std::string template_dir;          // 角点模板目录路径
    cv::Size target_size{100, 100};    // 模板匹配时的归一化尺寸
};
```

**config.yaml 对应项**：
```yaml
binary_corner:
  template_dir: "模板/NewMuBan"
  corners: 10
  kernel: 3
  scale: 3
  target_width: 100
  target_height: 100
```

### 2.2 构造函数

```cpp
explicit BinaryCornerExtractor(const Config& config);
```

**行为**：
- 校验 kernel_size 为奇数
- 调用 `PoseUtils::loadTemplates(config.template_dir, false)` 加载模板
- 用 `spdlog::info` 输出加载的模板数量

**无返回值**，若模板加载失败则 templates_ 为空（不会抛异常）。

### 2.3 主接口：`extract()`

```cpp
Status extract(const cv::Mat& binary_img, std::vector<cv::Point2f>& out_corners);
```

#### 输入

| 参数 | 类型 | 要求 |
|------|------|------|
| `binary_img` | `const cv::Mat&` | 二值图像，推荐 **CV_8UC1、值 0/255**。内部会自动处理：3 通道→灰度、非二值→阈值化（>127→255）、非 CV_8UC1→转换 |

**典型调用方预处理**（在 Pipeline.cpp 中）：
```cpp
// 对 ROI 灰度图做 Otsu 二值化
cv::Mat binary;
cv::threshold(roi_gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
// 传入提取器
std::vector<cv::Point2f> corners;
Status s = corner_extractor_->extract(binary, corners);
```

#### 输出

| 参数 | 类型 | 说明 |
|------|------|------|
| 返回值 | `Status` | `Success` / `EmptyInput` / `InvalidSize` / `InsufficientFeatures` |
| `out_corners` | `std::vector<cv::Point2f>&` | 提取的角点（已按模板顺序排列），坐标系与输入图像一致 |

#### 提取后状态查询

```cpp
const TemplateData*  lastMatchedTemplate()  const;  // 本次匹配到的模板，未匹配则 nullptr
double               lastMatchOverlap()     const;  // 本次模板匹配的 IoU 值 [0,1]
const vector<Point2f>& lastCornersBeforeReorder() const; // 重排前角点（调试用）
const vector<pair<string,string>>& processLog() const;     // 处理日志
```

### 2.4 内部流水线（7 步）

```
输入二值图
  │
  ├─[1] keepLargestRegion()      → 保留最大连通分量（connectedComponentsWithStats）
  │
  ├─[2] fillHoles()              → 填充孔洞（findContours + RETR_EXTERNAL + drawContours FILLED）
  │
  ├─[3] smoothBoundary()         → 形态学平滑（MORPH_CLOSE → MORPH_OPEN，核大小来自 config.kernel_size）
  │
  ├─[4] findBestMatch()          → 模板匹配（仅在 templates_ 非空时）
  │   ├─ 将平滑后图像 resize 到 target_size（INTER_NEAREST）
  │   ├─ extractAndNormalizeRoi() → 提取白色区域并归一化到 50×50
  │   └─ 遍历模板，calculateOverlap() 计算 IoU，取最佳
  │
  ├─[4.5] rotate_and_clean_image() → 旋转回正（当有匹配模板时）
  │   ├─ 使用模板角度旋转图像
  │   ├─ 形态学清理边缘（MORPH_OPEN + MORPH_CLOSE）
  │   └─ 记录旋转中心（用于后续逆变换）
  │
  ├─[5] extractLargestContour()  → 提取最大外轮廓（RETR_EXTERNAL）
  │
  ├─[6] extractCornersFromContour() → approxPolyDP 二分搜索
  │   ├─ 预处理：按 config.scale 缩放轮廓点
  │   ├─ epsilon 二分搜索：lo=0.001, hi=0.05，最多 8 轮
  │   ├─ 目标：approxPolyDP 得到精确 config.corners 个角点
  │   └─ 后处理：缩放回原始坐标
  │
  ├─[6.5] 逆旋转变换（若执行了步骤 4.5）
  │   将角点从正位图坐标变换回原始倾斜图坐标
  │
  └─[7] matchCorners() → 模板角点重排序
      ├─ 将模板角点缩放到 smoothed 空间
      ├─ reorderByGeometry() → 以模板角度为参考，用极角对齐角点顺序
      └─ 按模板顺序重排 binary_corners
```

### 2.5 关键算法细节

#### rotate_and_clean_image()（辅助函数，非成员）

```cpp
std::tuple<cv::Mat, cv::Point2f, cv::Point2f>
rotate_and_clean_image(const cv::Mat& binary_image, double angle);
```
- 输入：二值图 + 旋转角度（度）
- 输出：(旋转+清理后的图, 原图旋转中心, 旋转后图中心)
- 旋转后执行 MORPH_OPEN + MORPH_CLOSE + 阈值化，消除旋转产生的模糊边缘

#### reorderByGeometry()（静态方法）

```cpp
static std::vector<int> reorderByGeometry(
    const std::vector<cv::Point2f>& corners,
    const cv::Point2f& center,
    double reference_angle_deg,
    double ref_dist = -1.0);
```
- 零轴 = 正上方（图像坐标 y 轴负方向），CCW 为正
- 找最接近 `reference_angle` 的角点作为起点
- 若提供 `ref_dist`，在最近两个候选点中用距离消歧
- 轮廓已知为 CCW（OpenCV 白色外轮廓保证），始终正向遍历

### 2.6 静态可视化方法

```cpp
// 在图像上绘制角点（前 3 个红/绿/黄，其余蓝）
static cv::Mat drawCorners(const cv::Mat& img, const std::vector<cv::Point2f>& corners);

// 并排绘制：输入图角点（左）| 模板图角点（右），含中心点和角度射线
static cv::Mat drawMatchedCorners(
    const cv::Mat& input_img, const std::vector<cv::Point2f>& input_corners,
    const cv::Mat& template_img, const std::vector<cv::Point2f>& template_corners,
    double template_angle = 0.0);

// 调试用：返回 4 张独立图（二值角点 | 模板角点 | 二值极坐标 | 模板极坐标）
static std::vector<cv::Mat> debugVisualizeReordering(
    const cv::Mat& binary_img, const cv::Mat& template_img,
    const std::vector<cv::Point2f>& binary_corners,
    const std::vector<cv::Point2f>& template_corners,
    double template_angle, const cv::Size& coord_size);
```

### 2.7 依赖清单

| 依赖 | 用途 |
|------|------|
| OpenCV `core`, `imgproc`, `calib3d` | 图像处理、轮廓提取、形态学 |
| spdlog | 日志输出 |
| `Types.hpp` → `TemplateData`, `Status` | 数据结构 |
| `PoseUtils.hpp` → `loadTemplates()`, `calculateOverlap()`, `extractAndNormalizeRoi()` | 模板加载、IoU 计算 |

---

## 3. TinyTargetPoseDetector

**文件**：`include/TinyTargetPose.hpp` + `src/TinyTargetPose.cpp`

**目的**：检测微小矩形目标（≤20×20 px，或像素面积 ≤800）的 4 个角点。使用超分辨率放大 + 模板引导的角度匹配 + minAreaRect + 亚像素精化。

### 3.1 配置结构

```cpp
struct Config {
    std::string template_dir;          // 角点模板目录路径
    int pad_pixels = 0;               // 内部外扩（当前版本不使用）
    cv::Size target_size{50, 50};     // 模板匹配归一化尺寸
    int scale_factor = 4;             // 超分辨率放大倍数（2~8 推荐）
    double square_size_m = 0.20;      // 目标物理边长（米），用于 PnP
    cv::Mat camera_matrix;            // 3×3 相机内参矩阵（用于 PnP）
    cv::Mat dist_coeffs;              // 畸变系数（用于 PnP）
};
```

**config.yaml 对应项**：
```yaml
tiny_target_pose:
  template_dir: "模板/NewMuBan"
  pad_pixels: 0
  target_width: 50
  target_height: 50
  scale_factor: 4
  square_size_m: 0.2
```

### 3.2 构造函数

```cpp
explicit TinyTargetPoseDetector(const Config& config);
```

**行为**：
- 调用 `PoseUtils::loadTemplates(config.template_dir, false)` 加载模板
- 用 `spdlog::warn` 输出未加载到模板时的警告（不阻塞）
- 用 `spdlog::info` 输出加载的模板数量

### 3.3 轻量接口：`extractCorners()`

> 设计用于双目联合 PnP 场景，仅返回 4 个角点，不做 PnP。

```cpp
Status extractCorners(
    const cv::Mat& roi_gray,       // ROI 灰度图
    const cv::Mat& roi_color,      // ROI 彩色图（当前实现未使用，预留给未来扩展）
    std::vector<cv::Point2f>& corners_roi,  // 输出：ROI 坐标系下的 4 个角点
    int& best_angle                // 输出：模板匹配的最佳角度（-1 表示未匹配到）
);
```

#### 输入

| 参数 | 类型 | 要求 |
|------|------|------|
| `roi_gray` | `const cv::Mat&` | 灰度图，CV_8UC1。必须是已裁剪的 ROI（仅包含目标区域） |
| `roi_color` | `const cv::Mat&` | 彩色图（当前未使用，可传空 Mat 或同尺寸任意图） |

#### 输出

| 参数 | 类型 | 说明 |
|------|------|------|
| 返回值 | `Status` | `Success` / `EmptyInput` / `NoSuitableComponent` |
| `corners_roi` | `std::vector<cv::Point2f>&` | **4 个角点**，ROI 坐标，已排序（TL→TR→BR→BL），已按模板角度对齐起始点 |
| `best_angle` | `int&` | 匹配到的模板角度（0/90/180/270），未匹配为 -1 |

**坐标说明**：`corners_roi` 在 ROI 局部坐标系中 —— 即 ROI 图的左上角为 (0,0)。

### 3.4 完整接口：`estimatePose()`

```cpp
Status estimatePose(
    const cv::Mat& roi_gray,           // ROI 灰度图
    const cv::Mat& roi_color,          // ROI 彩色图
    TinyTargetResult& result,          // 输出结果
    const cv::Mat& full_img_gray = {},    // 完整灰度图（可选，用于可视化偏移）
    const cv::Mat& full_img_color = {},   // 完整彩色图（可选，用于可视化偏移）
    const cv::Point& roi_offset = {0,0},  // ROI 在完整图中的左上角偏移
    const std::string& out_dir = "debug_out_tiny",  // 可视化输出目录
    bool verbose = true,               // 是否输出详细日志
    bool save_vis = true               // 是否保存可视化图像
);
```

#### 输入

| 参数 | 类型 | 要求 |
|------|------|------|
| `roi_gray` | `const cv::Mat&` | 灰度 ROI，CV_8UC1 |
| `roi_color` | `const cv::Mat&` | 彩色 ROI，用于可视化标注 |
| `full_img_gray` | `const cv::Mat&` | 完整灰度图（非必须） |
| `full_img_color` | `const cv::Mat&` | 完整彩色图（非必须） |
| `roi_offset` | `const cv::Point&` | ROI 左上角在完整图像中的像素坐标 |

#### 输出 `TinyTargetResult`

| 字段 | 类型 | 说明 |
|------|------|------|
| `match_angle` | `int` | 匹配到的模板角度（度） |
| `match_overlap` | `double` | 模板匹配 IoU 值 [0,1] |
| `corners_in_image` | `Eigen::Matrix<double,4,2>` | 完整图像坐标下的 4 个角点 |
| `corners_in_roi` | `Eigen::Matrix<double,4,2>` | ROI 坐标下的 4 个角点 |
| `pose` | `PoseResult` | PnP 解算结果 |

### 3.5 内部流水线

```
输入 ROI 灰度图
  │
  ├─[1] 模板匹配（matchTemplate）
  │   ├─ Otsu 二值化 ROI
  │   ├─ 提取最大连通分量 → 裁剪正方形外接框
  │   ├─ 归一化到 target_size（INTER_NEAREST）
  │   ├─ 遍历所有模板，calculateOverlap() 计算 IoU
  │   └─ 返回 best_angle + best_overlap + 全部重叠度（降序）
  │
  ├─[2] 超分辨率 + 连通域分析
  │   ├─ resize × scale_factor（INTER_CUBIC）
  │   ├─ GaussianBlur(3×3, σ=0.3)
  │   ├─ Otsu 二值化
  │   ├─ MORPH_OPEN(3×3) + MORPH_CLOSE(5×5)
  │   └─ connectedComponentsWithStats
  │
  ├─[3] selectBestComponent() — 评分选择最佳连通域
  │   评分 = rect_ratio×0.25 + area_score×0.30 + center_score×0.30 + aspect_score×0.15
  │   ├─ rect_ratio: area / bbox_area（矩形度）
  │   ├─ area_score: 面积比例，最佳区间 [0.15, 0.6]
  │   ├─ center_score: 距图像中心越近越高
  │   └─ aspect_score: 1/aspect_ratio（越接近正方形越高）
  │   过滤条件：area < 200 的直接跳过
  │
  ├─[4] minAreaRect + 角点排序
  │   ├─ 提取最佳分量的最大轮廓
  │   ├─ cv::minAreaRect → 4 个顶点
  │   └─ PoseUtils::orderPoints() → TL→TR→BR→BL
  │
  ├─[5] 亚像素精化（refineCorners）
  │   ├─ copyMakeBorder + cornerSubPix
  │   ├─ winSize=(5,5), zeroZone=(-1,-1)
  │   └─ criteria: MAX_ITER=50, EPS=0.001
  │
  ├─[6] 角度对齐
  │   quadrant = round((360 - best_angle) / 90) % 4
  │   std::rotate 将对应角点旋至首位
  │
  ├─[7] 缩放回原始坐标
  │   所有角点 ÷ scale_factor
  │
  └─[8] PnP 解算（仅 estimatePose）
      ├─ 物点：4 个正方形顶点，边长 square_size_m
      ├─ solvePnP(ITERATIVE)
      ├─ 计算重投影误差
      └─ 计算相机在目标系中的位置
```

### 3.6 模板匹配细节

```cpp
// 内部结构（私有）
struct TemplateMatchResult {
    int best_angle = -1;
    double best_overlap = 0.0;
    std::vector<std::pair<int, double>> all_overlaps; // (角度, IoU)，降序
};
```

匹配流程：
1. Otsu 二值化 → 提取最大连通分量 → 正方形裁剪 → resize 到 target_size
2. 与每个模板的 `image_bool` 计算 IoU
3. 返回最佳角度和全部重叠度（降序排列）

### 3.7 依赖清单

| 依赖 | 用途 |
|------|------|
| OpenCV `core`, `imgproc`, `calib3d` | 图像处理、形态学、minAreaRect、cornerSubPix、solvePnP |
| Eigen | 角点坐标的矩阵存储（`Matrix<double,4,2>`） |
| spdlog | 日志输出 |
| `Types.hpp` → `TemplateData`, `TinyTargetResult`, `PoseResult`, `Status` | 数据结构 |
| `PoseUtils.hpp` → `loadTemplates()`, `calculateOverlap()`, `orderPoints()`, `imsave()`, `drawPoseAxes()` | 工具函数 |

---

## 4. 集成方式与调用链路

### 4.1 策略分发逻辑（Pipeline.cpp）

```
YOLO 检测框面积
  │
  ├─ 面积 > sift_pnp 阈值 (40000)
  │   → SiftPnPEstimator
  │
  ├─ 面积 ≤ tiny_pnp 阈值 (800)
  │   → TinyTargetPoseDetector
  │
  └─ 其余
      → BinaryCornerExtractor（也是 SIFT/Tiny 失败后的回退方案）
```

### 4.2 调用 BinaryCornerExtractor 的完整代码模型

```cpp
// === 初始化（一次） ===
BinaryCornerExtractor::Config corner_cfg;
corner_cfg.corners      = config.cornerCount();        // 如 10
corner_cfg.kernel_size  = config.cornerKernelSize();   // 如 3
corner_cfg.scale        = config.cornerScale();        // 如 3.0
corner_cfg.template_dir = config.cornerTemplateDir();  // 如 "模板/NewMuBan"
corner_cfg.target_size  = config.cornerTargetSize();   // 如 {100, 100}
auto corner_extractor = std::make_unique<BinaryCornerExtractor>(corner_cfg);

// === 每检测框处理 ===
// 1. 从完整图提取 ROI
cv::Mat roi_gray = full_img_gray(roi_rect);

// 2. 二值化
cv::Mat binary;
cv::threshold(roi_gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

// 3. 提取角点
std::vector<cv::Point2f> corners;
Status s = corner_extractor->extract(binary, corners);
if (s == Status::Success && !corners.empty()) {
    // 4. 获取模板匹配信息
    const auto* tmpl = corner_extractor->lastMatchedTemplate();
    double overlap = corner_extractor->lastMatchOverlap();

    // 5. 可选：PnP 解算
    if (tmpl && corners.size() >= 4) {
        double p2m = config.cornerPixelToMeterScale(); // 如 0.01
        std::vector<cv::Point3d> obj_pts;
        for (const auto& pt : tmpl->corners) {
            obj_pts.emplace_back(pt.x * p2m, pt.y * p2m, 0.0);
        }
        // ... solvePnP(obj_pts, corners_as_Point2d, K, dist, rvec, tvec)
    }

    // 6. 可选：可视化
    cv::Mat corner_viz = BinaryCornerExtractor::drawCorners(roi_color, corners);
    cv::Mat matched_viz = BinaryCornerExtractor::drawMatchedCorners(
        roi_color, corners, tmpl->image, tmpl->corners, tmpl->angle);
}
```

### 4.3 调用 TinyTargetPoseDetector 的完整代码模型

```cpp
// === 初始化（一次） ===
TinyTargetPoseDetector::Config tiny_cfg;
tiny_cfg.template_dir  = config.tinyTemplateDir();     // 如 "模板/NewMuBan"
tiny_cfg.pad_pixels    = config.tinyPadPixels();       // 如 0
tiny_cfg.target_size   = config.tinyTargetSize();      // 如 {50, 50}
tiny_cfg.scale_factor  = config.tinyScaleFactor();     // 如 4
tiny_cfg.square_size_m = config.tinySquareSizeM();     // 如 0.2
tiny_cfg.camera_matrix = config.sharedCameraMatrix();  // 3×3
tiny_cfg.dist_coeffs   = config.sharedDistCoeffs();    // 1×5
auto tiny_detector = std::make_unique<TinyTargetPoseDetector>(tiny_cfg);

// === 方式 A：仅提取角点（用于双目联合 PnP）===
std::vector<cv::Point2f> corners_roi;
int best_angle = -1;
Status s = tiny_detector->extractCorners(roi_gray, roi_color, corners_roi, best_angle);
// corners_roi 是 ROI 局部坐标下的 4 个角点，已排序且角度对齐

// === 方式 B：完整估计（角点 + PnP 位姿）===
TinyTargetResult result;
Status s = tiny_detector->estimatePose(
    roi_gray, roi_color,
    result,
    full_img_gray, full_img_color,  // 可选
    cv::Point(x1p, y1p),            // ROI 在完整图中的偏移
    out_dir,                        // 输出目录
    false,                          // verbose
    save_vis                        // 是否保存可视化
);
// result.corners_in_roi   → ROI 坐标下的 4 个角点 (Eigen::Matrix<double,4,2>)
// result.corners_in_image → 完整图坐标下的 4 个角点
// result.pose              → PnP 位姿结果
```

---

## 5. 模板文件格式

两个模块共用同一套模板目录（config 中可配置为相同或不同路径）。

### 5.1 目录结构

```
模板/NewMuBan/
├── 0_degrees.txt
├── 0_degrees.png
├── 90_degrees.txt
├── 90_degrees.png
├── 180_degrees.txt
├── 180_degrees.png
├── 270_degrees.txt
└── 270_degrees.png
```

### 5.2 `_degrees.txt` 格式

```
# 注释行（以 # 开头）
# 以下每行一个角点：序号: x, y
0: 12.5, 8.3
1: 87.2, 9.1
2: 86.8, 88.7
3: 13.1, 87.5
...
```

- 以 `#` 开头的行为注释
- 数据行格式：`{序号}: {x}, {y}`
- 坐标是**模板图像**坐标系中的像素坐标（左上角原点）
- 需至少 4 个角点
- 解析函数：`PoseUtils::readCorners(txt_path)`

### 5.3 `_degrees.png` 格式

- 对应角度的二值/灰度模板图像
- `TemplateData::image` = `cv::imread(path, IMREAD_GRAYSCALE)`
- `TemplateData::image_bool` = `image > 127`（自动预计算）
- **文件名中的角度数字**（如 0、90、180、270）会被解析为 `TemplateData::angle`

### 5.4 模板加载逻辑

```cpp
// PoseUtils::loadTemplates() 的核心逻辑
for (const auto& entry : fs::directory_iterator(template_dir)) {
    // 1. 匹配 "*_degrees.txt" 文件
    // 2. 从文件名提取角度（如 "0_degrees.txt" → angle=0）
    // 3. 读取同名的 .png（同路径，去掉 .txt + ".png"）
    // 4. 读取角点（readCorners）
    // 5. 预计算 image_bool = image > 127
    // 6. 存入 TemplateData 并按角度排序
}
```

---

## 6. 迁移检查清单

要在另一个项目中重新实现这两个模块，需要：

- [ ] **类型定义**：复制/重写 `Status` 枚举、`TemplateData`、`CornerResult`、`TinyTargetResult`、`PoseResult`
- [ ] **模板加载**：实现 `loadTemplates()` 和 `readCorners()`（约 60 行代码）
- [ ] **IoU 计算**：实现 `calculateOverlap()`（约 30 行 + OpenCV bitwise_and/or）
- [ ] **ROI 归一化**：实现 `extractAndNormalizeRoi()`（约 40 行）
- [ ] **BinaryCornerExtractor**：约 500 行 C++，纯 OpenCV 图像处理 + 轮廓运算
- [ ] **TinyTargetPoseDetector**：约 400 行 C++，OpenCV + minAreaRect + cornerSubPix + solvePnP
- [ ] **模板文件**：准备含 `_degrees.txt` + `_degrees.png` 的模板目录
- [ ] **依赖库**：OpenCV（core, imgproc, calib3d）、Eigen、spdlog（日志可替换为其他方案）
