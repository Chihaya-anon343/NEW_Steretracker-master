#pragma once

/**
 * @file Types.hpp
 * @brief GPNP 双目跟踪系统的核心数据结构。
 *
 * 所有模块间数据传递均使用此处定义的强类型结构体，
 * 替代 Python 版本中松散的字典式数据传递。
 */

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <map>
#include <string>
#include <vector>

namespace gpnp {

// ============================================================================
// 相机与双目几何
// ============================================================================

/// 双目相机对参数（构造后不可变）。
struct StereoCameraParams {
    Eigen::Matrix3d K;       ///< 3×3 相机内参矩阵
    Eigen::Matrix3d K_inv;   ///< K 的预计算逆矩阵
    Eigen::Matrix3d R_rl;    ///< 旋转：右相机 → 左相机
    Eigen::Vector3d t_rl;    ///< 平移：右相机在左相机坐标系中的位置
    double focal_length;     ///< K(0,0)，缓存以便快速访问
    double baseline;         ///< ||t_rl||，缓存以便快速访问（基线长度）
};

// ============================================================================
// 配置参数
// ============================================================================

/// Lucas-Kanade 光流参数。
struct LKParams {
    cv::Size winSize{21, 21};
    int maxLevel{3};
    cv::TermCriteria criteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01};
    double minEigThreshold{1e-4};
};

/// 跟踪器级别配置（构造时一次性设置）。
struct TrackerConfig {
    double scale{0.5};                   ///< AKAZE 图像缩放因子 (0~1)
    int gpnp_min_pts{4};                 ///< GPNP 最少点数要求
    bool use_initial_pnp{true};          ///< 是否启用 RANSAC+ITERATIVE 初始 PnP
    double template_real_width_mm{200.0}; ///< 模板物理宽度 (mm)
    double template_real_height_mm{200.0};///< 模板物理高度 (mm)
    LKParams lk_params;                  ///< 光流参数
    int akaze_min_area{40000};           ///< 选择 AKAZE 策略的最小 ROI 面积
    int tiny_max_area{800};              ///< TinyTarget 策略的最大 ROI 面积
    int dual_roi_secondary_expand{10};   ///< 双 ROI 模式下次级（class 1）ROI 的拓展像素数
    double dual_roi_akaze_scale{0.5};    ///< 双 ROI class 1 提取时的 AKAZE 缩放
};

// ============================================================================
// 特征数据
// ============================================================================

/// 单幅图像 AKAZE 特征提取结果。
struct FeatureSet {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;                        ///< CV_8U, N×61 二值描述子
    std::vector<cv::Point2f> points;            ///< 关键点坐标 (N×1×2 → N×2)
    int num_keypoints{0};
};

/// 不可变模板数据（构造时一次性提取）。
struct TemplateData {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat color_image;
    cv::Mat gray_image;
    std::vector<Eigen::Vector3d> pts_3d;        ///< 模板坐标系下的 3D 坐标 (mm)
    int template_width{0};
    int template_height{0};

    // --- BinaryCorner / TinyTarget 模板支持 ---
    cv::Mat image;                                ///< 原始模板 PNG 图像（灰度）
    cv::Mat image_bool;                           ///< 预计算二值掩膜（image > 127）
    std::vector<cv::Point2f> corners;             ///< 从 .txt 文件读取的角点坐标
    int angle{0};                                 ///< 旋转角度（度，从文件名解析）
};

// ============================================================================
// 光流跟踪结果
// ============================================================================

/// LK 光流 L→R 跟踪结果，含 FB 校验 + MAD 滤波。
struct TrackResult {
    std::vector<cv::Point2f> pts_left_good;      ///< 有效的左图特征点
    std::vector<cv::Point2f> pts_right_good;     ///< 有效的右图特征点
    std::vector<int> idx_from_filtered;          ///< 映射回原始 kp_left 的索引
    std::vector<double> disparity;               ///< = -dx_filtered（以负 dx 存储）
    std::vector<double> dx_filtered;             ///< MAD 滤波后的 x 视差
    int num_matched{0};

    // FB 校验统计（诊断用）
    int num_before_fb{0};
    int num_after_fb{0};
    int num_after_mad{0};
    double fb_error_mean{0.0};
};

// ============================================================================
// 立体投影结果
// ============================================================================

/// 双目深度估计与右图投影的结果。
struct ProjectionResult {
    std::vector<cv::Point2f> pts_right_projected; ///< 右图上的投影点
    std::vector<cv::Point2f> pts_left_used;       ///< 具有有效投影的左图点
    std::vector<cv::Point2f> pts_right_used;      ///< 具有有效投影的右图点
    std::vector<bool> valid_mask;                  ///< N 元素掩膜（true = 有效投影）
    int num_projected{0};
};

// ============================================================================
// 模板匹配结果
// ============================================================================

/// 模板匹配结果（三阶段：比率测试 → 交叉校验 → 单应性 RANSAC）。
struct MatchResult {
    std::vector<cv::DMatch> good_matches;        ///< 最终筛选后的 DMatch 列表
    std::vector<cv::Point2f> pts_left_match;     ///< 匹配的左图像素点
    std::vector<cv::Point2f> pts_template_match; ///< 匹配的模板像素点
    int num_matches{0};

    // 筛选统计（诊断用）
    int ratio_test_count{0};
    int cross_check_count{0};
    int homography_count{0};
};

// ============================================================================
// 位姿估计结果
// ============================================================================

/// 相机相对于模板的位姿：P_cam = R * P_template + t
struct PoseEstimate {
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()}; ///< 旋转：模板 → 相机
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};     ///< 平移 (mm)
    bool success{false};
    int num_points{0};

    /// 如果位姿表示一个有效的估计则返回 true。
    bool valid() const { return success && num_points > 0; }
};

// ============================================================================
// GPNP 监控 / 诊断
// ============================================================================

/// GPNP 优化的详细监控数据（每帧一份）。
struct GPNPMonitor {
    // 数据流
    int n_good_matches{0};
    int n_stereo_matched{0};
    int n_query_to_train{0};
    int n_idx_from_filtered{0};
    int n_matched{0};
    int n_intersection{0};
    int n_missing_query{0};
    int n_missing_train{0};
    int n_pts{0};
    int n_rays{0};
    int gpnp_min_pts{0};

    // 优化过程
    bool opt_success{false};
    int opt_status{0};
    std::string opt_message;
    int opt_nfev{0};
    double opt_initial_cost{0.0};
    double opt_final_cost{0.0};
    double opt_cost_reduction{0.0};
    double depth_guess{0.0};

    // 结果
    std::vector<double> q_opt;   ///< [x, y, z, w] 优化后的四元数
    std::vector<double> t_opt;   ///< [tx, ty, tz] 优化后的平移

    // 错误
    std::string failure_reason;
    std::string exception;
    double timing_ms{0.0};

    /// 检查是否记录了任何失败原因。
    bool failed() const { return !failure_reason.empty() || !exception.empty(); }
};

// ============================================================================
// 日志记录
// ============================================================================

/// 单帧日志记录（等价于 Python 的 _add_log 字典）。
struct LogEntry {
    int frame{0};
    double timestamp{0.0};
    bool is_first{false};
    bool fallback_used{false};

    int n_kp_left{0};
    int n_matched{0};
    int n_projected{0};
    int n_template_match{0};

    bool gpnp_success{false};
    int gpnp_n_pts{0};
    double disparity_median{0.0};

    double total_time_ms{0.0};
    std::map<std::string, double> timing; ///< 各阶段耗时 (ms)
};

// ============================================================================
// 完整流水线结果（单帧）
// ============================================================================

/// 单次 process() 调用的聚合结果。
/// 替代 Python 版本中 process() 返回的字典。
struct PipelineResult {
    // --- 特征提取 ---
    std::vector<cv::KeyPoint> kp_left;
    cv::Mat desc_left;
    int n_kp_left{0};

    // --- 光流跟踪 ---
    std::vector<cv::Point2f> pts_left_good;
    std::vector<cv::Point2f> pts_right_good;
    std::vector<double> disparity;
    std::vector<double> dx_filtered;
    std::vector<int> idx_from_filtered;

    // --- 立体投影 ---
    std::vector<cv::Point2f> pts_right_projected;
    std::vector<cv::Point2f> pts_left_used;
    std::vector<cv::Point2f> pts_right_used;

    // --- 模板匹配 ---
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts_left_match;
    std::vector<cv::Point2f> pts_template_match;
    int n_template_match{0};

    // --- 位姿估计 ---
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
    bool gpnp_success{false};
    int gpnp_n_pts{0};

    // --- 耗时统计 ---
    std::map<std::string, double> timing; ///< 阶段 → 毫秒

    // --- 帧元数据 ---
    cv::Mat left_color;
    cv::Mat right_color;
    bool is_first_frame{false};
    bool fallback_used{false};
    int n_matched{0};
    int n_projected{0};

    // --- ROI 偏移（0626: 用于将 ROI 坐标映射回全图） ---
    int left_roi_offset_x{0};
    int left_roi_offset_y{0};
    int right_roi_offset_x{0};
    int right_roi_offset_y{0};

    // --- MAD 同步所需的中间数据（0626） ---
    std::vector<bool> valid_mask;  ///< 投影步骤的 valid_mask

    /// 从所有耗时条目计算总处理时间。
    double total_time_ms() const {
        double total = 0.0;
        for (const auto& [_, t] : timing) total += t;
        return total;
    }
};

// ============================================================================
// ROI 矩形（0626: left_roi, right_roi 参数）
// ============================================================================

/// 矩形感兴趣区域：(x, y, width, height)。
struct RoiRect {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    /// 如果表示一个有效（非空）的 ROI 则返回 true。
    bool valid() const { return width > 0 && height > 0; }

    /// 如果偏移为零（全图模式，未应用 ROI）则返回 true。
    bool isZero() const { return x == 0 && y == 0; }
};

/// ROI 组：主 ROI（class 0）+ 可选的次级 ROI（class 1，用于大目标）。
struct RoiGroup {
    RoiRect primary;        ///< Class 0 ROI（始终被现有策略使用）
    RoiRect secondary;      ///< Class 1 ROI（仅当 is_dual == true 时有效）
    bool is_dual{false};    ///< 当次级 ROI 存在且有效时为 true

    /// 如果主 ROI 有效（最低要求）则返回 true。
    bool valid() const { return primary.valid(); }

    /// 如果主 ROI 和次级 ROI 均无效则返回 true。
    bool empty() const { return !primary.valid() && !secondary.valid(); }
};

// ============================================================================
// MAD 滤波结果（0626: 由 MadDisparityFilter::filter() 返回）
// ============================================================================

/// MadDisparityFilter::filter() 的输出，匹配 Python 字典结构。
struct MadFilterResult {
    std::vector<cv::Point2f> pts_left_filtered;
    std::vector<cv::Point2f> pts_right_filtered;
    std::vector<double> disparity;        ///< = -dx_filtered
    std::vector<double> dx_filtered;
    std::vector<double> dy_filtered;
    std::vector<int> idx_from_filtered;
    std::vector<bool> filter_mask;        ///< N 元素布尔掩膜
    bool downgraded{false};               ///< 若点数过少导致回退则为 true
};

// ============================================================================
// 可变跟踪状态（帧间缓存）
// ============================================================================

/// 封装跨帧持久化的所有可变状态。
/// 替代分散的 self._has_cache, self._cache, self._frame_count, self._logs。
struct TrackingState {
    bool has_cache{false};
    Eigen::Matrix3d R_prev{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t_prev{0.0, 0.0, 500.0};  ///< 默认：500mm 深度
    int frame_count{0};
    std::vector<LogEntry> logs;
    GPNPMonitor last_gpnp_monitor;             ///< 最近的 GPNP 诊断数据
};

// ============================================================================
// YOLO 检测类型（用于基于 ONNX Runtime 的目标检测）
// ============================================================================

/// ML 推理的设备后端。
enum class DeviceType {
    Auto,   ///< 优先尝试 CUDA，失败则回退到 CPU
    CPU,
    CUDA
};

/// YOLO 检测器 / 特征提取器返回的推理状态。
enum class Status {
    Success = 0,
    ModelLoadFailed,
    EmptyInput,
    InferenceFailed,
    UnknownError,

    // --- BinaryCorner / TinyTarget 状态码 ---
    InvalidSize,            ///< 输入图像尺寸无效
    InsufficientFeatures,   ///< 提取的特征/角点不足
    NoSuitableComponent     ///< 未找到合适的连通域
};

/// 单个目标检测结果。
struct Detection {
    int class_id{0};
    float confidence{0.0f};
    cv::Rect2f bbox;  ///< 原始图像坐标系中的边界框
};

/// YOLO 检测器配置（初始化 + ROI 生成）。
struct YoloConfig {
    std::string model_path;
    DeviceType device{DeviceType::Auto};
    float conf_threshold{0.5f};
    float iou_threshold{0.45f};
    cv::Size input_size{640, 640};
    int intra_op_threads{4};

    // ROI 生成参数
    int target_class_id{0};         ///< 目标物体的 class ID
    float roi_expand_ratio{0.1f};   ///< 按此比例扩展 ROI（0.1 = 10%）
    int roi_min_size{100};          ///< 最小 ROI 像素尺寸
};

} // namespace gpnp
