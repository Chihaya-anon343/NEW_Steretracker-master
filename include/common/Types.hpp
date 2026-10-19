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

/// 时序连贯性配置（序列模式：位姿链 + 策略粘滞 + 运动门控）。
/// 所有阈值以"帧"为单位 —— 离线序列时间戳为墙钟，不可用 dt。
struct TemporalConfig {
    bool enabled{true};                  ///< 总开关（false = 全部回到无时序行为）
    // --- 位姿链 (机制A) ---
    int max_cache_age_frames{30};        ///< seed 最大帧龄，超龄冷启动（<=0 = 不限制）
    double tie_epsilon_px{0.3};          ///< ε-平票窗口：重投影差在此内视为平票，平票内取离上帧位姿最近者
    // --- 运动门控 (机制C) ---
    double max_trans_ratio{0.35};        ///< |Δt| 上限 = 该值 × |t_prev|（相对门控，<=0 关闭平移检查）
    double max_scale_ratio{3.0};         ///< 径向快变容限：|t_new|/|t_seed| 在界内视为合法快速接近/远离（与 Δt 判据并列，任一满足即过；<=1 关闭该判据）
    double max_rot_deg{15.0};            ///< Δθ 上限（度/帧，<=0 关闭旋转检查）
    double switch_margin{2.0};           ///< 策略切换帧门控的阈值放宽倍数
    // --- 策略粘滞 (机制B) ---
    int hold_frames{3};                  ///< 去抖：相邻档提名连续确认帧数
    double hysteresis_up{1.25};          ///< 升档迟滞余量：EMA面积需 > 边界阈值 × 该值
    double hysteresis_down{0.8};         ///< 降档迟滞余量：EMA面积需 < 边界阈值 × 该值
    int locked_fail_limit{3};            ///< 锁定策略连续被 fallback 顶掉 N 帧后跟随胜者
    // --- 提名平滑 ---
    double area_ema_alpha{0.5};          ///< 面积 EMA 系数（仅用于策略提名/迟滞，不影响裁剪；<=0 = 不平滑）
};

/// Optional left-only BC frontend. Missing configuration retains dual BC behavior.
struct StereoFlowConfig {
    bool enabled{false};
    bool dual_roi_flow{false};
    bool dual_roi_legacy_fallback{true};
    bool disparity_only{false};
    bool disparity_full_chain{false}; ///< Left BC/TT and class1 disparity chain; opt-in for old configurations.
    bool roi_pyramid{false};
    bool roi_fallback_full{true};
    double roi_max_area_ratio{0.85};
    int roi_extra_margin_px{16};
    bool allow_left_pnp{false};
    int window_size{21};
    int max_level{3};
    int max_iterations{30};
    double epsilon{0.01};
    double min_eigenvalue{1e-4};
    double max_fb_error_px{1.0};
    double max_epipolar_error_px{1.5};
    double min_disparity_px{0.5};
    double max_disparity_px{400.0};
    int ncc_patch_size{11};
    double min_ncc{0.8};
    double uniqueness_margin{0.03};
    int max_search_candidates{512};
    bool coarse_retry{true};
    bool coarse_epipolar_band{true};  ///< false reproduces the original single-line coarse search.
    int min_matches{4};
    double min_valid_ratio{0.6};
    double min_coverage_ratio{0.25};
    double duplicate_distance_px{2.0};
    double initial_reprojection_px{8.0};
    double max_mean_reprojection_px{3.0};
    double max_point_reprojection_px{8.0};
    bool planar_geometry_filter{false}; ///< Opt-in: input points must belong to one planar target.
    double planar_max_error_px{0.25};
    int planar_min_points{6}; ///< TT four-point paths remain under their original checks.
    double planar_min_inlier_ratio{0.7};
    int planar_max_hypotheses{128};
};

enum class PoseSource { Unspecified, StereoGpnp, LeftPnpFallback };
inline const char* poseSourceName(PoseSource source) {
    switch (source) {
        case PoseSource::StereoGpnp: return "StereoGpnp";
        case PoseSource::LeftPnpFallback: return "LeftPnpFallback";
        default: return "Unspecified";
    }
}

/// Common planar frame: offset is class1 center in class0 axes (mm).
/// Centers are XY bounding-box midpoints of canonical template points.
struct TemplateGeometryConfig {
    bool enabled{false}; ///< Missing old config retains original model frames.
    Eigen::Vector2d class1_center_in_class0_mm{Eigen::Vector2d::Zero()};
};

/// Mono PnP: horizontal single-sided board, camera optical axis near nadir.
/// Template +X right/+Y down makes +Z point into the board (toward ground).
struct MonoPosePriorConfig {
    bool enabled{false}; ///< Absent configuration retains legacy solver behavior.
    double max_tilt_deg{30.0}; ///< Total angle: template +Z versus camera +Z.
};

struct RoiSelectionConfig {
  bool enabled{false};
  bool verify_mono{false};
  int max_candidates{3};
  double containment_min{0.95};
  double containment_margin_px{2.0};
  double min_component_ratio{0.25};
  double min_observed_span_px{4.0};
  int history_max_age{15};
};

/// 跟踪器级别配置（构造时一次性设置）。
struct TrackerConfig {
    int gpnp_min_pts{4};                 ///< GPNP 最少点数要求
    bool use_initial_pnp{true};          ///< 是否启用 RANSAC+ITERATIVE 初始 PnP
    double template_real_width_mm{200.0}; ///< 模板物理宽度 (mm)
    double template_real_height_mm{200.0};///< 模板物理高度 (mm)
    int tiny_max_area{800};              ///< TinyTarget 策略的最大 ROI 面积
    int tiny_max_area_class1{0};         ///< class1-only 时的 TinyTarget 阈值 (0=回退到 tiny_max_area)
    int dual_roi_secondary_expand{10};   ///< 双 ROI 模式下次级（class 1）ROI 的拓展像素数
    bool dual_roi_class1_fallback{true}; ///< 双 ROI 第 3 级退化: 合并与 BC-only 均失败后在 class1 ROI 上跑 BC→TT 链
    double dual_roi_primary_span_ratio{0.99}; ///< 巨型 primary 短路: primary 边长覆盖画幅 ≥ 该比例 → 目标外角点出视野, 跳过 Tier1/2 直接 class1 链 (<=0 关闭)
    bool dual_roi_tier1_only{false};     ///< Dual-ROI 消融: true = 仅 Tier1 (屏蔽 Tier2/Tier3/巨型 primary 短路)
    std::string dual_roi_class0_points_path; ///< Dual-ROI class0 人工标注点文件 (空=approxPolyDP 自动推导)
    std::string dual_roi_class1_points_path; ///< Dual-ROI class1 人工标注点文件 (空=Tier1b 联合解算停用)
    double dual_roi_joint_match_gate_px{12.0}; ///< Tier1b NN 匹配门限 (px, 投影预测点↔提取角点)
    TemporalConfig temporal;             ///< 时序连贯性（序列模式）
    StereoFlowConfig stereo_flow;
    TemplateGeometryConfig template_geometry;
    MonoPosePriorConfig mono_pose_prior;
    RoiSelectionConfig roi_selection;
};

// ============================================================================
// 特征数据
// ============================================================================

/// 不可变模板数据（构造时一次性提取）。
struct TemplateData {
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
    PoseSource source{PoseSource::Unspecified};

    /// 如果位姿表示一个有效的估计则返回 true。
    bool valid() const { return success && num_points > 0; }
};

/// 位姿链式热启动的种子（上一成功帧位姿，模板坐标系与 PnP 输出同系）。
struct PoseSeed {
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
};

/// 运动门控结果状态码。
enum class GateStatus {
    NotApplicable = 0,   ///< 无 seed，未启用门控
    Pass = 1,            ///< 热启动解通过门控
    Recovered = 2,       ///< 热启动解被拒，冷启动重解通过（放宽门控）
    Rejected = 3         ///< 冷热两解均被拒，该策略判失败继续退化链
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
    bool candidate_valid{false};
    bool pose_accepted{false};
    std::string initial_source;
    std::string geometry_reason;
    double mean_left_px{0.0}, mean_right_px{0.0}, max_reprojection_px{0.0};
    bool used_initial_pose_fallback{false};
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
    GPNPMonitor last_gpnp_monitor;
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
    double extract_ms{0.0}; ///< 特征提取总耗时 (ms)
    double pnp_ms{0.0};     ///< 位姿解算总耗时 (ms)

    std::string strategy_name;            ///< 使用的提取器策略
    std::string stereo_method;
    PoseSource pose_source{PoseSource::Unspecified};
    bool stereo_success{false};
    bool disparity_only{false};
    bool disparity_success{false};
    std::string failure_reason;
    int flow_coarse_attempts{0};
    double stereo_match_ms{0.0};
    cv::Rect flow_roi;
    bool flow_full_fallback{false};
    int flow_pyramid_builds{0}, dual_c0_pairs{0}, dual_c1_pairs{0};
    bool legacy_frontend_fallback{false};
    std::string route_detail;
    std::string flow_planar_status{"disabled"};
    int flow_planar_rejected{0};
    double flow_planar_ms{0};
    bool is_class1{false};                ///< 是否 class1 回退帧
    double t_x{0.0}, t_y{0.0}, t_z{0.0};        ///< 平移 (mm)
    double rvec_x{0.0}, rvec_y{0.0}, rvec_z{0.0};///< 旋转向量 (angle*axis, θ∈[0,π] 有 ±180° 跳变)
    double quat_w{1.0}, quat_x{0.0}, quat_y{0.0}, quat_z{0.0};///< 符号连续化四元数 (w,x,y,z)
    double rot_deg_unwrapped{0.0};              ///< 累计 unwrapped 旋转角 (度, 相对首个成功帧)

    // --- 时序连贯性观测 (temporal) ---
    bool warm_start_used{false};          ///< 本帧位姿解算是否使用了上帧 seed
    int gate_status{0};                   ///< GateStatus 枚举值 (0=N/A, 1=Pass, 2=Recovered, 3=Rejected)
    int seed_age{0};                      ///< seed 帧龄（成功帧清零）
    bool sticky_hold{false};              ///< 本帧粘滞抑制了提名切换 (locked != nominated)
    std::string nominated_strategy;       ///< 面积提名的策略名
    std::string switch_reason;            ///< 最近一次锁定切换原因 (init/hard_override/debounce/realign/reset)
};

// ============================================================================
// 完整流水线结果（单帧）
// ============================================================================

/// 单次 process() 调用的聚合结果。
/// 替代 Python 版本中 process() 返回的字典。
struct PipelineResult {
  // BC observations retain canonical IDs after quality/partial filtering.
  int bc_detected_count{0}, bc_rejected_count{0};
  int pose_min_inliers{4};
  std::vector<cv::Point2f> bc_rejected_points;
  std::vector<int> bc_rejected_ids;
  std::vector<double> bc_residuals_px;
  int roi_candidates{0}, roi_verified{0}, roi_selected{-1};
  double roi_containment{0.0};
  std::string roi_selection_reason;
  GPNPMonitor
      last_gpnp_monitor; ///< Solver snapshot for the selected pose attempt.
  // --- 特征提取 ---
  std::vector<cv::KeyPoint> kp_left;
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
  bool success{false}; ///< 流水线整体成功标志（提取 + PnP，单目/双目通用）
  std::string
      strategy_name; ///< 本帧胜出的策略名（如 "BinaryCorner", "TinyTarget"）
  std::string stereo_method;
  PoseSource pose_source{PoseSource::Unspecified};
  bool stereo_success{false};
  bool disparity_only{false};
  bool disparity_success{false};
  std::string failure_reason;
  int flow_coarse_attempts{0};
  double stereo_match_ms{0.0};
  cv::Rect flow_roi;
  bool flow_full_fallback{false};
  int flow_pyramid_builds{0}, dual_c0_pairs{0}, dual_c1_pairs{0};
  bool legacy_frontend_fallback{false};
  std::string route_detail;
  std::string flow_planar_status{"disabled"};
  int flow_planar_rejected{0};
  double flow_planar_ms{0};
  // Aligned with pts_left_good; original class-local IDs survive filtering.
  std::vector<int> flow_regions, flow_original_indices, flow_template_indices;
  std::string model_frame;
  std::vector<Eigen::Vector3d>
      pose_model_points; ///< Explicit model table indexed by
                         ///< good_matches.trainIdx.
  double quat_w{1.0}, quat_x{0.0}, quat_y{0.0},
      quat_z{0.0}; ///< 符号连续化四元数 (w,x,y,z, 由 finalizePose 填充)
  double rot_deg_unwrapped{
      0.0}; ///< 累计 unwrapped 旋转角 (度, 由 finalizePose 填充)

  // --- 耗时统计 ---
  std::map<std::string, double> timing; ///< 阶段 → 毫秒
  double extract_ms{
      0.0}; ///< 特征提取总耗时 (ms, 含退化链多次尝试 / Dual-ROI 并行两路)
  double pnp_ms{0.0}; ///< 位姿解算总耗时 (ms, 含
                      ///< InitialPnP/GPnP/TinyPnP/MonoPnP 及退化重解)

  // --- 帧元数据 ---
  cv::Mat left_color;
  cv::Mat right_color;
  bool is_first_frame{false};
  bool fallback_used{false};
  bool is_class1{false}; ///< class1-only 回退帧标志
  int n_matched{0};
  int n_projected{0};

  // --- ROI 偏移（0626: 用于将 ROI 坐标映射回全图） ---
  int left_roi_offset_x{0};
  int left_roi_offset_y{0};
  int right_roi_offset_x{0};
  int right_roi_offset_y{0};

  // --- 时序连贯性观测 (temporal, 由 TrackerBase::fillTemporalMeta 填充) ---
  bool warm_start_used{false};
  GateStatus gate_status{GateStatus::NotApplicable};
  int seed_age{0};
  bool sticky_hold{false};
  std::string nominated_strategy;
  std::string switch_reason;

  // --- MAD 同步所需的中间数据（0626） ---
  std::vector<bool> valid_mask; ///< 投影步骤的 valid_mask

  /// 从所有耗时条目计算总处理时间。
  double total_time_ms() const {
    double total = 0.0;
    for (const auto &[_, t] : timing)
      total += t;
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
    bool is_class1{false};  ///< 近距离回退：primary 是 class1 外扩的（极近），BC/Tiny 需用 class1 尺寸

    /// 如果主 ROI 有效（最低要求）则返回 true。
    bool valid() const { return primary.valid(); }

    /// 如果主 ROI 和次级 ROI 均无效则返回 true。
    bool empty() const { return !primary.valid() && !secondary.valid(); }
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
    // --- 连续姿态输出基准 (finalizePose 维护) ---
    Eigen::Quaterniond q_prev{1.0, 0.0, 0.0, 0.0}; ///< 上帧输出四元数 (符号连续化基准)
    bool has_quat_cache{false};
    double rot_deg_unwrapped{0.0};            ///< 累计 unwrapped 旋转角 (度)
    Eigen::Vector3d axis_acc{0.0, 0.0, 1.0};  ///< 平滑累计旋转轴 (Δ角符号判定基准)
    int frame_count{0};
    int cache_frame{-1};               ///< 写入缓存时的 frame_count（DualRoi 路径 cache_age_ 不自增，用帧号差补齐 seed 年龄）
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
