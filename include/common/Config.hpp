#pragma once

/**
 * @file Config.hpp
 * @brief 双目相机参数与跟踪器配置的工厂函数。
 *
 * 构造辅助函数，验证输入并预计算派生量。
 */

#include "Types.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gpnp {

inline void validateStereoFlowConfig(const StereoFlowConfig& c) {
    if (!std::isfinite(c.planar_max_error_px) || c.planar_max_error_px <= 0 ||
        c.planar_min_points < 6 || c.planar_min_points > 1024 ||
        !std::isfinite(c.planar_min_inlier_ratio) || c.planar_min_inlier_ratio <= 0.5 ||
        c.planar_min_inlier_ratio > 1 || c.planar_max_hypotheses < 1 || c.planar_max_hypotheses > 4096)
        throw std::invalid_argument("Invalid stereo_frontend planar geometry settings");
    if (c.dual_roi_flow && !c.enabled)
        throw std::invalid_argument("dual_roi_flow requires stereo_frontend.method=left_bc_lk");
    if (c.disparity_only && !c.enabled)
        throw std::invalid_argument("disparity_only requires stereo_frontend.method=left_bc_lk");
    if (c.disparity_full_chain && (!c.disparity_only || !c.enabled))
        throw std::invalid_argument("disparity_full_chain requires left_bc_lk and disparity_only=true");
    if (!std::isfinite(c.roi_max_area_ratio) || c.roi_max_area_ratio <= 0 || c.roi_max_area_ratio > 1 ||
        c.roi_extra_margin_px < 0 || c.roi_extra_margin_px > 512)
        throw std::invalid_argument("Invalid stereo_frontend ROI settings");
    if (c.window_size < 3 || c.window_size % 2 == 0 || c.max_level < 0 || c.max_level > 8 ||
        c.max_iterations < 1 || !std::isfinite(c.epsilon) || c.epsilon <= 0 ||
        !std::isfinite(c.min_eigenvalue) || c.min_eigenvalue < 0 ||
        c.ncc_patch_size < 3 || c.ncc_patch_size % 2 == 0 ||
        c.max_search_candidates < 2 || c.max_search_candidates > 8192 || c.min_matches < 4)
        throw std::invalid_argument("Invalid stereo_frontend window, pyramid or point settings");
    auto positive = [](double x) { return std::isfinite(x) && x > 0; };
    if (!positive(c.max_fb_error_px) || !positive(c.max_epipolar_error_px) ||
        !positive(c.min_disparity_px) || !positive(c.max_disparity_px) ||
        c.max_disparity_px <= c.min_disparity_px ||
        !positive(c.duplicate_distance_px) || !positive(c.initial_reprojection_px) ||
        !positive(c.max_mean_reprojection_px) || !positive(c.max_point_reprojection_px) ||
        !std::isfinite(c.min_ncc) || c.min_ncc < -1 || c.min_ncc > 1 ||
        !std::isfinite(c.uniqueness_margin) || c.uniqueness_margin < 0 || c.uniqueness_margin > 2 ||
        !std::isfinite(c.min_valid_ratio) || c.min_valid_ratio <= 0 || c.min_valid_ratio > 1 ||
        !std::isfinite(c.min_coverage_ratio) || c.min_coverage_ratio < 0 || c.min_coverage_ratio > 1)
        throw std::invalid_argument("Invalid stereo_frontend quality or disparity settings");
}

inline StereoFlowConfig readStereoFlowConfig(const cv::FileNode& node) {
    StereoFlowConfig c;
    if (node.empty()) return c;
    const std::string method = node["method"].empty() ? "dual_bc" : static_cast<std::string>(node["method"]);
    if (method != "dual_bc" && method != "left_bc_lk")
        throw std::invalid_argument("stereo_frontend.method must be dual_bc or left_bc_lk");
    c.enabled = method == "left_bc_lk";
    if (!node["dual_roi_flow"].empty()) c.dual_roi_flow = static_cast<int>(node["dual_roi_flow"]) != 0;
    if (!node["dual_roi_legacy_fallback"].empty()) c.dual_roi_legacy_fallback = static_cast<int>(node["dual_roi_legacy_fallback"]) != 0;
    if (!node["disparity_only"].empty()) c.disparity_only = static_cast<int>(node["disparity_only"]) != 0;
    if (!node["disparity_full_chain"].empty()) c.disparity_full_chain = static_cast<int>(node["disparity_full_chain"]) != 0;
    const std::string failure = node["on_failure"].empty() ? "reject" : static_cast<std::string>(node["on_failure"]);
    if (failure != "reject" && failure != "left_pnp")
        throw std::invalid_argument("stereo_frontend.on_failure must be reject or left_pnp");
    c.allow_left_pnp = failure == "left_pnp";
    auto integer = [&](const char* key, int& v) { if (!node[key].empty()) v = static_cast<int>(node[key]); };
    auto real = [&](const char* key, double& v) { if (!node[key].empty()) v = static_cast<double>(node[key]); };
    integer("window_size", c.window_size); integer("max_level", c.max_level);
    if (!node["roi_pyramid"].empty()) c.roi_pyramid = static_cast<int>(node["roi_pyramid"]) != 0;
    if (!node["roi_fallback_full"].empty()) c.roi_fallback_full = static_cast<int>(node["roi_fallback_full"]) != 0;
    real("roi_max_area_ratio", c.roi_max_area_ratio); integer("roi_extra_margin_px", c.roi_extra_margin_px);
    integer("max_iterations", c.max_iterations); real("epsilon", c.epsilon);
    real("min_eigenvalue", c.min_eigenvalue); real("max_fb_error_px", c.max_fb_error_px);
    real("max_epipolar_error_px", c.max_epipolar_error_px);
    real("min_disparity_px", c.min_disparity_px); real("max_disparity_px", c.max_disparity_px);
    integer("ncc_patch_size", c.ncc_patch_size); real("min_ncc", c.min_ncc);
    real("uniqueness_margin", c.uniqueness_margin); integer("max_search_candidates", c.max_search_candidates);
    if (!node["coarse_retry"].empty()) c.coarse_retry = static_cast<int>(node["coarse_retry"]) != 0;
    if (!node["coarse_epipolar_band"].empty()) c.coarse_epipolar_band = static_cast<int>(node["coarse_epipolar_band"]) != 0;
    integer("min_matches", c.min_matches); real("min_valid_ratio", c.min_valid_ratio);
    real("min_coverage_ratio", c.min_coverage_ratio); real("duplicate_distance_px", c.duplicate_distance_px);
    real("initial_reprojection_px", c.initial_reprojection_px);
    real("max_mean_reprojection_px", c.max_mean_reprojection_px);
    real("max_point_reprojection_px", c.max_point_reprojection_px);
    if (!node["planar_geometry_filter"].empty()) c.planar_geometry_filter = static_cast<int>(node["planar_geometry_filter"]) != 0;
    real("planar_max_error_px", c.planar_max_error_px);
    integer("planar_min_points", c.planar_min_points);
    real("planar_min_inlier_ratio", c.planar_min_inlier_ratio);
    integer("planar_max_hypotheses", c.planar_max_hypotheses);
    validateStereoFlowConfig(c);
    return c;
}

/// 从独立矩阵构造双目相机参数。
/// 验证矩阵维度并预计算 K_inv、focal_length、baseline。
inline StereoCameraParams makeStereoCameraParams(
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& R_rl,
    const Eigen::Vector3d& t_rl)
{
    if (K(2, 0) != 0.0 || K(2, 1) != 0.0 || K(2, 2) != 1.0) {
        throw std::invalid_argument("K 必须是标准内参形式 [fx 0 cx; 0 fy cy; 0 0 1]");
    }
    if (!K.allFinite() || !R_rl.allFinite() || !t_rl.allFinite()) {
        throw std::invalid_argument("相机参数包含 NaN 或 Inf");
    }
    if (std::abs(R_rl.determinant() - 1.0) > 1e-6) {
        throw std::invalid_argument("R_rl 行列式必须为 1.0（需要有效的旋转矩阵）");
    }

    StereoCameraParams p;
    p.K = K;
    p.K_inv = K.inverse();
    p.R_rl = R_rl;
    p.t_rl = t_rl;
    p.focal_length = K(0, 0);
    p.baseline = t_rl.norm();
    return p;
}

/// 构造带验证的跟踪器配置。
inline TrackerConfig makeTrackerConfig(
    int gpnp_min_pts = 4,
    bool use_initial_pnp = true,
    double template_real_width_mm = 200.0,
    double template_real_height_mm = 200.0,
    int tiny_max_area = 800,
    int tiny_max_area_class1 = 0,
    int dual_roi_secondary_expand = 10)
{
    if (gpnp_min_pts < 3) {
        throw std::invalid_argument("gpnp_min_pts 必须 >= 3，当前值: " + std::to_string(gpnp_min_pts));
    }
    if (template_real_width_mm <= 0.0 || template_real_height_mm <= 0.0) {
        throw std::invalid_argument("模板物理尺寸必须为正数");
    }
    if (tiny_max_area <= 0) {
        throw std::invalid_argument("tiny_max_area 必须为正数");
    }

    TrackerConfig cfg;
    cfg.gpnp_min_pts = gpnp_min_pts;
    cfg.use_initial_pnp = use_initial_pnp;
    cfg.template_real_width_mm = template_real_width_mm;
    cfg.template_real_height_mm = template_real_height_mm;
    cfg.tiny_max_area = tiny_max_area;
    cfg.tiny_max_area_class1 = tiny_max_area_class1;
    cfg.dual_roi_secondary_expand = dual_roi_secondary_expand;
    return cfg;
}

/// 构造带验证的 YOLO 检测器配置。
inline YoloConfig makeYoloConfig(
    const std::string& model_path,
    DeviceType device = DeviceType::Auto,
    float conf_threshold = 0.5f,
    float iou_threshold = 0.45f,
    cv::Size input_size = cv::Size(640, 640),
    int intra_op_threads = 4)
{
    if (model_path.empty()) {
        throw std::invalid_argument("YOLO model_path 不能为空");
    }
    if (conf_threshold <= 0.0f || conf_threshold > 1.0f) {
        throw std::invalid_argument("conf_threshold 必须在 (0, 1] 范围内，当前值: " + std::to_string(conf_threshold));
    }
    if (iou_threshold <= 0.0f || iou_threshold > 1.0f) {
        throw std::invalid_argument("iou_threshold 必须在 (0, 1] 范围内，当前值: " + std::to_string(iou_threshold));
    }
    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::invalid_argument("YOLO input_size 必须为正数");
    }
    if (intra_op_threads < 1) {
        throw std::invalid_argument("intra_op_threads 必须 >= 1");
    }

    YoloConfig cfg;
    cfg.model_path = model_path;
    cfg.device = device;
    cfg.conf_threshold = conf_threshold;
    cfg.iou_threshold = iou_threshold;
    cfg.input_size = input_size;
    cfg.intra_op_threads = intra_op_threads;
    return cfg;
}

} // namespace gpnp
