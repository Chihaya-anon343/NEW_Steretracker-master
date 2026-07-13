#pragma once

/**
 * @file Config.hpp
 * @brief 双目相机参数与跟踪器配置的工厂函数。
 *
 * 构造辅助函数，验证输入并预计算派生量。
 */

#include "Types.hpp"

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

namespace gpnp {

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
    double scale = 0.5,
    int gpnp_min_pts = 4,
    bool use_initial_pnp = true,
    double template_real_width_mm = 200.0,
    double template_real_height_mm = 200.0,
    int akaze_min_area = 40000,
    int tiny_max_area = 800,
    int dual_roi_secondary_expand = 10,
    double dual_roi_akaze_scale = 0.5)
{
    if (scale <= 0.0 || scale > 1.0) {
        throw std::invalid_argument("scale 必须在 (0, 1] 范围内，当前值: " + std::to_string(scale));
    }
    if (gpnp_min_pts < 3) {
        throw std::invalid_argument("gpnp_min_pts 必须 >= 3，当前值: " + std::to_string(gpnp_min_pts));
    }
    if (template_real_width_mm <= 0.0 || template_real_height_mm <= 0.0) {
        throw std::invalid_argument("模板物理尺寸必须为正数");
    }
    if (akaze_min_area <= 0) {
        throw std::invalid_argument("akaze_min_area 必须为正数");
    }
    if (tiny_max_area <= 0) {
        throw std::invalid_argument("tiny_max_area 必须为正数");
    }

    TrackerConfig cfg;
    cfg.scale = scale;
    cfg.gpnp_min_pts = gpnp_min_pts;
    cfg.use_initial_pnp = use_initial_pnp;
    cfg.template_real_width_mm = template_real_width_mm;
    cfg.template_real_height_mm = template_real_height_mm;
    cfg.akaze_min_area = akaze_min_area;
    cfg.tiny_max_area = tiny_max_area;
    cfg.dual_roi_secondary_expand = dual_roi_secondary_expand;
    cfg.dual_roi_akaze_scale = dual_roi_akaze_scale;
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