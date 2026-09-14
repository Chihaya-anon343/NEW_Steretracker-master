#pragma once

/**
 * @file NetworkKeypointExtractor.hpp
 * @brief 对比方法占位 —— YOLO ROI → 关键点网络 → 2D/3D 对应点 → PnP。
 *
 * 独立于主 pipeline 的 FeatureExtractor 策略体系：
 *   - 不参与面积分级 / 策略链选择 / 退化链 / ROI padding
 *   - 仅被 main_compare.cpp (pose_compare) 使用
 *
 * 输入约定: BGR ROI 子图 (由 YOLO class0/class1 检测框裁剪)
 * 输出约定: pts_2d 为 ROI 局部坐标, 与 pts_3d (模板系, mm, z=0) 严格 1:1
 */

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include <string>
#include <vector>

namespace gpnp {

class NetworkKeypointExtractor {
public:
    struct Config {
        std::string model_path;               ///< 占位: 关键点 ONNX 模型路径
        int num_keypoints = 0;                ///< 占位: 网络输出关键点数
        double template_real_width_mm = 0.0;  ///< 占位: 关键点 3D 对应平面宽度 (mm)
        double template_real_height_mm = 0.0; ///< 占位: 关键点 3D 对应平面高度 (mm)
    };

    struct Result {
        bool success = false;
        std::string message;                   ///< 失败原因 / 调试信息
        std::vector<cv::Point2f> pts_2d;       ///< ROI 局部坐标 (px)
        std::vector<Eigen::Vector3d> pts_3d;   ///< 模板 3D 对应 (mm, z=0), 与 pts_2d 1:1
    };

    explicit NetworkKeypointExtractor(const Config& cfg);

    /// 加载网络模型。占位实现: model_path 为空时打印警告并返回 false。
    bool initialize();

    /// ROI 图像 → 关键点 2D/3D 对应。占位实现: 恒返回失败。
    Result detect(const cv::Mat& roi_bgr);

private:
    Config cfg_;
    bool initialized_ = false;
};

} // namespace gpnp
