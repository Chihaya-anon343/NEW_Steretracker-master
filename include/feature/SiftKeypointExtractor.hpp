#pragma once

/**
 * @file SiftKeypointExtractor.hpp
 * @brief 对比方法: SIFT/ORB 特征点提取 —— ROI 图像 ↔ 模板特征匹配 → 2D/3D 对应点 → PnP。
 *
 * 独立于主 pipeline 的 FeatureExtractor 策略体系 (仿 AkazeExtractor, 不继承):
 *   - 不参与面积分级 / 策略链选择 / 退化链 / ROI padding
 *   - 仅被 main_compare.cpp (pose_compare) 使用
 *
 * method 切换 (配置 extractor.method):
 *   - "sift": cv::SIFT, 浮点描述子 → 匹配用 NORM_L2
 *   - "orb" : cv::ORB , 二值描述子 → 匹配用 NORM_HAMMING
 *
 * 输入约定: BGR ROI 子图 (由 YOLO class0/class1 检测框裁剪)
 * 输出约定: pts_2d 为 ROI 局部坐标, 与 pts_3d (模板系, mm, z=0) 严格 1:1
 * 匹配约定: 三阶段 (Lowe ratio test → cross-check → Homography RANSAC)
 */

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace gpnp {

class TemplateMatcher;

class SiftKeypointExtractor {
public:
    struct Config {
        std::string method = "sift";          ///< "sift" | "orb"
        std::string template_path;            ///< 模板图像路径
        double template_real_width_mm = 0.0;  ///< 模板物理宽度 (mm)
        double template_real_height_mm = 0.0; ///< 模板物理高度 (mm)
        double scale = 1.0;                   ///< 图像缩放因子 (0,1], 提取后坐标还原
        int nfeatures = 2000;                 ///< 特征数上限 (0=不限)
        double ratio_threshold = 0.75;        ///< Lowe ratio test 阈值
        int min_pts = 4;                      ///< PnP 最低 2D/3D 对应点数
    };

    struct Result {
        bool success = false;
        std::string message;                   ///< 失败原因 / 调试信息
        std::vector<cv::Point2f> pts_2d;       ///< ROI 局部坐标 (px)
        std::vector<Eigen::Vector3d> pts_3d;   ///< 模板 3D 对应 (mm, z=0), 与 pts_2d 1:1
    };

    explicit SiftKeypointExtractor(const Config& cfg);
    ~SiftKeypointExtractor();

    SiftKeypointExtractor(const SiftKeypointExtractor&) = delete;
    SiftKeypointExtractor& operator=(const SiftKeypointExtractor&) = delete;

    /// 加载模板图像并预提取模板特征 + 3D 点。失败返回 false (message 给出原因)。
    bool initialize();

    /// ROI 图像 → SIFT 特征 → 模板匹配 → 2D/3D 对应。未 initialize 时恒失败。
    Result detect(const cv::Mat& roi_bgr);

private:
    Config cfg_;
    bool initialized_ = false;

    cv::Ptr<cv::Feature2D> detector_;   ///< SIFT 或 ORB (按 cfg_.method)
    int matcher_norm_;                  ///< 描述子距离度量: SIFT→L2, ORB→HAMMING
    std::vector<cv::KeyPoint> tmpl_keypoints_;
    cv::Mat tmpl_descriptors_;
    std::vector<Eigen::Vector3d> tmpl_pts_3d_;
    std::unique_ptr<TemplateMatcher> matcher_;
};

} // namespace gpnp
