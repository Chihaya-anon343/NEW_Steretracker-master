#include "feature/SiftKeypointExtractor.hpp"

#include "matching/TemplateMatcher.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace gpnp {

SiftKeypointExtractor::SiftKeypointExtractor(const Config& cfg)
    : cfg_(cfg)
    , matcher_norm_(cv::NORM_L2)
{
    if (cfg_.scale <= 0.0 || cfg_.scale > 1.0) {
        throw std::invalid_argument("SiftKeypointExtractor: scale must be in (0, 1]");
    }
    if (cfg_.method == "orb") {
        // ORB 描述子为二进制 → NORM_HAMMING
        detector_ = cv::ORB::create(cfg_.nfeatures > 0 ? cfg_.nfeatures : 500);
        matcher_norm_ = cv::NORM_HAMMING;
    } else {
        // SIFT 描述子为 CV_32F 浮点向量 → NORM_L2
        detector_ = cv::SIFT::create(cfg_.nfeatures);
        matcher_norm_ = cv::NORM_L2;
    }
}

SiftKeypointExtractor::~SiftKeypointExtractor() = default;

// ============================================================================
// 初始化: 模板特征预提取 (仿 AkazeExtractor::extractTemplate)
// ============================================================================

bool SiftKeypointExtractor::initialize() {
    if (initialized_) return true;

    if (cfg_.template_path.empty()) {
        std::cerr << "[SiftKeypoint] 未配置 template_path, 无法初始化" << std::endl;
        return false;
    }
    if (cfg_.template_real_width_mm <= 0.0 || cfg_.template_real_height_mm <= 0.0) {
        std::cerr << "[SiftKeypoint] 模板物理尺寸无效 ("
                  << cfg_.template_real_width_mm << " x "
                  << cfg_.template_real_height_mm << " mm)" << std::endl;
        return false;
    }

    cv::Mat tmpl_gray = cv::imread(cfg_.template_path, cv::IMREAD_GRAYSCALE);
    if (tmpl_gray.empty()) {
        std::cerr << "[SiftKeypoint] 模板图像读取失败: " << cfg_.template_path << std::endl;
        return false;
    }

    // 模板以原始分辨率提取特征 (不下采样)
    detector_->detectAndCompute(tmpl_gray, cv::noArray(), tmpl_keypoints_, tmpl_descriptors_);
    if (tmpl_keypoints_.size() < static_cast<size_t>(cfg_.min_pts)) {
        std::cerr << "[SiftKeypoint] 模板特征点过少: " << tmpl_keypoints_.size() << std::endl;
        return false;
    }

    // 3D 坐标: 中心化平面点 (与 AKAZE 模板同公式, 模板物理中心为原点, z=0)
    const double tw = static_cast<double>(tmpl_gray.cols);
    const double th = static_cast<double>(tmpl_gray.rows);
    const double cx = cfg_.template_real_width_mm / 2.0;
    const double cy = cfg_.template_real_height_mm / 2.0;
    tmpl_pts_3d_.reserve(tmpl_keypoints_.size());
    for (const auto& kp : tmpl_keypoints_) {
        Eigen::Vector3d pt;
        pt.x() = kp.pt.x / tw * cfg_.template_real_width_mm - cx;
        pt.y() = kp.pt.y / th * cfg_.template_real_height_mm - cy;
        pt.z() = 0.0;
        tmpl_pts_3d_.push_back(pt);
    }

    matcher_ = std::make_unique<TemplateMatcher>(cfg_.ratio_threshold, 5.0, matcher_norm_);

    initialized_ = true;
    std::cout << "[SiftKeypoint:" << cfg_.method << "] 初始化完成: 模板 " << tw << "x" << th
              << " px, 特征点 " << tmpl_keypoints_.size() << " 个 ("
              << cfg_.template_path << ")" << std::endl;
    return true;
}

// ============================================================================
// 检测: ROI 图像 → SIFT → 模板匹配 → 2D/3D 对应 (仿 AkazeExtractor::extract)
// ============================================================================

SiftKeypointExtractor::Result SiftKeypointExtractor::detect(const cv::Mat& roi_bgr) {
    Result result;
    if (!initialized_) {
        result.message = "not initialized";
        return result;
    }

    if (roi_bgr.empty()) {
        result.message = "empty input";
        return result;
    }

    cv::Mat gray;
    if (roi_bgr.channels() == 3)
        cv::cvtColor(roi_bgr, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roi_bgr;

    cv::Mat working_img;
    if (cfg_.scale < 1.0) {
        cv::resize(gray, working_img, cv::Size(), cfg_.scale, cfg_.scale, cv::INTER_LINEAR);
    } else {
        working_img = gray; // 浅拷贝
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(working_img, cv::noArray(), keypoints, descriptors);
    if (cfg_.scale < 1.0) {
        double inv_scale = 1.0 / cfg_.scale;
        for (auto& kp : keypoints) {
            kp.pt.x *= inv_scale;
            kp.pt.y *= inv_scale;
            kp.size *= inv_scale;
        }
    }

    if (static_cast<int>(keypoints.size()) < cfg_.min_pts) {
        std::ostringstream os;
        os << "insufficient keypoints: " << keypoints.size();
        result.message = os.str();
        return result;
    }

    // 三阶段匹配: ratio test → cross-check → Homography RANSAC
    MatchResult mr = matcher_->match(descriptors, keypoints,
                                     tmpl_descriptors_, tmpl_keypoints_);

    for (const auto& m : mr.good_matches) {
        result.pts_2d.push_back(keypoints[m.queryIdx].pt);
        result.pts_3d.push_back(tmpl_pts_3d_[m.trainIdx]);
    }

    if (static_cast<int>(result.pts_2d.size()) < cfg_.min_pts) {
        std::ostringstream os;
        os << "insufficient matches: " << result.pts_2d.size();
        result.message = os.str();
        result.pts_2d.clear();
        result.pts_3d.clear();
        return result;
    }

    result.success = true;
    std::ostringstream os;
    os << "matches=" << result.pts_2d.size()
       << " (ratio=" << mr.ratio_test_count
       << ", cross=" << mr.cross_check_count
       << ", homo=" << mr.homography_count << ")";
    result.message = os.str();
    return result;
}

} // namespace gpnp
