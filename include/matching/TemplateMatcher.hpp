#pragma once

/**
 * @file TemplateMatcher.hpp
 * @brief Template feature matching with 3-stage geometric verification.
 *
 * Module: matching
 * Function: Match left-image AKAZE features against cached template features
 *           through a cascaded filter:
 *             1. KNN ratio test (L→T, threshold 0.75)
 *             2. Cross-check (T→L ratio test + symmetry verification)
 *             3. Homography RANSAC (5.0 px threshold)
 * Input:   desc_left, kp_left, template descriptors, template keypoints
 * Output:  MatchResult {good_matches, pts_left_match, pts_template_match, stats}
 * Dependencies: OpenCV features2d, calib3d
 * Relations: Used by StereoTracker; output feeds GPnPSolver and InitialPnPSolver
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

namespace gpnp {

class TemplateMatcher {
public:
    /**
     * @brief Construct template matcher.
     * @param ratio_threshold  Lowe's ratio test threshold (default 0.75)
     * @param ransac_threshold Homography RANSAC reprojection error (default 5.0 px)
     */
    explicit TemplateMatcher(double ratio_threshold = 0.75,
                             double ransac_threshold = 5.0);

    ~TemplateMatcher() = default;

    // ========================================================================
    // 主匹配 API
    // ========================================================================

    /**
     * @brief Match left-image features to template features.
     *
     * Three-stage cascaded filter:
     *   1. KNN ratio test (L→T): for each left descriptor, find 2 nearest
     *      template matches. Keep if distance1 < ratio * distance2.
     *   2. Cross-check (T→L): same ratio test in reverse. Verify symmetry:
     *      a match (l→t) is valid iff t→l maps back to the same l.
     *   3. Homography RANSAC: compute homography from remaining matches,
     *      keep only inliers.
     *
     * Each stage requires at least 4 matches; falls through early if not met.
     *
     * @param desc_left    Left image AKAZE descriptors (CV_8U)
     * @param kp_left      Left image keypoints
     * @param desc_template Template AKAZE descriptors (CV_8U, cached)
     * @param kp_template  Template keypoints (cached)
     * @return MatchResult with filtered matches and point arrays
     */
    MatchResult match(const cv::Mat& desc_left,
                      const std::vector<cv::KeyPoint>& kp_left,
                      const cv::Mat& desc_template,
                      const std::vector<cv::KeyPoint>& kp_template);

    // ========================================================================
    // 访问器
    // ========================================================================

    double ratioThreshold() const { return ratio_threshold_; }
    double ransacThreshold() const { return ransac_threshold_; }

private:
    double ratio_threshold_;
    double ransac_threshold_;
    cv::Ptr<cv::BFMatcher> bf_matcher_; ///< 暴力匹配器，使用汉明距离

    /// 对 KNN 匹配结果应用 Lowe 比率测试。
    static std::vector<cv::DMatch> applyRatioTest(
        const std::vector<std::vector<cv::DMatch>>& knn_matches,
        double ratio_thresh);
};

} // namespace gpnp
