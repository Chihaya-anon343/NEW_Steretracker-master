#include "matching/TemplateMatcher.hpp"

#include <opencv2/calib3d.hpp>

#include <unordered_map>

namespace gpnp {

TemplateMatcher::TemplateMatcher(double ratio_threshold, double ransac_threshold)
    : ratio_threshold_(ratio_threshold)
    , ransac_threshold_(ransac_threshold)
{
    // NORM_HAMMING 是 AKAZE 二进制描述符必需的
    bf_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false); // crossCheck=false
}

// ============================================================================
// 比例测试辅助
// ============================================================================

std::vector<cv::DMatch> TemplateMatcher::applyRatioTest(
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    double ratio_thresh)
{
    std::vector<cv::DMatch> filtered;
    for (const auto& matches : knn_matches) {
        if (matches.size() < 2) continue;
        if (matches[0].distance < ratio_thresh * matches[1].distance) {
            filtered.push_back(matches[0]);
        }
    }
    return filtered;
}

// ============================================================================
// 主匹配
// ============================================================================

MatchResult TemplateMatcher::match(
    const cv::Mat& desc_left,
    const std::vector<cv::KeyPoint>& kp_left,
    const cv::Mat& desc_template,
    const std::vector<cv::KeyPoint>& kp_template)
{
    MatchResult result;
    constexpr int kMinMatchesForHomography = 4;

    // ---- 步骤1: KNN + 比例测试（L→T）----
    std::vector<std::vector<cv::DMatch>> raw_matches_l2t;
    bf_matcher_->knnMatch(desc_left, desc_template, raw_matches_l2t, 2);

    std::vector<cv::DMatch> matches_l2t = applyRatioTest(raw_matches_l2t, ratio_threshold_);
    result.ratio_test_count = static_cast<int>(matches_l2t.size());

    // 转换为点数组
    auto buildPointArrays = [&](const std::vector<cv::DMatch>& matches) {
        std::vector<cv::Point2f> left_pts, tmpl_pts;
        left_pts.reserve(matches.size());
        tmpl_pts.reserve(matches.size());
        for (const auto& m : matches) {
            left_pts.push_back(kp_left[m.queryIdx].pt);
            tmpl_pts.push_back(kp_template[m.trainIdx].pt);
        }
        return std::make_pair(left_pts, tmpl_pts);
    };

    if (static_cast<int>(matches_l2t.size()) < kMinMatchesForHomography) {
        auto [lp, tp] = buildPointArrays(matches_l2t);
        result.good_matches = matches_l2t;
        result.pts_left_match = lp;
        result.pts_template_match = tp;
        result.cross_check_count = 0;
        result.homography_count = 0;
        result.num_matches = static_cast<int>(matches_l2t.size());
        return result;
    }

    // ---- 步骤2: 交叉检查（T→L）----
    std::vector<std::vector<cv::DMatch>> raw_matches_t2l;
    bf_matcher_->knnMatch(desc_template, desc_left, raw_matches_t2l, 2);

    std::vector<cv::DMatch> matches_t2l = applyRatioTest(raw_matches_t2l, ratio_threshold_);

    // 构建反向映射：trainIdx(T) → queryIdx(L)
    // T→L 匹配中：queryIdx = 模板索引, trainIdx = 左图索引
    std::unordered_map<int, int> t2l_map;
    t2l_map.reserve(matches_t2l.size());
    for (const auto& m : matches_t2l) {
        t2l_map[m.queryIdx] = m.trainIdx; // 模板索引 → 左图索引
    }

    // 交叉检查：L→T 匹配有效当且仅当 T→L 映射回相同的左图索引
    std::vector<cv::DMatch> cross_checked;
    cross_checked.reserve(matches_l2t.size());
    for (const auto& m : matches_l2t) {
        auto it = t2l_map.find(m.trainIdx);
        if (it != t2l_map.end() && it->second == m.queryIdx) {
            cross_checked.push_back(m);
        }
    }

    result.cross_check_count = static_cast<int>(cross_checked.size());

    if (static_cast<int>(cross_checked.size()) < kMinMatchesForHomography) {
        auto [lp, tp] = buildPointArrays(cross_checked);
        result.good_matches = cross_checked;
        result.pts_left_match = lp;
        result.pts_template_match = tp;
        result.homography_count = 0;
        result.num_matches = static_cast<int>(cross_checked.size());
        return result;
    }

    // ---- 步骤3: 单应性 RANSAC 几何一致性 ----
    auto [src_pts, dst_pts] = buildPointArrays(cross_checked);

    std::vector<uchar> inlier_mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, ransac_threshold_, inlier_mask);

    if (H.empty() || inlier_mask.empty()) {
        result.good_matches = cross_checked;
        result.pts_left_match = src_pts;
        result.pts_template_match = dst_pts;
        result.homography_count = 0;
        result.num_matches = static_cast<int>(cross_checked.size());
        return result;
    }

    // 按内点掩膜过滤
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts_left_match, pts_template_match;
    good_matches.reserve(cross_checked.size());
    pts_left_match.reserve(cross_checked.size());
    pts_template_match.reserve(cross_checked.size());

    for (size_t i = 0; i < cross_checked.size(); ++i) {
        if (inlier_mask[i]) {
            good_matches.push_back(cross_checked[i]);
            pts_left_match.push_back(src_pts[i]);
            pts_template_match.push_back(dst_pts[i]);
        }
    }

    result.homography_count = static_cast<int>(good_matches.size());
    result.good_matches = good_matches;
    result.pts_left_match = pts_left_match;
    result.pts_template_match = pts_template_match;
    result.num_matches = static_cast<int>(good_matches.size());

    return result;
}

} // namespace gpnp