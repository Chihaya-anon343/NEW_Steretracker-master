#include "feature/OpticalFlowTracker.hpp"

#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gpnp {

OpticalFlowTracker::OpticalFlowTracker(const LKParams& params)
    : params_(params)
{
}

// ============================================================================
// 主追踪（0626：仅 LK + FB；MAD 已移至 MadDisparityFilter）
// ============================================================================

TrackResult OpticalFlowTracker::track(const cv::Mat& left_gray,
                                       const cv::Mat& right_gray,
                                       const std::vector<cv::Point2f>& pts_left) {
    TrackResult result;

    if (pts_left.empty()) {
        return result;
    }

    // 校验：LK 光流要求左右图像尺寸一致
    if (left_gray.size() != right_gray.size()) {
        throw std::invalid_argument(
            "OpticalFlowTracker: left and right images must have identical dimensions. "
            "Got left=" + std::to_string(left_gray.cols) + "x" + std::to_string(left_gray.rows) +
            ", right=" + std::to_string(right_gray.cols) + "x" + std::to_string(right_gray.rows));
    }

    const int n_pts = static_cast<int>(pts_left.size());

    // ---- 步骤1: LK 光流 L→R ----
    std::vector<cv::Point2f> pts_right;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(left_gray, right_gray, pts_left, pts_right,
                             status, err,
                             params_.winSize, params_.maxLevel,
                             params_.criteria, 0, params_.minEigThreshold);

    // ---- 步骤2: LK 光流 R→L（反向检测）----
    std::vector<cv::Point2f> pts_left_back;
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    cv::calcOpticalFlowPyrLK(right_gray, left_gray, pts_right, pts_left_back,
                             status_back, err_back,
                             params_.winSize, params_.maxLevel,
                             params_.criteria, 0, params_.minEigThreshold);

    // ---- 步骤3: 按状态 + FB 检查过滤 ----
    result.num_before_fb = n_pts;

    std::vector<cv::Point2f> pts_left_raw, pts_right_raw, pts_left_back_raw;
    std::vector<int> idx_from_raw;
    pts_left_raw.reserve(n_pts);
    pts_right_raw.reserve(n_pts);
    pts_left_back_raw.reserve(n_pts);
    idx_from_raw.reserve(n_pts);

    for (int i = 0; i < n_pts; ++i) {
        if (status[i] && status_back[i]) {
            pts_left_raw.push_back(pts_left[i]);
            pts_right_raw.push_back(pts_right[i]);
            pts_left_back_raw.push_back(pts_left_back[i]);
            idx_from_raw.push_back(i);
        }
    }

    int n_raw = static_cast<int>(pts_left_raw.size());

    // 前向-反向误差检查：||pts_left_raw - pts_left_back_raw|| < 1.0
    std::vector<bool> fb_valid(n_raw, false);
    double fb_error_sum = 0.0;
    int fb_valid_count = 0;

    for (int i = 0; i < n_raw; ++i) {
        double dx_fb = pts_left_raw[i].x - pts_left_back_raw[i].x;
        double dy_fb = pts_left_raw[i].y - pts_left_back_raw[i].y;
        double fb_err = std::sqrt(dx_fb * dx_fb + dy_fb * dy_fb);
        if (fb_err < 1.0) {
            fb_valid[i] = true;
            fb_error_sum += fb_err;
            ++fb_valid_count;
        }
    }

    result.num_after_fb = fb_valid_count;
    result.fb_error_mean = (fb_valid_count > 0) ? (fb_error_sum / fb_valid_count) : 0.0;

    // 提取 FB 验证通过的点
    std::vector<cv::Point2f> pts_left_fb, pts_right_fb;
    std::vector<int> idx_from_fb;
    pts_left_fb.reserve(fb_valid_count);
    pts_right_fb.reserve(fb_valid_count);
    idx_from_fb.reserve(fb_valid_count);

    for (int i = 0; i < n_raw; ++i) {
        if (fb_valid[i]) {
            pts_left_fb.push_back(pts_left_raw[i]);
            pts_right_fb.push_back(pts_right_raw[i]);
            idx_from_fb.push_back(idx_from_raw[i]);
        }
    }

    int n_fb = static_cast<int>(pts_left_fb.size());

    // ---- 步骤4: 透传（0626：MAD 已移至 process() 层级）----
    // 计算 dx/dy 供下游 MAD 使用，但此处不做过滤。
    std::vector<double> disparity(n_fb);
    std::vector<double> dx_filtered(n_fb);
    for (int i = 0; i < n_fb; ++i) {
        double d = static_cast<double>(pts_left_fb[i].x - pts_right_fb[i].x);
        dx_filtered[i] = d;
        disparity[i] = -d;
    }

    // ---- 构建输出（0626：全部 FB 验证点，不含 MAD）----
    result.pts_left_good = std::move(pts_left_fb);
    result.pts_right_good = std::move(pts_right_fb);
    result.dx_filtered = std::move(dx_filtered);
    result.disparity = std::move(disparity);
    result.idx_from_filtered = std::move(idx_from_fb);
    result.num_matched = static_cast<int>(result.pts_left_good.size());
    result.num_after_mad = result.num_matched; // MAD 不在此处应用

    return result;
}

} // namespace gpnp