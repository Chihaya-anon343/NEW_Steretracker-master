#pragma once

/**
 * @file OpticalFlowTracker.hpp
 * @brief Lucas-Kanade 光流追踪，包含前后向检查
 *        和基于 MAD 的离群点过滤。
 *
 * 模块：feature
 * 功能：通过 LK 光流追踪从左到右立体图像的特征，
 *        使用前后向检查验证，并利用中位数绝对偏差 (MAD)
 *        对视差分量进行离群点过滤。
 * 输入：left_gray, right_gray (CV_8UC1), pts_left (vector<Point2f>)
 * 输出：TrackResult {pts_left_good, pts_right_good, idx_from_filtered, disparity}
 * 依赖：OpenCV video/tracking, GeometryUtils.hpp
 * 关系：由 StereoTracker 使用；输出供给 StereoProjector 和 GPnPSolver
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

namespace gpnp {

class OpticalFlowTracker {
public:
    /**
     * @brief 构造光流追踪器。
     * @param params  Lucas-Kanade 参数
     */
    explicit OpticalFlowTracker(const LKParams& params = LKParams{});

    ~OpticalFlowTracker() = default;

    // ========================================================================
    // 主追踪 API
    // ========================================================================

    /**
     * @brief 从左到右图像追踪特征。
     *
     * 流水线：
     *   1. LK 光流 左→右
     *   2. LK 光流 右→左（反向）
     *   3. 前后向一致性检查（误差 < 1.0 像素）
     *   4. 对 dx（仅正视差）和 dy 进行 MAD 离群点过滤
     *   5. 如果过滤后点数 < 3，则优雅降级
     *
     * @param left_gray   左相机灰度图像
     * @param right_gray  右相机灰度图像
     * @param pts_left    左图像特征点（N×1×2 或 N×2 排列）
     * @return 包含有效立体对应关系的 TrackResult
     */
    TrackResult track(const cv::Mat& left_gray,
                      const cv::Mat& right_gray,
                      const std::vector<cv::Point2f>& pts_left);

    // ========================================================================
    // 访问器
    // ========================================================================

    const LKParams& params() const { return params_; }

private:
    LKParams params_;

    /// 计算 vector<double> 的中位数（内部用于 MAD）。
    static double median(std::vector<double> values);

    /// 计算 MAD：median(|x - median(x)|)。
    static double mad(std::vector<double> values);
};

} // namespace gpnp
