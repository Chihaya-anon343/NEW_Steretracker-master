#pragma once

/**
 * @file MadDisparityFilter.hpp
 * @brief 基于中位数绝对偏差（MAD）的视差异常点滤波器（Python 0626 算法）。
 *
 * 模块: feature
 * 功能: 通过视差一致性过滤立体匹配点对。
 *         dx: 中位数绝对偏差（MAD）滤波
 *         dy: 仅对 dx 有效子集进行 2-sigma 滤波
 * 输入:   pts_left, pts_right (N×2), idx_from_good (N个整数)
 * 输出:   包含过滤后点集、视差、掩膜等的 MadFilterResult
 * 依赖: Types.hpp, GeometryUtils.hpp
 * 关系: 由 StereoTracker 在 process() 层级调用（全图坐标）
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

class MadDisparityFilter {
public:
    /**
     * @brief 构造 MAD 视差滤波器。
     * @param min_pts_threshold  应用滤波所需的最小点数（低于此阈值则跳过）
     * @param mad_factor         dx 阈值使用的 MAD 乘数（默认 3.0）
     */
    explicit MadDisparityFilter(int min_pts_threshold = 3, double mad_factor = 3.0);

    ~MadDisparityFilter() = default;

    // ========================================================================
    // Filtering
    // ========================================================================

    /**
     * @brief 对立体匹配点对应用 MAD 视差滤波。
     *
     * 算法（与 Python 0626 MadDisparityFilter.filter 一致）:
     *   1. 计算 dx = left.x - right.x, dy = left.y - right.y
     *   2. dx 滤波: MAD (|dx - median(dx)| > mad_factor * max(mad(dx), 1.0)) 且 dx > 0
     *   3. dy 滤波: 2-sigma (|dy - mean(dy[dx_valid])| > 2.5 * max(std(dy[dx_valid]), 0.5))
     *   4. 组合掩膜: dx_valid AND dy_valid
     *   5. 若过滤后点数 < min_pts_threshold，则优雅降级回退
     *
     * @param pts_left       左图匹配点 (N×2, 全图坐标)
     * @param pts_right      右图匹配点 (N×2, 全图坐标)
     * @param idx_from_good  映射到原始 AKAZE 关键点的索引
     * @return 包含过滤后点集、视差、掩膜、索引的 MadFilterResult
     */
    MadFilterResult filter(const std::vector<cv::Point2f>& pts_left,
                           const std::vector<cv::Point2f>& pts_right,
                           const std::vector<int>& idx_from_good);

private:
    int min_pts_threshold_;
    double mad_factor_;

    /// 计算向量的均值。
    static double mean(const std::vector<double>& v);

    /// 计算标准差（总体标准差，与 numpy 默认行为一致）。
    static double stdev(const std::vector<double>& v, double mean_val);
};

} // namespace gpnp
