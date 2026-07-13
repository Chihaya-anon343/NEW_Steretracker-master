#pragma once

/**
 * @file StereoProjector.hpp
 * @brief Stereo depth estimation and right-image projection.
 *
 * Module: stereo
 * Function: For each matched stereo pair, compute disparity → depth via
 *           triangulation, then project the 3D point into the right camera
 *           using the stereo extrinsics.
 * Input:   pts_left_good, pts_right_good, StereoCameraParams
 * Output:  ProjectionResult {pts_right_projected, pts_left_used, pts_right_used}
 * Dependencies: GeometryUtils.hpp, Types.hpp
 * Relations: Used by StereoTracker; output used for visualization only
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

namespace gpnp {

class StereoProjector {
public:
    /**
     * @brief Construct stereo projector.
     * @param params  Stereo camera parameters (K, K_inv, R_rl, t_rl, focal_length, baseline)
     */
    explicit StereoProjector(const StereoCameraParams& params);

    ~StereoProjector() = default;

    // 值语义（存储参数副本，非引用）
    StereoProjector(const StereoProjector&) = default;
    StereoProjector& operator=(const StereoProjector&) = default;
    StereoProjector(StereoProjector&&) noexcept = default;
    StereoProjector& operator=(StereoProjector&&) noexcept = default;

    /**
     * @brief Project left-image points to the right image via depth triangulation.
     *
     * For each matched pair (uL,vL) ↔ (uR,vR):
     *   1. Compute disparity = uL - uR (must be >= 0.01 px)
     *   2. Compute depth = focal_length * baseline / |disparity| (in meters)
     *   3. Filter: depth > 0 && depth <= 100.0 meters
     *   4. Compute 3D point in left camera frame: P_left = ray_left * depth
     *   5. Transform to right camera: P_right = R_rl^T * (P_left - t_rl)
     *   6. Project to right image: pts_right_projected[i] = K * P_right / z
     *
     * @param pts_left   Matched left image points
     * @param pts_right  Matched right image points
     * @return ProjectionResult with projected points and valid subsets
     */
    ProjectionResult project(const std::vector<cv::Point2f>& pts_left,
                             const std::vector<cv::Point2f>& pts_right);

private:
    StereoCameraParams params_; ///< 相机参数（按值存储）
};

} // namespace gpnp
