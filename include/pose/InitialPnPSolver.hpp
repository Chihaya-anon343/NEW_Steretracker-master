#pragma once

/**
 * @file InitialPnPSolver.hpp
 * @brief RANSAC+ITERATIVE PnP for first-frame pose initialization.
 *
 * Module: pose
 * Function: Estimate initial camera pose from template 3D↔2D correspondences
 *           using RANSAC PnP + refinement, with validity checks:
 *             - t[2] > 0 (camera in front of template plane)
 *             - 10mm < |t| < 20000mm
 *             - All values finite
 * Input:   MatchResult, template 3D points (Eigen::Vector3d), K, gpnp_min_pts
 * Output:  PoseEstimate {R, t, success}
 * Dependencies: OpenCV calib3d, Eigen
 * Relations: Called by StereoTracker on first frame; provides warm-start for GPnPSolver
 */

#include "common/Types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

class InitialPnPSolver {
public:
    /**
     * @brief Construct initial PnP solver.
     * @param gpnp_min_pts  Minimum points required for PnP
     */
    explicit InitialPnPSolver(int gpnp_min_pts = 4);

    ~InitialPnPSolver() = default;

    // ========================================================================
    // 位姿估计
    // ========================================================================

    /**
     * @brief Solve initial camera pose via RANSAC PnP + ITERATIVE refinement.
     *
     * Pipeline:
     *   1. Extract 3D↔2D correspondences from matches
     *   2. cv::solvePnPRansac (ITERATIVE, 300 iters, 8.0px reproj error, 0.99 confidence)
     *   3. cv::solvePnP refinement with inlier subset
     *   4. Validity checks on result
     *
     * Falls back to identity rotation + hardcoded translation on failure.
     *
     * @param match_result  Template matching result (good_matches, pts_left_match)
     * @param pts_3d_T      Template 3D coordinates (mm, z=0)
     * @param K             3×3 camera intrinsic matrix
     * @return PoseEstimate with R, t, and success flag
     */
    PoseEstimate solve(const MatchResult& match_result,
                       const std::vector<Eigen::Vector3d>& pts_3d_T,
                       const Eigen::Matrix3d& K);

private:
    int gpnp_min_pts_;
};

} // namespace gpnp
