#pragma once

/**
 * @file GPnPSolver.hpp
 * @brief Generalized PnP solver using stereo ray constraints.
 *
 * Module: pose
 * Function: Estimate camera pose by minimizing the cross product between
 *           reconstructed 3D points and stereo camera rays.
 *
 *           For each template point matched to a stereo pair:
 *             left ray:  origin=(0,0,0),  direction=K⁻¹·(uL,vL,1) normalized
 *             right ray: origin=t_rl,     direction=R_rl·(K⁻¹·(uR,vR,1) normalized)
 *
 *           Residual: cross(R·pt_3d + t - origin, direction) → 3 values per ray
 *           Total residuals: 6 × n_pts for 7 parameters [qw,qx,qy,qz,tx,ty,tz]
 *
 *           Optimized via Eigen::Levenberg-Marquardt (replaces scipy.optimize.least_squares).
 *
 * Input:   PipelineResult (matches, stereo points, disparity), template 3D points,
 *          StereoCameraParams, warm-start pose (optional)
 * Output:  PoseEstimate {R, t, success, n_pts} + GPNPMonitor
 * Dependencies: Eigen (unsupported/NonLinearOptimization), GeometryUtils.hpp
 * Relations: Called by StereoTracker each frame; uses warm-start from cache
 */

#include "common/Types.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vector>

namespace gpnp {

class GPnPSolver {
public:
    /**
     * @brief Construct GPNP solver.
     * @param params  Stereo camera parameters
     * @param gpnp_min_pts  Minimum intersection points for GPNP
     */
    GPnPSolver(const StereoCameraParams& params, int gpnp_min_pts = 4);

    ~GPnPSolver();

    // 可移动（按值存储，安全移动）
    GPnPSolver(GPnPSolver&&) noexcept = default;
    GPnPSolver& operator=(GPnPSolver&&) noexcept = default;

    // 不可拷贝（持有内部状态）
    GPnPSolver(const GPnPSolver&) = delete;
    GPnPSolver& operator=(const GPnPSolver&) = delete;

    // ========================================================================
    // 位姿估计
    // ========================================================================

    /**
     * @brief Solve GPNP for camera pose.
     *
     * Pipeline:
     *   1. Build queryIdx→trainIdx map from good_matches
     *   2. Intersect idx_from_filtered (stereo-matched) with template-matched indices
     *   3. Build stereo rays for intersected points
     *   4. Compute depth guess from median disparity
     *   5. Run Levenberg-Marquardt optimization
     *   6. Extract and validate pose
     *
     * @param result         Pipeline result with all intermediate data
     * @param pts_3d_T       Template 3D coordinates (mm, z=0)
     * @param R_init         Warm-start rotation (3×3, optional)
     * @param t_init         Warm-start translation (mm, optional)
     * @param timing_ms_out  [out] Optimization timing in ms
     * @return PoseEstimate with R, t, success flag
     */
    PoseEstimate solve(PipelineResult& result,
                       const std::vector<Eigen::Vector3d>& pts_3d_T,
                       const Eigen::Matrix3d* R_init,
                       const Eigen::Vector3d* t_init,
                       double& timing_ms_out);

    /**
     * @brief Get the most recent GPNP optimization monitor.
     */
    const GPNPMonitor& lastMonitor() const { return monitor_; }

private:
    StereoCameraParams params_;  ///< 按值存储（支持移动语义）
    int gpnp_min_pts_;
    GPNPMonitor monitor_;

    /// 在失败时打印监视数据到标准输出（匹配 Python _print_gpnp_monitor）。
    void printMonitor() const;
};

} // namespace gpnp
