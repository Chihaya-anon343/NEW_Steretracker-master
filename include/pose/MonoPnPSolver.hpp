#pragma once

/**
 * @file MonoPnPSolver.hpp
 * @brief 单目 EPnP 求解器 —— 纯 EPnP (RANSAC + ITERATIVE 精化)，不使用 GPNP。
 *
 * 与双目路径的 GPnPSolver 不同，本模块：
 *   - 仅依赖 2D↔3D 对应点（左图像素 ↔ 模板 3D 坐标）
 *   - 直接调用 OpenCV solvePnPRansac (EPNP) + solvePnP (ITERATIVE)
 *   - 无 warm-start、无帧间缓存、无双目射线约束
 *   - 适用于单目模式 (mono_mode)
 *
 * Module: pose
 * Input:   pts_2d (左图匹配点), pts_3d (模板 3D 坐标), K (相机内参)
 * Output:  PoseEstimate {R, t, success}
 * Dependencies: OpenCV calib3d, Eigen
 */

#include "common/Types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

class MonoPnPSolver {
public:
    MonoPnPSolver() = default;
    ~MonoPnPSolver() = default;

    // 不可拷贝
    MonoPnPSolver(const MonoPnPSolver&) = delete;
    MonoPnPSolver& operator=(const MonoPnPSolver&) = delete;

    /// 单目 EPnP 位姿估计。
    ///
    /// Pipeline:
    ///   1. 将 Eigen::Vector3d 的 3D 点转为 cv::Point3f
    ///   2. cv::solvePnPRansac (EPNP, 300 iters, 8.0px reproj, 0.99)
    ///   3. 用 inlier 子集调用 cv::solvePnP (ITERATIVE) 精化
    ///   4. 有效性校验: t[2] > 0, 10 < |t| < 20000, 所有值有限
    ///
    /// @param pts_2d  左图匹配点（像素坐标）
    /// @param pts_3d  模板 3D 坐标（毫米，z=0）
    /// @param K        3×3 相机内参矩阵
    /// @return PoseEstimate with R, t, success flag
    PoseEstimate solve(const std::vector<cv::Point2f>& pts_2d,
                       const std::vector<Eigen::Vector3d>& pts_3d,
                       const Eigen::Matrix3d& K);
};

} // namespace gpnp