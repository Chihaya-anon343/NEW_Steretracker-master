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

    /// 带时序 seed 的位姿估计（序列模式位姿链）。
    ///
    /// seed 非空时新增候选 W: solvePnP(ITERATIVE, useExtrinsicGuess=true)，
    /// 初值取 seed，与 EPnP-RANSAC/IPPE 候选同池竞争、同一有效性校验。
    /// 择优两阶段: 先取重投影最小值 reproj_min，重投影 ≤ reproj_min + tie_epsilon_px
    /// 的候选构成平票集；seed 存在时平票集内取与 seed 几何距离最小者
    /// （平面 IPPE 二义性下稳定在同一分支），否则取重投影最小者（原行为）。
    ///
    /// n==4 且 seed 非空时（2026-09 棘轮冻结修复）：验收容差收紧为相对容差
    /// clamp(5%×2D跨度, 1, 8) px（8px 平坦阈值在 15~30px 小目标上相当于
    /// 30~50% 相对容差，陈旧 seed 解恒通过会导致深度冻结），候选为
    /// SeedITER + 冷启动 ITERATIVE + IPPE 同池竞争择优；全部候选未过
    /// 相对容差时回退旧平坦 8px 路径（不劣于原行为）。
    ///
    /// @param seed            上帧位姿种子（nullptr = 与无 seed 版本等价）
    /// @param tie_epsilon_px  平票窗口 (px)，<=0 时退化为纯重投影最小
    PoseEstimate solve(const std::vector<cv::Point2f>& pts_2d,
                       const std::vector<Eigen::Vector3d>& pts_3d,
                       const Eigen::Matrix3d& K,
                       const PoseSeed* seed,
                       double tie_epsilon_px);
};

} // namespace gpnp