#pragma once

/**
 * @file GeometryUtils.hpp
 * @brief 纯头文件几何工具函数。
 *
 * 用 Eigen 替代 Python numpy/scipy.spatial.transform 的操作。
 * 所有函数均为 inline，支持纯头文件使用。
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpnp {

// ============================================================================
// 相机几何
// ============================================================================

/**
 * @brief 将像素坐标转换为归一化相机射线。
 *
 * 等价于 Python:
 *   ray = K_inv @ [u, v, 1]
 *   return ray / norm(ray)
 *
 * @param u        像素 x 坐标
 * @param v        像素 y 坐标
 * @param K_inv    3×3 相机内参逆矩阵
 * @return 相机坐标系中的单位方向向量
 */
inline Eigen::Vector3d pixelToCameraRay(double u, double v, const Eigen::Matrix3d& K_inv) {
    Eigen::Vector3d ray = K_inv * Eigen::Vector3d(u, v, 1.0);
    return ray.normalized();
}

/**
 * @brief 批量像素坐标 → 归一化相机射线方向。
 *
 * 等价于 Python 0626 _batch_pixel_to_camera_ray 的向量化版本:
 *   homo = [pixels, ones(N,1)]
 *   rays = homo @ K_inv.T
 *   return rays / norm(rays, axis=1)
 *
 * @param pixels   N×2 矩阵，每行为 (u,v) 像素坐标
 * @param K_inv    3×3 相机内参逆矩阵
 * @param rays_out [输出] N×3 矩阵，每行为单位方向向量
 */
inline void batchPixelToCameraRay(const Eigen::MatrixX2d& pixels,
                                   const Eigen::Matrix3d& K_inv,
                                   Eigen::MatrixX3d& rays_out) {
    const int N = static_cast<int>(pixels.rows());
    // 构建齐次坐标: [pixels | ones(N,1)]
    Eigen::MatrixXd homo(N, 3);
    homo.leftCols<2>() = pixels;
    homo.col(2).setOnes();
    // rays = homo @ K_inv.T  (N×3)
    rays_out = homo * K_inv.transpose();
    // 逐行归一化
    for (int i = 0; i < N; ++i) {
        double n = rays_out.row(i).norm();
        if (n > 1e-12) rays_out.row(i) /= n;
    }
}

/**
 * @brief 将相机坐标系中的 3D 点投影到图像坐标。
 *
 * 等价于 Python:
 *   uv = K @ P_cam
 *   return uv[:2] / uv[2]
 *
 * @param P_cam  相机坐标系中的 3D 点
 * @param K      3×3 相机内参矩阵
 * @return (u, v) 像素坐标
 */
inline Eigen::Vector2d projectToImage(const Eigen::Vector3d& P_cam, const Eigen::Matrix3d& K) {
    Eigen::Vector3d uv = K * P_cam;
    return Eigen::Vector2d(uv.x() / uv.z(), uv.y() / uv.z());
}

// ============================================================================
// 四元数 / 旋转工具
// ============================================================================

/**
 * @brief 将旋转矩阵转换为四元数 [x, y, z, w]（scipy 约定）。
 *
 * @param R  3×3 旋转矩阵
 * @return [qx, qy, qz, qw]，scipy 兼容顺序
 */
inline Eigen::Vector4d rotationMatrixToQuat(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q(R);
    // Eigen 内部存储为 [x, y, z, w]；coeffs() 返回 [x, y, z, w]
    return q.coeffs(); // [x, y, z, w] — 匹配 scipy 约定
}

/**
 * @brief 将四元数 [x, y, z, w] 转换为旋转矩阵。
 *
 * @param q_vec  [qx, qy, qz, qw]，scipy 兼容顺序
 * @return 3×3 旋转矩阵
 */
inline Eigen::Matrix3d quatToRotationMatrix(const Eigen::Vector4d& q_vec) {
    Eigen::Quaterniond q(q_vec(3), q_vec(0), q_vec(1), q_vec(2)); // w, x, y, z
    return q.toRotationMatrix();
}

/**
 * @brief 归一化四元数并确保标量部分为正。
 *
 * 等价于 Python:
 *   q = q / np.linalg.norm(q)
 *   if q[3] < 0: q = -q
 *
 * @param q_vec  [qx, qy, qz, qw]，scipy 兼容顺序
 * @return 归一化后的四元数，qw >= 0
 */
inline Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d& q_vec) {
    double norm = q_vec.norm();
    Eigen::Vector4d qn = q_vec / norm;
    if (qn(3) < 0.0) { // qw < 0
        qn = -qn;
    }
    return qn;
}

// ============================================================================
// 统计工具（匹配 numpy 行为）
// ============================================================================

/**
 * @brief 计算向量的中位数（匹配 numpy.median，对奇数/偶数长度都适用）。
 *
 * 对于偶数长度向量，返回中间两个元素的平均值。
 *
 * @param values  输入值（拷贝，不原地修改）
 * @return 中位数值，空向量时返回 0.0
 */
inline double computeMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;

    size_t n = values.size();
    size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    double mid_val = values[mid];

    if (n % 2 == 0) {
        // 中间两个元素的平均值（匹配 numpy 行为）
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1), values.end());
        return (values[mid - 1] + mid_val) / 2.0;
    }
    return mid_val;
}

/**
 * @brief 计算中位数绝对偏差: median(|values - median(values)|)。
 *
 * @param values  输入值（拷贝）
 * @return MAD 值，空向量时返回 0.0
 */
inline double computeMAD(std::vector<double> values) {
    if (values.empty()) return 0.0;

    double med = computeMedian(values);
    for (auto& v : values) {
        v = std::abs(v - med);
    }
    return computeMedian(std::move(values));
}

/**
 * @brief 计算每个元素相对于中位数的绝对偏差。
 *
 * @param values  输入值
 * @return |values[i] - median(values)| 的向量
 */
inline std::vector<double> absDeviation(const std::vector<double>& values) {
    if (values.empty()) return {};
    double med = computeMedian(values);
    std::vector<double> result;
    result.reserve(values.size());
    for (double v : values) {
        result.push_back(std::abs(v - med));
    }
    return result;
}

} // namespace gpnp