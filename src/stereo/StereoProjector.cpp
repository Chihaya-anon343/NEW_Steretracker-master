#include "stereo/StereoProjector.hpp"
#include "common/GeometryUtils.hpp"

#include <Eigen/Dense>

#include <cmath>

namespace gpnp {

StereoProjector::StereoProjector(const StereoCameraParams& params)
    : params_(params) // 值拷贝存储
{
}

// ============================================================================
// 投影（0626：向量化批量矩阵运算）
// ============================================================================

ProjectionResult StereoProjector::project(
    const std::vector<cv::Point2f>& pts_left,
    const std::vector<cv::Point2f>& pts_right)
{
    ProjectionResult result;

    const int N = static_cast<int>(pts_left.size());
    if (N == 0) return result;

    const double f = params_.focal_length;
    const double b = params_.baseline;
    const Eigen::Matrix3d& K = params_.K;
    const Eigen::Matrix3d& K_inv = params_.K_inv;
    const Eigen::Matrix3d& R_rl = params_.R_rl;
    const Eigen::Vector3d& t_rl = params_.t_rl;

    // ---- 步骤1：计算所有点的视差和深度 ----
    Eigen::VectorXd disp(N);
    Eigen::VectorXd depth(N);
    std::vector<bool> valid_depth(N, false);

    for (int i = 0; i < N; ++i) {
        double d = static_cast<double>(pts_left[i].x - pts_right[i].x);
        disp(i) = d;
        if (std::abs(d) >= 0.01) {
            double z = f * b / std::abs(d);
            if (z > 0.0 && z <= 100.0) {
                depth(i) = z;
                valid_depth[i] = true;
            }
        }
    }

    // 统计有效点数
    int n_valid = 0;
    for (bool v : valid_depth) if (v) ++n_valid;
    if (n_valid == 0) return result;

    // ---- 步骤2：批量光线计算 ----
    // 构建齐次像素矩阵：[pts_left | 1(N,1)]
    Eigen::MatrixXd uv_hom(N, 3);
    for (int i = 0; i < N; ++i) {
        uv_hom(i, 0) = pts_left[i].x;
        uv_hom(i, 1) = pts_left[i].y;
        uv_hom(i, 2) = 1.0;
    }
    // 光线 = uv_hom @ K_inv.T  →  N×3
    Eigen::MatrixXd rays = uv_hom * K_inv.transpose();
    // 逐行归一化
    for (int i = 0; i < N; ++i) {
        double n = rays.row(i).norm();
        if (n > 1e-16) rays.row(i) /= n;
    }

    // ---- 步骤3：左相机坐标 = 方向 * 深度（广播）----
    Eigen::MatrixXd P_left(N, 3);
    for (int i = 0; i < N; ++i) {
        P_left.row(i) = rays.row(i) * depth(i);
    }

    // ---- 步骤4：右相机坐标 = (P_left - t_rl) @ R_rl ----
    // R_rl.T @ (P_left - t_rl) ≡ (P_left - t_rl) @ R_rl
    Eigen::RowVector3d t_vec = t_rl.transpose();
    Eigen::MatrixXd P_rel = P_left.rowwise() - t_vec;
    Eigen::MatrixXd P_right = P_rel * R_rl;

    // ---- 步骤5：投影到右图（批量）----
    // uv_h_right = (K @ P_right.T).T  →  N×3
    Eigen::MatrixXd uv_h_right = (K * P_right.transpose()).transpose();

    // 除以深度z，过滤无效投影
    std::vector<cv::Point2f> pts_right_proj(N);
    std::vector<bool> valid_proj(N, false);

    for (int i = 0; i < N; ++i) {
        if (!valid_depth[i]) continue;
        double w = uv_h_right(i, 2);
        if (std::abs(w) > 1e-8) {
            pts_right_proj[i] = cv::Point2f(
                static_cast<float>(uv_h_right(i, 0) / w),
                static_cast<float>(uv_h_right(i, 1) / w));
            valid_proj[i] = true;
        }
    }

    // ---- 步骤6：按有效投影过滤（范数 > 1e-6）----
    constexpr double kEps = 1e-6;
    result.valid_mask.assign(N, false);
    for (int i = 0; i < N; ++i) {
        if (valid_proj[i]) {
            double norm_sq = pts_right_proj[i].x * pts_right_proj[i].x
                           + pts_right_proj[i].y * pts_right_proj[i].y;
            if (norm_sq > kEps) {
                result.valid_mask[i] = true;
                result.pts_right_projected.push_back(pts_right_proj[i]);
                result.pts_left_used.push_back(pts_left[i]);
                result.pts_right_used.push_back(pts_right[i]);
            }
        }
    }

    result.num_projected = static_cast<int>(result.pts_right_projected.size());
    return result;
}

} // namespace gpnp
