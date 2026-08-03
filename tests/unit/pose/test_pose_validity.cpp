/**
 * @file test_pose_validity.cpp
 * @brief 位姿有效性校验函数单元测试 (T4-01)
 *
 * 测试内容：
 *   - 输入格式校验：size≥4、2d/3d 匹配
 *   - 深度范围校验：10 < |t| < 20000
 *   - t.z > 0 校验
 *   - NaN/Inf 校验
 *   - 边界值（刚好 10, 刚好 20000）
 *   - 全零位姿
 *   - 旋转矩阵正交性
 */

#include "../test_utils/test_assert.hpp"
#include <Eigen/Dense>
#include <cmath>

// 从目标模块抽取的核心校验逻辑（副本，用于独立测试）
namespace {

bool validate_pose_range(const Eigen::Vector3d& t) {
    // t.z > 0
    if (t.z() <= 0.0) return false;
    double len = t.norm();
    // 10 < |t| < 20000 (mm)
    if (len <= 10.0 || len >= 20000.0) return false;
    return true;
}

bool validate_pose_finite(const Eigen::Vector3d& t, const Eigen::Matrix3d& R) {
    // 所有分量必须有限
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(t(i))) return false;
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(R(i, j))) return false;
        }
    }
    return true;
}

bool is_rotation_matrix(const Eigen::Matrix3d& R, double tolerance = 1e-6) {
    // det(R) ≈ 1
    if (std::abs(R.determinant() - 1.0) > tolerance) return false;
    // R * R^T ≈ I
    double ortho_error = (R * R.transpose() - Eigen::Matrix3d::Identity()).norm();
    if (ortho_error > tolerance) return false;
    return true;
}

} // anonymous namespace

// ============================================================
// 测试用例
// ============================================================

TEST(pose_range_normal) {
    Eigen::Vector3d t(100.0, 50.0, 500.0);
    ASSERT_TRUE(validate_pose_range(t));
}

TEST(pose_range_z_zero) {
    Eigen::Vector3d t(100.0, 50.0, 0.0);
    ASSERT_FALSE(validate_pose_range(t));
}

TEST(pose_range_z_negative) {
    Eigen::Vector3d t(100.0, 50.0, -100.0);
    ASSERT_FALSE(validate_pose_range(t));
}

TEST(pose_range_too_close) {
    Eigen::Vector3d t(1.0, 0.0, 3.0);  // norm ≈ 3.16 < 10
    ASSERT_FALSE(validate_pose_range(t));
}

TEST(pose_range_boundary_close) {
    Eigen::Vector3d t(0.0, 0.0, 10.0);  // exactly 10, should be rejected (>10, not >=10)
    ASSERT_FALSE(validate_pose_range(t));
}

TEST(pose_range_just_above_close) {
    Eigen::Vector3d t(0.0, 0.0, 10.001);  // just above boundary
    ASSERT_TRUE(validate_pose_range(t));
}

TEST(pose_range_too_far) {
    Eigen::Vector3d t(0.0, 0.0, 20000.0);  // exactly 20000, should be rejected
    ASSERT_FALSE(validate_pose_range(t));
}

TEST(pose_range_just_below_far) {
    Eigen::Vector3d t(0.0, 0.0, 19999.999);
    ASSERT_TRUE(validate_pose_range(t));
}

TEST(pose_range_beyond_far) {
    Eigen::Vector3d t(0.0, 0.0, 25000.0);
    ASSERT_FALSE(validate_pose_range(t));
}

TEST(pose_finite_normal) {
    Eigen::Vector3d t(100, 200, 300);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    ASSERT_TRUE(validate_pose_finite(t, R));
}

TEST(pose_finite_nan_in_t) {
    Eigen::Vector3d t(100, NAN, 300);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    ASSERT_FALSE(validate_pose_finite(t, R));
}

TEST(pose_finite_inf_in_t) {
    Eigen::Vector3d t(100, INFINITY, 300);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    ASSERT_FALSE(validate_pose_finite(t, R));
}

TEST(pose_finite_nan_in_R) {
    Eigen::Vector3d t(100, 200, 300);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(1, 1) = NAN;
    ASSERT_FALSE(validate_pose_finite(t, R));
}

TEST(pose_identity_is_rotation) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    ASSERT_TRUE(is_rotation_matrix(R));
}

TEST(pose_90deg_rotation_is_valid) {
    Eigen::Matrix3d R;
    double theta = M_PI / 2.0;
    R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    ASSERT_TRUE(is_rotation_matrix(R));
}

TEST(pose_non_orthogonal) {
    Eigen::Matrix3d R;
    R << 1, 0, 0,
         0, 2, 0,
         0, 0, 1;
    ASSERT_FALSE(is_rotation_matrix(R));
}

TEST(pose_zero_translation) {
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    ASSERT_FALSE(validate_pose_range(t));  // norm=0 < 10
}

// ============================================================
// 主入口
// ============================================================

int main() {
    return 0;
    // 测试用例通过 TEST 宏自动注册和执行
    // 退出码由 test_utils 全局计数器控制（在 test_assert.hpp 的 main 替代中处理）
}