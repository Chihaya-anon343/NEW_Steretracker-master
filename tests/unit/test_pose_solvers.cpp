// ============================================================================
// test_pose_solvers.cpp — PnP 求解器单元测试
//
// 覆盖:
//   - InitialPnPSolver : RANSAC PnP + ITERATIVE 精化, 位姿有效性校验
//   - MonoPnPSolver    : 单目 EPnP + ITERATIVE 精化, 位姿有效性校验
//   - GPnPSolver       : 双目交叉射线 LM 优化, 合成双目对位姿恢复
//
// 策略: 合成 2D/3D 对应点, 用已知位姿投影生成真值.
// ============================================================================
#include "../framework/TestAssert.hpp"

#include "pose/InitialPnPSolver.hpp"
#include "pose/MonoPnPSolver.hpp"
#include "pose/GPnPSolver.hpp"
#include "common/Types.hpp"
#include "common/GeometryUtils.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace gpnp;

// ----------------------------------------------------------------------------
// 合成数据辅助
// ----------------------------------------------------------------------------

/// 标准 K: fx=fy=1000, cx=cy=512
Eigen::Matrix3d makeK() {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 1000.0;
    K(1, 1) = 1000.0;
    K(0, 2) = 512.0;
    K(1, 2) = 512.0;
    return K;
}

/// Z=0 平面模板 3D 点 (mm), 共 8 点
std::vector<Eigen::Vector3d> makeTemplatePts3D() {
    return {
        Eigen::Vector3d(-80, -50, 0),  Eigen::Vector3d(80, -50, 0),
        Eigen::Vector3d(80, 50, 0),    Eigen::Vector3d(-80, 50, 0),
        Eigen::Vector3d(-50, -25, 0),  Eigen::Vector3d(50, -25, 0),
        Eigen::Vector3d(50, 25, 0),    Eigen::Vector3d(-50, 25, 0),
    };
}

/// Z=0 平面正方形 4 角 (mm), 边长 87mm — 对齐 tiny_target.square_size_m_class0
std::vector<Eigen::Vector3d> makeSquarePts3D() {
    const double h = 43.5;
    return {
        Eigen::Vector3d(-h, -h, 0),  Eigen::Vector3d(h, -h, 0),
        Eigen::Vector3d(h,  h, 0),   Eigen::Vector3d(-h, h, 0),
    };
}

/// 相机绕 Y 轴旋转 ry 弧度, 平移 tz 的位姿
void makePose(double ry, double tz_mm, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    Eigen::Matrix3d Ry;
    Ry = Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY());
    R = Ry;
    t = Eigen::Vector3d(0.0, 0.0, tz_mm);
}

/// 3D 点经 [R|t] 变换后投影到 K
cv::Point2f project(const Eigen::Vector3d& P, const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& t, const Eigen::Matrix3d& K) {
    const Eigen::Vector3d Pc = R * P + t;
    const double z = Pc.z();
    Eigen::Vector3d uv = K * (Pc / z);
    return cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
}

/// 在像素上叠加高斯噪声 (σ=0.5px)
void addPixelNoise(std::vector<cv::Point2f>& pts, double sigma = 0.5) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, sigma);
    for (auto& p : pts) {
        p.x += static_cast<float>(dist(rng));
        p.y += static_cast<float>(dist(rng));
    }
}

// ----------------------------------------------------------------------------
// InitialPnPSolver 测试
// ----------------------------------------------------------------------------

/// 构造 MatchResult: 每对 (queryIdx=i → trainIdx=i) 一一对应
MatchResult makeMatchResult(const std::vector<Eigen::Vector3d>& pts_3d,
                            const std::vector<cv::Point2f>& pts_left) {
    MatchResult m;
    m.pts_left_match = pts_left;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        m.good_matches.emplace_back(static_cast<int>(i), static_cast<int>(i), 0.f);
    }
    m.num_matches = static_cast<int>(pts_3d.size());
    return m;
}

void test_initial_pnp_recovers_pose() {
    const auto K = makeK();
    const auto pts_3d = makeTemplatePts3D();

    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    makePose(0.15, 1500.0, R_gt, t_gt); // ~1.5m 前上方

    std::vector<cv::Point2f> pts_left;
    for (const auto& p : pts_3d) pts_left.push_back(project(p, R_gt, t_gt, K));
    addPixelNoise(pts_left, 0.3);

    InitialPnPSolver solver(4);
    const auto match = makeMatchResult(pts_3d, pts_left);
    const PoseEstimate pose = solver.solve(match, pts_3d, K);

    TEST_ASSERT_MSG(pose.success, "InitialPnP 应成功求解合成位姿");

    // 平移误差: ||t - t_gt|| / ||t_gt|| < 5%
    const double t_err = (pose.t - t_gt).norm() / t_gt.norm();
    TEST_ASSERT_MSG(t_err < 0.05,
                    "InitialPnP 平移误差过大: " + std::to_string(t_err));

    // 旋转误差: 角度差 < 5°
    const Eigen::Matrix3d R_err = R_gt.transpose() * pose.R;
    const Eigen::AngleAxisd aa(R_err);
    TEST_ASSERT_MSG(aa.angle() < 0.09 /* ~5° */,
                    "InitialPnP 旋转误差过大: " + std::to_string(aa.angle()));
}

void test_initial_pnp_rejects_bad_pose() {
    const auto K = makeK();
    const auto pts_3d = makeTemplatePts3D();

    // 退化输入: 点数不足
    std::vector<cv::Point2f> pts(3, cv::Point2f(500, 500));
    InitialPnPSolver solver(4);
    const auto match = makeMatchResult(pts_3d, pts);
    const PoseEstimate pose = solver.solve(match, pts_3d, K);
    TEST_ASSERT(!pose.success);
}

void test_initial_pnp_validity_checks() {
    const auto K = makeK();
    const auto pts_3d = makeTemplatePts3D();

    // 深度超范围 (|t| > 20000mm) → 非法, 应失败
    // 注意: 不能用相机在目标后方 (z<0) 的场景测试 —— PnP 存在手性歧义,
    // 后方投影与前方镜像投影几乎一致, 求解器会找到合法前向解 (t.z>0)。
    std::vector<cv::Point2f> pts;
    for (const auto& p : pts_3d) pts.push_back(project(p, Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 25000), K));

    InitialPnPSolver solver(4);
    const auto match = makeMatchResult(pts_3d, pts);
    const PoseEstimate pose = solver.solve(match, pts_3d, K);
    TEST_ASSERT(!pose.success);
}

// ----------------------------------------------------------------------------
// MonoPnPSolver 测试
// ----------------------------------------------------------------------------

void test_mono_pnp_recovers_pose() {
    const auto K = makeK();
    const auto pts_3d = makeTemplatePts3D();

    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    makePose(-0.1, 800.0, R_gt, t_gt);

    std::vector<cv::Point2f> pts_2d;
    for (const auto& p : pts_3d) pts_2d.push_back(project(p, R_gt, t_gt, K));
    addPixelNoise(pts_2d, 0.3);

    MonoPnPSolver solver;
    const PoseEstimate pose = solver.solve(pts_2d, pts_3d, K);

    TEST_ASSERT_MSG(pose.success, "MonoPnP 应成功求解合成位姿");

    const double t_err = (pose.t - t_gt).norm() / t_gt.norm();
    TEST_ASSERT_MSG(t_err < 0.05, "MonoPnP 平移误差过大: " + std::to_string(t_err));

    const Eigen::Matrix3d R_err = R_gt.transpose() * pose.R;
    const Eigen::AngleAxisd aa(R_err);
    TEST_ASSERT_MSG(aa.angle() < 0.09, "MonoPnP 旋转误差过大: " + std::to_string(aa.angle()));
}

void test_mono_pnp_rejects_insufficient_points() {
    const auto K = makeK();
    const auto pts_3d = makeTemplatePts3D();

    std::vector<cv::Point2f> pts_2d = {
        cv::Point2f(500, 500), cv::Point2f(520, 500), cv::Point2f(520, 520)
    };
    MonoPnPSolver solver;
    const PoseEstimate pose = solver.solve(pts_2d, pts_3d, K);
    TEST_ASSERT(!pose.success);
}

// ----------------------------------------------------------------------------
// MonoPnPSolver 4点 seeded 棘轮冻结回归测试 (2026-09 修复)
//
// 旧行为: n==4 + seed = 单一 seed 初值 ITERATIVE + 8px 平坦校验。
// 4 点共面 IPPE 二义性下 ITERATIVE 恒收敛到 seed 盆地, 8px 阈值在
// 15~30px 小目标上相当于 30~50% 相对容差 → 陈旧 seed 解恒通过,
// 逼近目标时深度冻结 (实测日志 t.z 钉死 ~8060mm 8 帧不放)。
// 新行为: 相对容差 clamp(5%×2D跨度,1,8)px + SeedITER/冷ITER/IPPE 候选竞争。
// ----------------------------------------------------------------------------

/// 多帧逼近 + 逐帧 seed 链: 深度 8000→5300mm (每帧 -300, 尺度 ~5%/帧增长),
/// 每帧用上帧解作 seed, 模拟 normal 模式位姿链。逐帧深度误差须 < 6%。
void test_mono_pnp_4pt_stale_seed_tracks_approach() {
    const auto K = makeK();
    const auto pts_3d = makeSquarePts3D();

    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;

    PoseEstimate prev;
    for (int f = 0; f < 10; ++f) {
        const double tz = 8000.0 - 300.0 * f;
        makePose(0.1, tz, R_gt, t_gt);

        std::vector<cv::Point2f> pts_2d;
        for (const auto& p : pts_3d) pts_2d.push_back(project(p, R_gt, t_gt, K));
        addPixelNoise(pts_2d, 0.2);

        PoseEstimate pose;
        MonoPnPSolver solver;
        if (f == 0) {
            pose = solver.solve(pts_2d, pts_3d, K);
        } else {
            PoseSeed seed;
            seed.R = prev.R;
            seed.t = prev.t;
            pose = solver.solve(pts_2d, pts_3d, K, &seed, 0.3);
        }

        // 视觉跨度 ~11-16px 的小目标, 深度误差须逐帧受控 (无冻结/跳变)
        const double t_err = std::abs(pose.t.norm() - tz) / tz;
        TEST_ASSERT_MSG(pose.success && t_err < 0.06,
                        "frame " + std::to_string(f) +
                        " 深度误差过大: |t|=" + std::to_string(pose.t.norm()) +
                        " (期望 " + std::to_string(tz) +
                        ", err=" + std::to_string(t_err) + ")");
        prev = pose;
    }
}

/// 单帧冻结 seed: 真值 5000mm (span ~17px), seed 钉死在 8000mm (span ~11px)。
/// 旧代码: seed 解重投影 ~3px < 8px 恒通过 → 返回 ~8000mm (60% 深度误差);
/// 新代码: 相对容差 (~1px) 拦截 seed 解, 冷启动/IPPE 候选胜出。
void test_mono_pnp_4pt_stale_seed_loses_competition() {
    const auto K = makeK();
    const auto pts_3d = makeSquarePts3D();

    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    makePose(0.1, 5000.0, R_gt, t_gt);

    std::vector<cv::Point2f> pts_2d;
    for (const auto& p : pts_3d) pts_2d.push_back(project(p, R_gt, t_gt, K));
    addPixelNoise(pts_2d, 0.3);

    PoseSeed seed;
    seed.R = R_gt;
    seed.t = Eigen::Vector3d(0.0, 0.0, 8000.0);   // 深度钉死在 1.6× 真值

    MonoPnPSolver solver;
    const PoseEstimate pose = solver.solve(pts_2d, pts_3d, K, &seed, 0.3);

    const double t_err = (pose.t - t_gt).norm() / t_gt.norm();
    TEST_ASSERT_MSG(pose.success && t_err < 0.05,
                    "冻结 seed 未被候选竞争淘汰: |t|=" +
                    std::to_string(pose.t.norm()) +
                    " (期望 5000, err=" + std::to_string(t_err) + ")");
}

// ----------------------------------------------------------------------------
// GPnPSolver 测试
// ----------------------------------------------------------------------------

StereoCameraParams makeStereoParams() {
    StereoCameraParams p;
    p.K = makeK();
    p.K_inv = p.K.inverse();
    p.R_rl = Eigen::Matrix3d::Identity();
    // t_rl is the right-camera center expressed in the left-camera frame.
    p.t_rl = Eigen::Vector3d(120.0, 0, 0); // baseline 120mm
    p.focal_length = 1000.0;
    p.baseline = 120.0;
    return p;
}

void test_gpnp_recovers_pose_stereo() {
    const auto params = makeStereoParams();
    const auto K = makeK();
    const auto pts_3d = makeTemplatePts3D();

    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    makePose(0.1, 1200.0, R_gt, t_gt);

    // 左右图投影 → 构建 PipelineResult
    PipelineResult result;
    result.good_matches.clear();
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        const Eigen::Vector3d Pw = pts_3d[i];
        const cv::Point2f pL = project(Pw, R_gt, t_gt, K);
        // R_rl maps right-camera vectors to the left frame, while t_rl is
        // the right-camera center in the left frame. Therefore left -> right:
        // P_r = R_rl^T * (P_l - t_rl).
        const Eigen::Vector3d Pc_l = R_gt * Pw + t_gt;
        const Eigen::Vector3d Pc_r = params.R_rl.transpose() * (Pc_l - params.t_rl);
        cv::Point2f pR(512, 512);
        if (Pc_r.z() > 0) {
            const Eigen::Vector3d uv = K * (Pc_r / Pc_r.z());
            pR = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
        }

        result.pts_left_match.push_back(pL);
        result.disparity.push_back(static_cast<float>(pL.x - pR.x));
        result.pts_left_good.push_back(pL);
        result.pts_right_good.push_back(pR);
        result.idx_from_filtered.push_back(static_cast<int>(i));
        result.good_matches.emplace_back(static_cast<int>(i), static_cast<int>(i), 0.f);
    }
    result.n_kp_left = static_cast<int>(pts_3d.size());
    result.n_matched = static_cast<int>(pts_3d.size());
    result.success = true;

    // 真值附近 warm-start (允许一定偏差)
    Eigen::Matrix3d R_init;
    Eigen::Vector3d t_init;
    makePose(0.1 + 0.02, 1200.0 + 50.0, R_init, t_init);

    GPnPSolver solver(params, 4);
    double timing = 0.0;
    const PoseEstimate pose = solver.solve(result, pts_3d, &R_init, &t_init, timing);

    TEST_ASSERT_MSG(pose.success, "GPnP 应成功求解合成双目位姿");

    const double t_err = (pose.t - t_gt).norm() / t_gt.norm();
    // 共面目标存在平移-旋转歧义 (Z=0 平面上 X 平移与 Y 轴旋转耦合),
    // 阈值放宽到 15% 以容纳这种固有退化。
    TEST_ASSERT_MSG(t_err < 0.15, "GPnP 平移误差过大: " + std::to_string(t_err));

    const Eigen::Matrix3d R_err = R_gt.transpose() * pose.R;
    const Eigen::AngleAxisd aa(R_err);
    TEST_ASSERT_MSG(aa.angle() < 0.15, "GPnP 旋转误差过大: " + std::to_string(aa.angle()));
}

void test_gpnp_rejects_insufficient_points() {
    const auto params = makeStereoParams();
    const auto pts_3d = makeTemplatePts3D();

    PipelineResult result;
    for (size_t i = 0; i < 2; ++i) { // 仅 2 点 < min_pts=4
        result.pts_left_match.emplace_back(500.f, 500.f);
        result.pts_left_good.emplace_back(500.f, 500.f);
        result.pts_right_good.emplace_back(480.f, 500.f);
        result.disparity.push_back(20.f);
        result.idx_from_filtered.push_back(static_cast<int>(i));
        result.good_matches.emplace_back(static_cast<int>(i), static_cast<int>(i), 0.f);
    }

    GPnPSolver solver(params, 4);
    double timing = 0.0;
    const PoseEstimate pose = solver.solve(result, pts_3d, nullptr, nullptr, timing);
    TEST_ASSERT(!pose.success);
}

REGISTER_TEST(test_initial_pnp_recovers_pose);
REGISTER_TEST(test_initial_pnp_rejects_bad_pose);
REGISTER_TEST(test_initial_pnp_validity_checks);

REGISTER_TEST(test_mono_pnp_recovers_pose);
REGISTER_TEST(test_mono_pnp_rejects_insufficient_points);
REGISTER_TEST(test_mono_pnp_4pt_stale_seed_tracks_approach);
REGISTER_TEST(test_mono_pnp_4pt_stale_seed_loses_competition);

REGISTER_TEST(test_gpnp_recovers_pose_stereo);
REGISTER_TEST(test_gpnp_rejects_insufficient_points);

} // namespace

// ============================================================================
// 测试入口
// ============================================================================

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}
