#include "pose/GPnPSolver.hpp"
#include "common/GeometryUtils.hpp"

#include <unsupported/Eigen/NonLinearOptimization>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

namespace gpnp {

// ============================================================================
// GPNP 代价函子（0626：预拆分光线、直接四元数→R、交错输出）
// ============================================================================

struct GPNPCostFunctor {
    typedef double Scalar;
    typedef Eigen::VectorXd InputType;
    typedef Eigen::VectorXd ValueType;
    typedef Eigen::MatrixXd JacobianType;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

    const Eigen::MatrixX3d& pts_3d_T;   // N×3
    const Eigen::MatrixX3d& origins_L;  // N×3
    const Eigen::MatrixX3d& dirs_L;     // N×3
    const Eigen::MatrixX3d& origins_R;  // N×3
    const Eigen::MatrixX3d& dirs_R;     // N×3
    const int n_pts;
    const Eigen::Vector3d t_warm;       // 热启动平移（先验锚点）

    GPNPCostFunctor(const Eigen::MatrixX3d& _pts_3d_T,
                    const Eigen::MatrixX3d& _origins_L,
                    const Eigen::MatrixX3d& _dirs_L,
                    const Eigen::MatrixX3d& _origins_R,
                    const Eigen::MatrixX3d& _dirs_R,
                    int _n_pts,
                    const Eigen::Vector3d& _t_warm = Eigen::Vector3d::Zero())
        : pts_3d_T(_pts_3d_T), origins_L(_origins_L), dirs_L(_dirs_L),
          origins_R(_origins_R), dirs_R(_dirs_R), n_pts(_n_pts),
          t_warm(_t_warm)
    {}

    int inputs() const { return 7; }
    int values() const { return 6 * n_pts + 3; }  // +3 用于平移先验 (tx,ty,tz)

    /// 中心差分数值雅可比。
    int df(const InputType& x, JacobianType& fjac) const {
        const int n = inputs(), m = values();
        fjac.resize(m, n);
        InputType xp = x, xm = x;
        ValueType fp(m), fm(m);
        const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
        for (int j = 0; j < n; ++j) {
            double h = eps * std::max(1.0, std::abs(x(j)));
            xp(j) = x(j) + h; xm(j) = x(j) - h;
            (*this)(xp, fp); (*this)(xm, fm);
            fjac.col(j) = (fp - fm) / (2.0 * h);
            xp(j) = x(j); xm(j) = x(j);
        }
        return 0;
    }

    /// 计算残差：直接四元数→R、向量化叉积、交错输出。
    int operator()(const InputType& x, ValueType& fvec) const {
        // 四元数→R（归一化无关：除以 q² 以获得平滑雅可比）
        // 参数 = [w, x, y, z, tx, ty, tz]
        double qw = x(0), qx = x(1), qy = x(2), qz = x(3);
        double qn2 = qw*qw + qx*qx + qy*qy + qz*qz;
        if (qn2 < 1e-20) { fvec.setZero(); return 0; }
        double inv_qn2 = 1.0 / qn2;

        // R = 1/q² * [q²-2(y²+z²)  2(xy-wz)     2(xz+wy) ]
        //            [2(xy+wz)       q²-2(x²+z²)  2(yz-wx) ]
        //            [2(xz-wy)       2(yz+wx)      q²-2(x²+y²)]
        double R00 = (qn2 - 2.0*(qy*qy + qz*qz)) * inv_qn2;
        double R01 = 2.0*(qx*qy - qw*qz) * inv_qn2;
        double R02 = 2.0*(qx*qz + qw*qy) * inv_qn2;
        double R10 = 2.0*(qx*qy + qw*qz) * inv_qn2;
        double R11 = (qn2 - 2.0*(qx*qx + qz*qz)) * inv_qn2;
        double R12 = 2.0*(qy*qz - qw*qx) * inv_qn2;
        double R20 = 2.0*(qx*qz - qw*qy) * inv_qn2;
        double R21 = 2.0*(qy*qz + qw*qx) * inv_qn2;
        double R22 = (qn2 - 2.0*(qx*qx + qy*qy)) * inv_qn2;

        double tx = x(4), ty = x(5), tz = x(6);

        // 逐点计算残差（交错：cross_l[3] + cross_r[3]）
        for (int i = 0; i < n_pts; ++i) {
            double px = pts_3d_T(i,0), py = pts_3d_T(i,1), pz = pts_3d_T(i,2);
            // P_cam = R * pt_3d + t（标准行访问矩阵-向量乘法）
            double Px = px*R00 + py*R01 + pz*R02 + tx;
            double Py = px*R10 + py*R11 + pz*R12 + ty;
            double Pz = px*R20 + py*R21 + pz*R22 + tz;

            // 左叉积
            double dLx = Px - origins_L(i,0), dLy = Py - origins_L(i,1), dLz = Pz - origins_L(i,2);
            fvec(6*i+0) = dLy*dirs_L(i,2) - dLz*dirs_L(i,1);
            fvec(6*i+1) = dLz*dirs_L(i,0) - dLx*dirs_L(i,2);
            fvec(6*i+2) = dLx*dirs_L(i,1) - dLy*dirs_L(i,0);

            // 右叉积
            double dRx = Px - origins_R(i,0), dRy = Py - origins_R(i,1), dRz = Pz - origins_R(i,2);
            fvec(6*i+3) = dRy*dirs_R(i,2) - dRz*dirs_R(i,1);
            fvec(6*i+4) = dRz*dirs_R(i,0) - dRx*dirs_R(i,2);
            fvec(6*i+5) = dRx*dirs_R(i,1) - dRy*dirs_R(i,0);
        }

        // 平移先验：朝热启动值的二次回复力。
        // 打破共面 PnP 歧义 —— 没有这个，(R, t) 和 (R', t')
        // 在共面模板点下 t 差异很大仍可产生相同的叉积残差。
        //   w_tz 更重：深度受共面歧义影响最大
        //   w_t² 添加到 J^T*J 对角线 → 在 t 方向抑制第一步 LM 步长
        const double w_t = 0.5;
        const double w_tz = 2.0;
        fvec(6 * n_pts + 0) = w_t  * (tx - t_warm(0));
        fvec(6 * n_pts + 1) = w_t  * (ty - t_warm(1));
        fvec(6 * n_pts + 2) = w_tz * (tz - t_warm(2));

        return 0;
    }
};

// ============================================================================
// GPnPSolver 实现
// ============================================================================

GPnPSolver::GPnPSolver(const StereoCameraParams& params, int gpnp_min_pts)
    : params_(params), gpnp_min_pts_(gpnp_min_pts) {}

GPnPSolver::~GPnPSolver() = default;

// ============================================================================
// 求解
// ============================================================================

PoseEstimate GPnPSolver::solve(PipelineResult& result,
                                 const std::vector<Eigen::Vector3d>& pts_3d_T,
                                 const Eigen::Matrix3d* R_init,
                                 const Eigen::Vector3d* t_init,
                                 double& timing_ms_out)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    monitor_ = GPNPMonitor{};
    monitor_.gpnp_min_pts = gpnp_min_pts_;
    PoseEstimate pose_result;
    pose_result.success = false;

    auto fail = [&](const std::string& reason) {
        monitor_.failure_reason = reason;
        auto t_end = std::chrono::high_resolution_clock::now();
        timing_ms_out = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        monitor_.timing_ms = timing_ms_out;
        result.timing["gpnp"] = timing_ms_out;
        printMonitor();
        return pose_result;
    };

    // ---- 步骤1: queryIdx → trainIdx 映射 ----
    const auto& good_matches = result.good_matches;
    monitor_.n_good_matches = static_cast<int>(good_matches.size());
    if (good_matches.empty()) return fail("No template matches");

    // ---- 步骤2: 立体匹配点 ----
    const auto& pts_left_good = result.pts_left_good;
    const auto& pts_right_good = result.pts_right_good;
    const auto& idx_from_filtered = result.idx_from_filtered;
    monitor_.n_idx_from_filtered = static_cast<int>(idx_from_filtered.size());
    monitor_.n_stereo_matched = std::min(static_cast<int>(pts_left_good.size()),
                                         static_cast<int>(pts_right_good.size()));
    if (idx_from_filtered.empty()) return fail("No stereo-matched points");

    int n_matched = std::min({static_cast<int>(idx_from_filtered.size()),
                              static_cast<int>(pts_left_good.size()),
                              static_cast<int>(pts_right_good.size())});
    monitor_.n_matched = n_matched;

    // ---- 步骤3: 基于查找表的交集（0626 向量化）----
    // 构建查找表：queryIdx → trainIdx
    int max_qidx = 0;
    for (int i = 0; i < n_matched && i < static_cast<int>(idx_from_filtered.size()); ++i)
        max_qidx = std::max(max_qidx, idx_from_filtered[i]);
    for (const auto& m : good_matches) max_qidx = std::max(max_qidx, m.queryIdx);
    max_qidx = std::max(max_qidx, 0);

    std::vector<int> train_lut(max_qidx + 1, -1);
    for (const auto& m : good_matches) train_lut[m.queryIdx] = m.trainIdx;

    // 应用查找表：train_idx = train_lut[kp_left_idx]
    std::vector<int> keep;
    keep.reserve(n_matched);
    int n_missing_query = 0, n_missing_train = 0;
    const int n_pts3d = static_cast<int>(pts_3d_T.size());

    for (int i = 0; i < n_matched; ++i) {
        int kp_idx = idx_from_filtered[i];
        int tr_idx = (kp_idx <= max_qidx) ? train_lut[kp_idx] : -1;
        if (tr_idx < 0) { ++n_missing_query; continue; }
        if (tr_idx >= n_pts3d) { ++n_missing_train; continue; }
        keep.push_back(i);
    }

    int n_pts = static_cast<int>(keep.size());
    monitor_.n_intersection = n_pts;
    monitor_.n_missing_query = n_missing_query;
    monitor_.n_missing_train = n_missing_train;
    monitor_.n_pts = n_pts;

    if (n_pts < gpnp_min_pts_)
        return fail("Intersection points insufficient (" + std::to_string(n_pts) + " < " + std::to_string(gpnp_min_pts_) + ")");

    // ---- 步骤4: 构建数组并批量生成光线（0626 向量化）----
    // 为保留的索引收集3D点和像素坐标
    Eigen::MatrixX3d pts_3d_arr(n_pts, 3);
    Eigen::MatrixX2d left_pix(n_pts, 2), right_pix(n_pts, 2);

    for (int k = 0; k < n_pts; ++k) {
        int i = keep[k];
        int tr_idx = train_lut[idx_from_filtered[i]];
        const auto& p3d = pts_3d_T[tr_idx];
        pts_3d_arr(k, 0) = p3d.x(); pts_3d_arr(k, 1) = p3d.y(); pts_3d_arr(k, 2) = p3d.z();
        left_pix(k, 0) = pts_left_good[i].x;  left_pix(k, 1) = pts_left_good[i].y;
        right_pix(k, 0) = pts_right_good[i].x; right_pix(k, 1) = pts_right_good[i].y;
    }

    // 批量生成左光线
    Eigen::MatrixXd homo_L(n_pts, 3);
    homo_L.leftCols<2>() = left_pix; homo_L.col(2).setOnes();
    Eigen::MatrixX3d dirs_L = homo_L * params_.K_inv.transpose();
    for (int i = 0; i < n_pts; ++i) {
        double n = dirs_L.row(i).norm(); if (n > 1e-12) dirs_L.row(i) /= n;
    }
    Eigen::MatrixX3d origins_L = Eigen::MatrixX3d::Zero(n_pts, 3);

    // 批量生成右光线
    Eigen::MatrixXd homo_R(n_pts, 3);
    homo_R.leftCols<2>() = right_pix; homo_R.col(2).setOnes();
    Eigen::MatrixX3d dirs_R_c2 = homo_R * params_.K_inv.transpose();
    for (int i = 0; i < n_pts; ++i) {
        double n = dirs_R_c2.row(i).norm(); if (n > 1e-12) dirs_R_c2.row(i) /= n;
    }
    Eigen::MatrixX3d dirs_R = dirs_R_c2 * params_.R_rl.transpose();
    for (int i = 0; i < n_pts; ++i) {
        double n = dirs_R.row(i).norm(); if (n > 1e-12) dirs_R.row(i) /= n;
    }
    Eigen::MatrixX3d origins_R(n_pts, 3);
    for (int i = 0; i < n_pts; ++i) origins_R.row(i) = params_.t_rl.transpose();

    monitor_.n_rays = n_pts * 2;

    // ---- 步骤5: 深度猜测 ----
    double median_disp = 0.0;
    if (!result.disparity.empty()) {
        std::vector<double> abs_disp;
        for (double d : result.disparity) abs_disp.push_back(std::abs(d));
        median_disp = computeMedian(std::move(abs_disp));
    }
    double depth_guess = 500.0;
    if (median_disp > 1.0) {
        depth_guess = params_.focal_length * params_.baseline / median_disp;
        depth_guess = std::clamp(depth_guess, 50.0, 5000.0);
    }
    monitor_.depth_guess = depth_guess;

    // ---- 步骤6: 初始参数向量（0626：[w,x,y,z,tx,ty,tz]）----
    Eigen::VectorXd x0(7);
    if (R_init != nullptr && t_init != nullptr) {
        // 代价函数使用标准 R * p + t 约定。
        // R_init 已按 R_physical 约定；直接使用。
        Eigen::Quaterniond q_eigen(*R_init);
        x0 << q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z(),
              (*t_init)(0), (*t_init)(1), (*t_init)(2);
    } else {
        x0 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, depth_guess;
    }

    // ---- 步骤7: LM 优化 ----
    Eigen::Vector3d t_warm = (t_init != nullptr) ? *t_init : Eigen::Vector3d(0, 0, depth_guess);
    GPNPCostFunctor functor(pts_3d_arr, origins_L, dirs_L, origins_R, dirs_R, n_pts, t_warm);

    Eigen::VectorXd fvec_init(functor.values());
    functor(x0, fvec_init);
    monitor_.opt_initial_cost = fvec_init.squaredNorm();

    Eigen::LevenbergMarquardt<GPNPCostFunctor, double> lm(functor);
    lm.parameters.ftol = 1e-8;
    lm.parameters.xtol = 1e-8;
    lm.parameters.gtol = 1e-8;
    lm.parameters.maxfev = 200;

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x0);

    // 提取结果
    double qw_opt = x0(0), qx_opt = x0(1), qy_opt = x0(2), qz_opt = x0(3);
    Eigen::Vector3d t_opt(x0(4), x0(5), x0(6));
    Eigen::Quaterniond q_opt_eigen(qw_opt, qx_opt, qy_opt, qz_opt);
    q_opt_eigen.normalize();
    if (q_opt_eigen.w() < 0) q_opt_eigen.coeffs() = -q_opt_eigen.coeffs();

    monitor_.q_opt = {q_opt_eigen.x(), q_opt_eigen.y(), q_opt_eigen.z(), q_opt_eigen.w()};
    monitor_.t_opt = {t_opt(0), t_opt(1), t_opt(2)};

    Eigen::VectorXd fvec_final(functor.values());
    functor(x0, fvec_final);
    monitor_.opt_final_cost = fvec_final.squaredNorm();
    monitor_.opt_cost_reduction = monitor_.opt_initial_cost - monitor_.opt_final_cost;
    monitor_.opt_status = static_cast<int>(status);
    monitor_.opt_nfev = lm.nfev;

    bool opt_success = (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters);
    monitor_.opt_success = opt_success;
    monitor_.opt_message = opt_success ? "Converged" : "Failed";

    // 代价函数使用标准 R * p + t 约定。
    // 优化后的四元数直接编码 R_physical。
    pose_result.R = q_opt_eigen.toRotationMatrix();
    pose_result.t = t_opt;
    pose_result.num_points = n_pts;
    pose_result.success = opt_success;

    // 诊断：将 GPnP 结果与 InitialPnP 热启动进行比较
    if (R_init != nullptr && t_init != nullptr && opt_success) {
        double t_dot = pose_result.t.dot(*t_init);
        double R_diff = (pose_result.R - *R_init).norm();
        std::cout << "  [GPnP Diag] t_warm=[" << (*t_init)(0) << "," << (*t_init)(1) << "," << (*t_init)(2) << "]"
                  << "  t_opt=[" << pose_result.t(0) << "," << pose_result.t(1) << "," << pose_result.t(2) << "]"
                  << "  t·t_warm=" << t_dot << "  |ΔR|=" << R_diff << std::endl;
        if (t_dot < 0.0) {
            std::cerr << "  [GPnP WARNING] Translation sign flipped! t·t_warm=" << t_dot << std::endl;
        }

        // PnP 热启动结果的健全性检查（非深度猜测）。
        // 对于共面模板，GPnP 的 R 可能被 r₃ 歧义污染
        // （不受约束的第三列），而 t 由立体正确细化。
        bool from_pnp = (*R_init - Eigen::Matrix3d::Identity()).norm() > 0.01;
        if (from_pnp) {
            if (t_dot < 0.0) {
                // 平移符号翻转 → 完全回退到 InitialPnP
                std::cerr << "  [GPnP FALLBACK] Translation sign flipped (t·t_warm=" << t_dot
                          << "), using InitialPnP pose" << std::endl;
                pose_result.R = *R_init;
                pose_result.t = *t_init;
                pose_result.success = true;
            } else if (R_diff > 1.5) {
                // 大 |ΔR| 且 t 符号正确 → 共面 r₃ 歧义。
                // 保留 InitialPnP R（几何一致），使用 GPnP t（立体细化）。
                std::cout << "  [GPnP HYBRID] |ΔR|=" << R_diff
                          << " > 1.5 (coplanar ambiguity), using InitialPnP R + GPnP t" << std::endl;
                pose_result.R = *R_init;
                // pose_result.t = t_opt（已从 GPnP 设置）
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    timing_ms_out = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    monitor_.timing_ms = timing_ms_out;
    result.timing["gpnp"] = timing_ms_out;

    if (!opt_success) {
        if (monitor_.failure_reason.empty())
            monitor_.failure_reason = "Optimizer failed";
        printMonitor();
    }
    return pose_result;
}

// ============================================================================
// 监控输出
// ============================================================================

void GPnPSolver::printMonitor() const {
    const std::string sep(70, '-');
    std::cout << "\n" << sep << "\n";
    std::cout << "  ⚠ GPNP Failed | Reason: "
              << (monitor_.failure_reason.empty() ? "Unknown" : monitor_.failure_reason) << "\n";
    std::cout << sep << "\n\n  [Data Flow Statistics]\n";
    std::cout << "    good_matches                      : " << monitor_.n_good_matches << "\n";
    std::cout << "    query_to_train (mapping)           : " << monitor_.n_query_to_train << "\n";
    std::cout << "    idx_from_filtered                  : " << monitor_.n_idx_from_filtered << "\n";
    std::cout << "    n_stereo_matched                   : " << monitor_.n_stereo_matched << "\n";
    std::cout << "    n_matched                          : " << monitor_.n_matched << "\n";
    std::cout << "    n_intersection                     : " << monitor_.n_intersection << "\n";
    std::cout << "      ├─ missing queryIdx              : " << monitor_.n_missing_query << "\n";
    std::cout << "      └─ missing trainIdx              : " << monitor_.n_missing_train << "\n";
    std::cout << "    n_pts (points for PnP)             : " << monitor_.n_pts << "\n";
    std::cout << "    gpnp_min_pts                       : " << monitor_.gpnp_min_pts << "\n";
    std::cout << "\n  [Optimizer Status]\n";
    std::cout << "    success         : " << (monitor_.opt_success ? "true" : "false") << "\n";
    std::cout << "    status          : " << monitor_.opt_status << "\n";
    std::cout << "    nfev            : " << monitor_.opt_nfev << "\n";
    std::cout << "    initial cost    : " << monitor_.opt_initial_cost << "\n";
    std::cout << "    final cost      : " << monitor_.opt_final_cost << "\n";
    if (!monitor_.q_opt.empty() && !monitor_.t_opt.empty()) {
        std::cout << "\n  [Optimization Result]\n";
        std::cout << "    q  : [" << monitor_.q_opt[0] << ", " << monitor_.q_opt[1]
                  << ", " << monitor_.q_opt[2] << ", " << monitor_.q_opt[3] << "]\n";
        std::cout << "    t  : [" << monitor_.t_opt[0] << ", " << monitor_.t_opt[1]
                  << ", " << monitor_.t_opt[2] << "]\n";
    }
    if (!monitor_.exception.empty())
        std::cout << "\n  [Exception] " << monitor_.exception << "\n";
    std::cout << "\n  GPNP Time: " << monitor_.timing_ms << " ms\n";
    std::cout << sep << "\n" << std::endl;
}

} // namespace gpnp