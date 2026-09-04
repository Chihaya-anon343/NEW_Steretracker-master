/**
 * @file MonoPnPSolver.cpp
 * @brief 单目 PnP 求解器实现：多候选（EPnP RANSAC + 共面 IPPE）+ 重投影误差择优。
 */

#include "pose/MonoPnPSolver.hpp"
#include "common/LogConfig.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gpnp {

namespace {

/// 重投影误差判优阈值 (px)，与 RANSAC reprojectionError 一致
constexpr double kMaxReprojErrorPx = 8.0;
constexpr double kMinDepthMm = 10.0;
constexpr double kMaxDepthMm = 100000.0;

/// 4 点 seeded 路径的小目标相对重投影容差: 8px 平坦阈值在 15~30px 目标上
/// 相当于 30~50% 的相对容差, 陈旧 seed 解恒通过 → 深度棘轮冻结 (2026-09 修复)。
/// 容差 = 5% × 2D span, 夹在 [1, 8] px: 小目标收紧, 大 ROI 与旧阈值一致。
constexpr double kSmallTargetReprojRatio = 0.05;
constexpr double kMinReprojTolPx = 1.0;

// 平均重投影误差 (px)；输入异常返回 max
double meanReprojError(const std::vector<cv::Point3f>& obj,
                       const std::vector<cv::Point2f>& img,
                       const cv::Mat& rvec, const cv::Mat& tvec,
                       const cv::Mat& K_cv) {
    if (rvec.empty() || tvec.empty() || obj.empty() || obj.size() != img.size())
        return std::numeric_limits<double>::max();
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, K_cv, cv::Mat(), proj);
    double sum = 0.0;
    for (size_t i = 0; i < proj.size(); ++i)
        sum += std::hypot(proj[i].x - img[i].x, proj[i].y - img[i].y);
    return sum / static_cast<double>(proj.size());
}

// 解有效性: 有限 + t.z>0 + 深度范围 + 重投影误差达标
// (NaN 与任何比较均为 false，可一并排除非有限解)
bool isValidSolution(const cv::Mat& rvec, const cv::Mat& tvec, double reproj,
                     double max_reproj_px) {
    if (rvec.empty() || tvec.empty()) return false;
    if (!(reproj < max_reproj_px)) return false;
    if (!(tvec.at<double>(2) > 0.0)) return false;
    double tn = cv::norm(tvec);
    return tn > kMinDepthMm && tn < kMaxDepthMm;
}

bool isValidSolution(const cv::Mat& rvec, const cv::Mat& tvec, double reproj) {
    return isValidSolution(rvec, tvec, reproj, kMaxReprojErrorPx);
}

// 单个 PnP 候选解
struct PnpCandidate {
    cv::Mat rvec, tvec;
    double reproj;
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};  // seed 平票择优用
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
};

// 候选解与 seed 的几何距离: 旋转角 + 归一化平移差（归一化使深度无关可比）
double seedDistance(const PnpCandidate& c, const PoseSeed& seed) {
    Eigen::Matrix3d dR = c.R * seed.R.transpose();
    double ang = Eigen::AngleAxisd(dR).angle();
    double t_ref = std::max(seed.t.norm(), 1e-9);
    return ang + (c.t - seed.t).norm() / t_ref;
}

// 择优: ε-平票集内偏 seed, 否则纯重投影最小 (4点 seeded 与 >4 点路径共用)。
// 平面 IPPE 二义性会产生两个重投影都低的解，逐帧纯 reproj 择优会随机跳分支；
// 坏解必须先通过 reproj ≤ min+ε 才可能因"靠近上帧"胜出 ——
// 时序连续性只在"两个解同样好"时起作用。candidates 非空为前提。
const PnpCandidate* selectBestCandidate(const std::vector<PnpCandidate>& candidates,
                                        const PoseSeed* seed, double tie_epsilon_px) {
    double reproj_min = candidates[0].reproj;
    for (const auto& c : candidates)
        reproj_min = std::min(reproj_min, c.reproj);
    const double eps = (seed && tie_epsilon_px > 0.0) ? tie_epsilon_px : 0.0;
    const PnpCandidate* best = nullptr;
    for (const auto& c : candidates) {
        if (c.reproj > reproj_min + eps) continue;   // 不在平票集
        if (!best) { best = &c; continue; }
        if (seed) {
            if (seedDistance(c, *seed) < seedDistance(*best, *seed)) best = &c;
        } else if (c.reproj < best->reproj) {
            best = &c;
        }
    }
    return best;
}

cv::Mat seedRvec(const PoseSeed& seed) {
    cv::Mat R_cv = (cv::Mat_<double>(3, 3)
        << seed.R(0, 0), seed.R(0, 1), seed.R(0, 2),
           seed.R(1, 0), seed.R(1, 1), seed.R(1, 2),
           seed.R(2, 0), seed.R(2, 1), seed.R(2, 2));
    cv::Mat rv;
    cv::Rodrigues(R_cv, rv);
    return rv;
}

cv::Mat seedTvec(const PoseSeed& seed) {
    return (cv::Mat_<double>(3, 1)
        << seed.t(0), seed.t(1), seed.t(2));
}

} // namespace

PoseEstimate MonoPnPSolver::solve(const std::vector<cv::Point2f>& pts_2d,
                                   const std::vector<Eigen::Vector3d>& pts_3d,
                                   const Eigen::Matrix3d& K) {
    return solve(pts_2d, pts_3d, K, nullptr, 0.0);
}

PoseEstimate MonoPnPSolver::solve(const std::vector<cv::Point2f>& pts_2d,
                                   const std::vector<Eigen::Vector3d>& pts_3d,
                                   const Eigen::Matrix3d& K,
                                   const PoseSeed* seed,
                                   double tie_epsilon_px) {
    PoseEstimate pose;

    const int n = static_cast<int>(pts_2d.size());
    if (n < 4 || pts_3d.size() != pts_2d.size()) {
        std::cout << "[MonoPnP] 点数不足: pts_2d=" << n
                  << ", pts_3d=" << pts_3d.size() << " (需要 ≥4)" << std::endl;
        return pose;
    }

    // --- 1. 将Eigen::Vector3d 3D 点转换为 cv::Point3f ---
    std::vector<cv::Point3f> object_points;
    object_points.reserve(n);
    for (const auto& p : pts_3d) {
        object_points.emplace_back(static_cast<float>(p.x()),
                                   static_cast<float>(p.y()),
                                   static_cast<float>(p.z()));
    }

    // 构造相机内参矩阵
    cv::Mat K_cv = (cv::Mat_<double>(3, 3)
        << K(0, 0), K(0, 1), K(0, 2),
           K(1, 0), K(1, 1), K(1, 2),
           K(2, 0), K(2, 1), K(2, 2));

    // --- 诊断: 输入概况 ---
    if (g_verbose_console) {
        float minx = std::numeric_limits<float>::max();
        float maxx = std::numeric_limits<float>::lowest();
        float miny = std::numeric_limits<float>::max();
        float maxy = std::numeric_limits<float>::lowest();
        for (const auto& p : pts_2d) {
            minx = std::min(minx, p.x);  maxx = std::max(maxx, p.x);
            miny = std::min(miny, p.y);  maxy = std::max(maxy, p.y);
        }
        double min3x = std::numeric_limits<double>::max();
        double max3x = std::numeric_limits<double>::lowest();
        double min3y = std::numeric_limits<double>::max();
        double max3y = std::numeric_limits<double>::lowest();
        for (const auto& p : pts_3d) {
            min3x = std::min(min3x, p.x());  max3x = std::max(max3x, p.x());
            min3y = std::min(min3y, p.y());  max3y = std::max(max3y, p.y());
        }
        double span3d = std::hypot(max3x - min3x, max3y - min3y);
        std::cout << "[MonoPnP] 输入: n=" << n
                  << ", 2D bbox=[" << minx << "," << miny << "]~["
                  << maxx << "," << maxy << "] (" << (maxx - minx) << "x"
                  << (maxy - miny) << "px)"
                  << ", 3D span=" << span3d << "mm"
                  << ", K: f=(" << K(0, 0) << "," << K(1, 1) << ") c=("
                  << K(0, 2) << "," << K(1, 2) << ")" << std::endl;
    }

    // --- 2. PnP 求解 ---
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool pnp_ok = false;

    // 候选池与收集器（4点 seeded 与 >4 点路径共用）：
    // 计算误差 → 校验 → 打印 → 收集；返回是否有效。
    // collect_tol 为重投影验收容差，仅 4 点 seeded 路径会将其收紧为相对容差。
    std::vector<PnpCandidate> candidates;
    double collect_tol = kMaxReprojErrorPx;
    auto collect = [&](const std::string& tag, const cv::Mat& rv, const cv::Mat& tv) -> bool {
        double re = meanReprojError(object_points, pts_2d, rv, tv, K_cv);
        bool ok = isValidSolution(rv, tv, re, collect_tol);
        if (g_verbose_console)
            std::cout << "[MonoPnP] 候选 " << tag
                      << ": |t|=" << cv::norm(tv)
                      << ", t.z=" << tv.at<double>(2)
                      << ", 重投影=" << re << "px"
                      << (ok ? " [有效]" : " [无效]") << std::endl;
        if (ok) {
            PnpCandidate c;
            c.rvec = rv.clone();
            c.tvec = tv.clone();
            c.reproj = re;
            cv::Mat R_cv;
            cv::Rodrigues(rv, R_cv);
            for (int i = 0; i < 3; ++i) {
                c.t(i) = tv.at<double>(i);
                for (int j = 0; j < 3; ++j)
                    c.R(i, j) = R_cv.at<double>(i, j);
            }
            candidates.push_back(std::move(c));
        }
        return ok;
    };

    if (n == 4) {
        // 4 点直接 ITERATIVE：RANSAC 无意义（最小集=全集），EPnP 有共面二义性。
        if (seed == nullptr) {
            // 无 seed: 保持原行为（单次 ITERATIVE，不做校验）
            try {
                cv::Mat rv, tv;
                cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                             rv, tv, false, cv::SOLVEPNP_ITERATIVE);
                pnp_ok = !rv.empty() && !tv.empty();
                if (pnp_ok) { rvec = rv; tvec = tv; inliers = {0, 1, 2, 3}; }
            } catch (const cv::Exception& e) {
                std::cout << "[MonoPnP] ITERATIVE (4pts) 异常: " << e.what() << std::endl;
                return pose;
            }
        } else {
            // 有 seed: 相对容差 + 候选竞争（2026-09 深度棘轮冻结修复）。
            // 旧行为 = seed 初值 ITERATIVE + 8px 平坦校验：4 点共面二义性下
            // ITERATIVE 恒收敛到 seed 盆地，8px 在 15~30px 小目标上是 30~50%
            // 相对容差，陈旧 seed 解恒通过 → 逼近目标时深度冻结。
            // 现与冷启动 ITERATIVE / IPPE 同池竞争，坏 seed 解重投影显著更差，
            // 被相对容差在收集阶段拦截；全部候选失败时回退旧 8px 路径（不劣于原）。
            float minx = std::numeric_limits<float>::max();
            float maxx = std::numeric_limits<float>::lowest();
            float miny = std::numeric_limits<float>::max();
            float maxy = std::numeric_limits<float>::lowest();
            for (const auto& p : pts_2d) {
                minx = std::min(minx, p.x);  maxx = std::max(maxx, p.x);
                miny = std::min(miny, p.y);  maxy = std::max(maxy, p.y);
            }
            const double span2d = std::hypot(maxx - minx, maxy - miny);
            collect_tol = std::clamp(kSmallTargetReprojRatio * span2d,
                                     kMinReprojTolPx, kMaxReprojErrorPx);

            // ---- 候选W: seed 初值 ITERATIVE ----
            try {
                cv::Mat rv = seedRvec(*seed), tv = seedTvec(*seed);
                cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                             rv, tv,
                             true,                          // useExtrinsicGuess
                             cv::SOLVEPNP_ITERATIVE);
                collect("4pt-SeedITER", rv, tv);
            } catch (const cv::Exception& e) {
                std::cout << "[MonoPnP] 4点 Seed ITERATIVE 异常（忽略该候选）: "
                          << e.what() << std::endl;
            }

            // ---- 候选C: 冷启动 ITERATIVE ----
            try {
                cv::Mat rv, tv;
                cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                             rv, tv, false, cv::SOLVEPNP_ITERATIVE);
                collect("4pt-ColdITER", rv, tv);
            } catch (const cv::Exception& e) {
                std::cout << "[MonoPnP] 4点 Cold ITERATIVE 异常（忽略该候选）: "
                          << e.what() << std::endl;
            }

            // ---- 候选B: IPPE（共面闭式解，返回 ≤2 个解）----
            // TT 的 4 点在 z=0 平面，IPPE 适用；非共面输入抛异常，捕获后跳过。
            try {
                std::vector<cv::Mat> rvecs_ippe, tvecs_ippe;
                cv::solvePnPGeneric(object_points, pts_2d, K_cv, cv::Mat(),
                                    rvecs_ippe, tvecs_ippe,
                                    false, cv::SOLVEPNP_IPPE);
                for (size_t s = 0; s < rvecs_ippe.size(); ++s)
                    collect("4pt-IPPE#" + std::to_string(s),
                            rvecs_ippe[s], tvecs_ippe[s]);
            } catch (const cv::Exception& e) {
                if (g_verbose_console)
                    std::cout << "[MonoPnP] 4点 IPPE 跳过: " << e.what() << std::endl;
            }

            if (!candidates.empty()) {
                const PnpCandidate* best = selectBestCandidate(candidates, seed,
                                                               tie_epsilon_px);
                rvec = best->rvec;
                tvec = best->tvec;
                pnp_ok = true;
                if (g_verbose_console)
                    std::cout << "[MonoPnP] 4点择优: 重投影=" << best->reproj
                              << "px, |t|=" << cv::norm(tvec) << "mm"
                              << " (容差=" << collect_tol << "px, seed平票偏好)"
                              << std::endl;
                // 最终内点计数（重投影 < 8px 的点数）
                std::vector<cv::Point2f> proj;
                cv::projectPoints(object_points, rvec, tvec, K_cv, cv::Mat(), proj);
                inliers.clear();
                for (size_t i = 0; i < proj.size(); ++i) {
                    double err = std::hypot(proj[i].x - pts_2d[i].x,
                                            proj[i].y - pts_2d[i].y);
                    if (err < kMaxReprojErrorPx)
                        inliers.push_back(static_cast<int>(i));
                }
                if (inliers.empty()) inliers.push_back(0);  // 防御，有效解下不会发生
            } else {
                // 全部候选未过相对容差 → 回退旧平坦 8px 路径（保持原行为可达）
                std::cout << "[MonoPnP] 4点候选均未过相对容差 (" << collect_tol
                          << "px), 回退平坦 8px 路径" << std::endl;
                try {
                    cv::Mat rv = seedRvec(*seed), tv = seedTvec(*seed);
                    cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                                 rv, tv,
                                 true,                      // useExtrinsicGuess
                                 cv::SOLVEPNP_ITERATIVE);
                    pnp_ok = !rv.empty() && !tv.empty();
                    if (pnp_ok) {
                        double re = meanReprojError(object_points, pts_2d, rv, tv, K_cv);
                        if (!isValidSolution(rv, tv, re)) pnp_ok = false;
                    }
                    if (!pnp_ok) {
                        // seeded 校验失败 → 无 seed 冷解兜底
                        rv.release(); tv.release();
                        cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                                     rv, tv, false, cv::SOLVEPNP_ITERATIVE);
                        pnp_ok = !rv.empty() && !tv.empty();
                    }
                    if (pnp_ok) { rvec = rv; tvec = tv; inliers = {0, 1, 2, 3}; }
                } catch (const cv::Exception& e) {
                    std::cout << "[MonoPnP] 4点回退 ITERATIVE 异常: "
                              << e.what() << std::endl;
                }
            }
        }
    } else {
        // >4 点: 多候选求解（EPnP RANSAC + 共面 IPPE），按平均重投影误差择优
        // （候选池 candidates 与收集器 collect 已在本分支外共用声明）

        // ---- 候选W: 上帧位姿 seed 的 ITERATIVE 解（时序连贯性候选）----
        if (seed) {
            try {
                cv::Mat rv = seedRvec(*seed), tv = seedTvec(*seed);
                cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                             rv, tv,
                             true,                          // useExtrinsicGuess
                             cv::SOLVEPNP_ITERATIVE);
                collect("SeedITER", rv, tv);
            } catch (const cv::Exception& e) {
                std::cout << "[MonoPnP] Seed ITERATIVE 异常（忽略该候选）: " << e.what() << std::endl;
            }
        }

        // ---- 候选A: EPnP RANSAC + inlier ITERATIVE 精化 ----
        std::vector<int> ransac_inliers;
        bool ransac_ok = false;
        try {
            cv::Mat rv, tv;
            cv::solvePnPRansac(object_points, pts_2d, K_cv, cv::Mat(),
                               rv, tv,
                               false,                        // useExtrinsicGuess
                               300,                          // iterationsCount
                               8.0,                          // reprojectionError
                               0.99,                         // confidence
                               ransac_inliers,
                               cv::SOLVEPNP_EPNP);
            ransac_ok = !rv.empty() && !tv.empty() &&
                        ransac_inliers.size() >= 4;
            if (ransac_ok) { rvec = rv; tvec = tv; }
        } catch (const cv::Exception& e) {
            std::cout << "[MonoPnP] solvePnPRansac 异常: " << e.what() << std::endl;
        }

        if (!ransac_ok) {
            std::cout << "[MonoPnP] RANSAC EPnP 失败（内点="
                      << ransac_inliers.size() << "）" << std::endl;
        } else if (collect("EPnP_RANSAC", rvec, tvec)) {
            // 仅当 RANSAC 结果有效时才作为 ITERATIVE 初值（垃圾初值必然精化出垃圾）
            std::vector<cv::Point3f> inl_obj;
            std::vector<cv::Point2f> inl_img;
            inl_obj.reserve(ransac_inliers.size());
            inl_img.reserve(ransac_inliers.size());
            for (int idx : ransac_inliers) {
                inl_obj.push_back(object_points[idx]);
                inl_img.push_back(pts_2d[idx]);
            }
            try {
                cv::Mat rv = rvec.clone(), tv = tvec.clone();
                cv::solvePnP(inl_obj, inl_img, K_cv, cv::Mat(),
                             rv, tv,
                             true,                          // useExtrinsicGuess
                             cv::SOLVEPNP_ITERATIVE);
                collect("ITER精化", rv, tv);
            } catch (const cv::Exception& e) {
                std::cout << "[MonoPnP] ITERATIVE 精化异常（保留 RANSAC 结果）: "
                          << e.what() << std::endl;
            }
        }

        // ---- 候选B: IPPE（共面专用闭式解，返回 ≤2 个解）----
        // 单目全部策略（BC/AKAZE/TT）的 3D 点均在 z=0 平面，IPPE 适用；
        // 非共面输入会抛异常，捕获后跳过即可。
        try {
            std::vector<cv::Mat> rvecs_ippe, tvecs_ippe;
            cv::solvePnPGeneric(object_points, pts_2d, K_cv, cv::Mat(),
                                rvecs_ippe, tvecs_ippe,
                                false, cv::SOLVEPNP_IPPE);
            for (size_t s = 0; s < rvecs_ippe.size(); ++s)
                collect("IPPE#" + std::to_string(s), rvecs_ippe[s], tvecs_ippe[s]);
        } catch (const cv::Exception& e) {
            if (g_verbose_console)
                std::cout << "[MonoPnP] IPPE 跳过: " << e.what() << std::endl;
        }

        // ---- 择优: 两阶段（ε-平票 → seed 时序偏好 / 纯重投影最小）----
        if (candidates.empty()) {
            std::cout << "[MonoPnP] 无有效候选（|t|越界/t.z≤0/重投影≥"
                      << kMaxReprojErrorPx << "px），输入对应关系或内参可疑" << std::endl;
            return pose;
        }
        const PnpCandidate* best = selectBestCandidate(candidates, seed, tie_epsilon_px);
        rvec = best->rvec;
        tvec = best->tvec;
        pnp_ok = true;
        if (g_verbose_console)
            std::cout << "[MonoPnP] 择优: 重投影=" << best->reproj
                      << "px, |t|=" << cv::norm(tvec) << "mm"
                      << (seed ? " (seed平票偏好)" : "") << std::endl;

        // 最终内点计数（重投影 < 阈值的点数）
        std::vector<cv::Point2f> proj;
        cv::projectPoints(object_points, rvec, tvec, K_cv, cv::Mat(), proj);
        inliers.clear();
        for (size_t i = 0; i < proj.size(); ++i) {
            double err = std::hypot(proj[i].x - pts_2d[i].x,
                                    proj[i].y - pts_2d[i].y);
            if (err < kMaxReprojErrorPx)
                inliers.push_back(static_cast<int>(i));
        }
        if (inliers.empty()) inliers.push_back(0);  // 防御，有效解下不会发生
    }

    if (!pnp_ok) {
        std::cout << "[MonoPnP] PnP 失败" << std::endl;
        return pose;
    }

    // --- 4. 转换为 Eigen 格式 ---
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for (int i = 0; i < 3; ++i) {
        t(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; ++j) {
            R(i, j) = R_cv.at<double>(i, j);
        }
    }

    // --- 5. 有效性校验（防御：择优阶段已过滤，正常不再触发） ---
    // t[2] > 0: 相机必须在模板平面前方
    if (t(2) <= 0.0) {
        std::cout << "[MonoPnP] 无效：t.z = " << t(2) << " ≤ 0（相机在模板后方）" << std::endl;
        return pose;
    }

    double t_norm = t.norm();
    if (t_norm < 10.0 || t_norm > 100000.0) {
        std::cout << "[MonoPnP] 无效：|t| = " << t_norm
                  << " mm（超出 [10, 20000]）" << std::endl;
        return pose;
    }

    if (!R.allFinite() || !t.allFinite()) {
        std::cout << "[MonoPnP] 无效：结果包含非有限值" << std::endl;
        return pose;
    }

    // --- 成功 ---
    pose.R = R;
    pose.t = t;
    pose.success = true;
    pose.num_points = static_cast<int>(inliers.size());

    if (g_verbose_console)
        std::cout << "[MonoPnP] 成功: inliers=" << inliers.size()
                  << "/" << n << ", t=(" << t(0) << ", " << t(1) << ", " << t(2) << ") mm"
                  << std::endl;

    return pose;
}

} // namespace gpnp
