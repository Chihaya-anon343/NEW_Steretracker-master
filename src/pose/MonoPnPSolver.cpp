/**
 * @file MonoPnPSolver.cpp
 * @brief 单目 PnP 求解器实现：多候选（EPnP RANSAC + 共面 IPPE）+ 重投影误差择优。
 */

#include "pose/MonoPnPSolver.hpp"
#include "pose/BoundedMonoRefinement.hpp"
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

bool refineMonoWithinTilt(const std::vector<cv::Point2f>& image,
        const std::vector<Eigen::Vector3d>& model, const Eigen::Matrix3d& K,
        double max_tilt_deg, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    if(image.size()!=model.size() || image.size()<4 || !R.allFinite() || !t.allFinite()) return false;
    const double limit=max_tilt_deg*CV_PI/180.;
    const double depth_scale=std::max(t.norm(),100.);
    using Vector6=Eigen::Matrix<double,6,1>;
    Vector6 x;
    const auto normal=R.col(2).eval();
    const double tilt=std::acos(std::clamp(normal.z(),-1.,1.));
    Eigen::Vector3d axis(-normal.y(),normal.x(),0.);
    if(axis.norm()>1e-12) axis.normalize(); else axis=Eigen::Vector3d::UnitX();
    const auto base=Eigen::AngleAxisd(tilt,axis).toRotationMatrix();
    const Eigen::Matrix3d spin=base.transpose()*R;
    x<<tilt*axis.x(),tilt*axis.y(),std::atan2(spin(1,0),spin(0,0)),t.x()/depth_scale,t.y()/depth_scale,t.z()/depth_scale;
    auto projectCone=[&](Vector6& p) {
        const double norm=p.head<2>().norm();
        if(norm>limit) p.head<2>()*=limit/norm;
    };
    auto unpack=[&](const Vector6& p,Eigen::Matrix3d& rot,Eigen::Vector3d& trans) {
        const Eigen::Vector3d v(p[0],p[1],0.);
        const double angle=v.norm();
        const Eigen::Matrix3d lean=angle>1e-12 ? Eigen::AngleAxisd(angle,v/angle).toRotationMatrix() : Eigen::Matrix3d::Identity();
        rot=lean*Eigen::AngleAxisd(p[2],Eigen::Vector3d::UnitZ()).toRotationMatrix();
        trans=p.tail<3>()*depth_scale;
    };
    auto residual=[&](const Vector6& p,Eigen::VectorXd& r) {
        Eigen::Matrix3d rot;Eigen::Vector3d trans;unpack(p,rot,trans);
        r.resize(2*model.size());
        for(size_t i=0;i<model.size();++i) {
            const Eigen::Vector3d point=rot*model[i]+trans;
            if(!point.allFinite() || point.z()<=1e-6) return false;
            const Eigen::Vector3d q=K*point;
            r[2*i]=q.x()/q.z()-image[i].x;
            r[2*i+1]=q.y()/q.z()-image[i].y;
        }
        return r.allFinite();
    };
    projectCone(x);
    Eigen::VectorXd r;if(!residual(x,r)) return false;
    double cost=r.squaredNorm(),lambda=1e-3;
    for(int iteration=0;iteration<25;++iteration) {
        Eigen::MatrixXd J(r.size(),6);
        bool jac_valid=true;
        for(int k=0;k<6;++k) {
            Vector6 plus=x,minus=x;plus[k]+=1e-6;minus[k]-=1e-6;
            Eigen::VectorXd rp,rm;
            if(!residual(plus,rp)||!residual(minus,rm)){jac_valid=false;break;}
            J.col(k)=(rp-rm)/2e-6;
        }
        if(!jac_valid) break;
        const Eigen::Matrix<double,6,6> H=J.transpose()*J;
        Eigen::Matrix<double,6,6> damped=H;
        for(int k=0;k<6;++k)damped(k,k)+=lambda*std::max(H(k,k),1.);
        const Vector6 delta=damped.ldlt().solve(-J.transpose()*r);
        if(!delta.allFinite()) break;
        Vector6 trial=x+delta;projectCone(trial);
        Eigen::VectorXd next;
        if(residual(trial,next) && next.squaredNorm()<cost) {
            const double decrease=cost-next.squaredNorm();
            const double step=(trial-x).norm();
            x=trial;r=next;cost=r.squaredNorm();lambda=std::max(1e-9,lambda*.3);
            if(step<1e-8 || decrease<1e-10) break;
        } else {
            lambda*=10.;if(lambda>1e10) break;
        }
    }
    unpack(x,R,t);
    return R.allFinite()&&t.allFinite();
}

PoseEstimate MonoPnPSolver::solve(const std::vector<cv::Point2f>& pts_2d,
                                   const std::vector<Eigen::Vector3d>& pts_3d,
                                   const Eigen::Matrix3d& K) {
    return solve(pts_2d, pts_3d, K, nullptr, 0.0);
}

PoseEstimate MonoPnPSolver::solve(const std::vector<cv::Point2f> &pts_2d,
                                  const std::vector<Eigen::Vector3d> &pts_3d,
                                  const Eigen::Matrix3d &K,
                                  const PoseSeed *seed, double tie_epsilon_px,
                                  int min_support) {
  PoseEstimate pose;
  last_rejection_="mono_pnp_no_valid_candidate";

  const int n = static_cast<int>(pts_2d.size());
  if (min_support < 4 || n < min_support || pts_3d.size() != pts_2d.size()) {
    last_rejection_="insufficient_pose_points";
    std::cout << "[MonoPnP] 点数不足: pts_2d=" << n
              << ", pts_3d=" << pts_3d.size() << " (需要 ≥4)"
              << "\n";
    return pose;
  }
  if (prior_.enabled) {
    if (!K.allFinite() || K(0, 0) <= 0.0 || K(1, 1) <= 0.0)
      return pose;
    for (size_t i = 0; i < pts_2d.size(); ++i)
      if (!std::isfinite(pts_2d[i].x) || !std::isfinite(pts_2d[i].y) ||
          !pts_3d[i].allFinite() || std::abs(pts_3d[i].z()) > 1e-6)
        return pose;
  }

  // --- 1. 将Eigen::Vector3d 3D 点转换为 cv::Point3f ---
  std::vector<cv::Point3f> object_points;
  object_points.reserve(n);
  for (const auto &p : pts_3d) {
    object_points.emplace_back(static_cast<float>(p.x()),
                               static_cast<float>(p.y()),
                               static_cast<float>(p.z()));
  }

  // 构造相机内参矩阵
  cv::Mat K_cv = (cv::Mat_<double>(3, 3) << K(0, 0), K(0, 1), K(0, 2), K(1, 0),
                  K(1, 1), K(1, 2), K(2, 0), K(2, 1), K(2, 2));

  // --- 诊断: 输入概况 ---
  if (g_verbose_console) {
    float minx = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::lowest();
    float miny = std::numeric_limits<float>::max();
    float maxy = std::numeric_limits<float>::lowest();
    for (const auto &p : pts_2d) {
      minx = std::min(minx, p.x);
      maxx = std::max(maxx, p.x);
      miny = std::min(miny, p.y);
      maxy = std::max(maxy, p.y);
    }
    double min3x = std::numeric_limits<double>::max();
    double max3x = std::numeric_limits<double>::lowest();
    double min3y = std::numeric_limits<double>::max();
    double max3y = std::numeric_limits<double>::lowest();
    for (const auto &p : pts_3d) {
      min3x = std::min(min3x, p.x());
      max3x = std::max(max3x, p.x());
      min3y = std::min(min3y, p.y());
      max3y = std::max(max3y, p.y());
    }
    double span3d = std::hypot(max3x - min3x, max3y - min3y);
    std::cout << "[MonoPnP] 输入: n=" << n << ", 2D bbox=[" << minx << ","
              << miny << "]~[" << maxx << "," << maxy << "] (" << (maxx - minx)
              << "x" << (maxy - miny) << "px)"
              << ", 3D span=" << span3d << "mm"
              << ", K: f=(" << K(0, 0) << "," << K(1, 1) << ") c=(" << K(0, 2)
              << "," << K(1, 2) << ")"
              << "\n";
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
  auto collect = [&](const std::string &tag, const cv::Mat &rv,
                     const cv::Mat &tv) -> bool {
    if (prior_.enabled && (rv.empty() || tv.empty() || !cv::checkRange(rv) ||
                           !cv::checkRange(tv)))
      return false;
    double re = meanReprojError(object_points, pts_2d, rv, tv, K_cv);
    bool ok = isValidSolution(rv, tv, re, collect_tol);
    auto enough_support = [&](const cv::Mat &r, const cv::Mat &t) {
      if (min_support <= 4)
        return true;
      std::vector<cv::Point2f> projected;
      cv::projectPoints(object_points, r, t, K_cv, cv::Mat(), projected);
      int count = 0;
      for (size_t i = 0; i < projected.size(); ++i)
        if (cv::norm(projected[i] - pts_2d[i]) < collect_tol)
          ++count;
      return count >= min_support;
    };
    if (ok && !enough_support(rv, tv))
      ok = false;
    if (g_verbose_console)
      std::cout << "[MonoPnP] 候选 " << tag << ": |t|=" << cv::norm(tv)
                << ", t.z=" << tv.at<double>(2) << ", 重投影=" << re << "px"
                << (ok ? " [有效]" : " [无效]") << "\n";
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
      auto rejection = planarPosePriorRejection(c.R, c.t, pts_3d, prior_);
      if (prior_.enabled && rejection == "tilt_out_of_range") {
        // A slightly out-of-cone observation must be refit at the
        // physical boundary, rather than selecting its wrong mirror.
        if (refineMonoWithinTilt(pts_2d, pts_3d, K, prior_.max_tilt_deg, c.R,
                                 c.t)) {
          cv::Mat bounded_R(3, 3, CV_64F), bounded_t(3, 1, CV_64F);
          for (int i = 0; i < 3; ++i) {
            bounded_t.at<double>(i) = c.t(i);
            for (int j = 0; j < 3; ++j)
              bounded_R.at<double>(i, j) = c.R(i, j);
          }
          cv::Rodrigues(bounded_R, c.rvec);
          c.tvec = bounded_t;
          c.reproj =
              meanReprojError(object_points, pts_2d, c.rvec, c.tvec, K_cv);
          rejection = planarPosePriorRejection(c.R, c.t, pts_3d, prior_);
          if (!isValidSolution(c.rvec, c.tvec, c.reproj, collect_tol) ||
              !enough_support(c.rvec, c.tvec))
            rejection = "bounded_reprojection_failed";
        }
      }
      if(!rejection.empty())last_rejection_="pose_prior_rejected:"+rejection;
      if (rejection.empty()) {
        candidates.push_back(std::move(c));
      } else if (g_verbose_console) {
        const double tilt =
            std::acos(std::clamp(c.R(2, 2), -1.0, 1.0)) * 180.0 / CV_PI;
        std::cout << "[MonoPnP][Prior] reject " << tag
                  << " reason=" << rejection << " tilt=" << tilt
                  << " limit=" << prior_.max_tilt_deg << "deg\n";
      }
    }
    // A geometrically inadmissible EPnP initialization may refine to an
    // admissible pose: return base validity, never add it to the pose pool.
    return ok;
  };

  if (n == 4) {
    // 4 点直接 ITERATIVE：RANSAC 无意义（最小集=全集），EPnP 有共面二义性。
    if (seed == nullptr && !prior_.enabled) {
      // 无 seed: 保持原行为（单次 ITERATIVE，不做校验）
      try {
        cv::Mat rv, tv;
        cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(), rv, tv, false,
                     cv::SOLVEPNP_ITERATIVE);
        pnp_ok = !rv.empty() && !tv.empty();
        if (pnp_ok) {
          rvec = rv;
          tvec = tv;
          inliers = {0, 1, 2, 3};
        }
      } catch (const cv::Exception &e) {
        std::cout << "[MonoPnP] ITERATIVE (4pts) 异常: " << e.what() << "\n";
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
      for (const auto &p : pts_2d) {
        minx = std::min(minx, p.x);
        maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y);
        maxy = std::max(maxy, p.y);
      }
      const double span2d = std::hypot(maxx - minx, maxy - miny);
      collect_tol = std::clamp(kSmallTargetReprojRatio * span2d,
                               kMinReprojTolPx, kMaxReprojErrorPx);

      // ---- 候选W: seed 初值 ITERATIVE ----
      if (seed)
        try {
          cv::Mat rv = seedRvec(*seed), tv = seedTvec(*seed);
          cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(), rv, tv,
                       true, // useExtrinsicGuess
                       cv::SOLVEPNP_ITERATIVE);
          collect("4pt-SeedITER", rv, tv);
        } catch (const cv::Exception &e) {
          std::cout << "[MonoPnP] 4点 Seed ITERATIVE 异常（忽略该候选）: "
                    << e.what() << "\n";
        }

      // ---- 候选C: 冷启动 ITERATIVE ----
      try {
        cv::Mat rv, tv;
        cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(), rv, tv, false,
                     cv::SOLVEPNP_ITERATIVE);
        collect("4pt-ColdITER", rv, tv);
      } catch (const cv::Exception &e) {
        std::cout << "[MonoPnP] 4点 Cold ITERATIVE 异常（忽略该候选）: "
                  << e.what() << "\n";
      }

      // ---- 候选B: IPPE（共面闭式解，返回 ≤2 个解）----
      // TT 的 4 点在 z=0 平面，IPPE 适用；非共面输入抛异常，捕获后跳过。
      try {
        std::vector<cv::Mat> rvecs_ippe, tvecs_ippe;
        cv::solvePnPGeneric(object_points, pts_2d, K_cv, cv::Mat(), rvecs_ippe,
                            tvecs_ippe, false, cv::SOLVEPNP_IPPE);
        for (size_t s = 0; s < rvecs_ippe.size(); ++s)
          collect("4pt-IPPE#" + std::to_string(s), rvecs_ippe[s],
                  tvecs_ippe[s]);
      } catch (const cv::Exception &e) {
        if (g_verbose_console)
          std::cout << "[MonoPnP] 4点 IPPE 跳过: " << e.what() << "\n";
      }

      if (!candidates.empty()) {
        const PnpCandidate *best =
            selectBestCandidate(candidates, seed, tie_epsilon_px);
        rvec = best->rvec;
        tvec = best->tvec;
        pnp_ok = true;
        if (g_verbose_console)
          std::cout << "[MonoPnP] 4点择优: 重投影=" << best->reproj
                    << "px, |t|=" << cv::norm(tvec) << "mm"
                    << " (容差=" << collect_tol << "px, seed平票偏好)"
                    << "\n";
        // 最终内点计数（重投影 < 8px 的点数）
        std::vector<cv::Point2f> proj;
        cv::projectPoints(object_points, rvec, tvec, K_cv, cv::Mat(), proj);
        inliers.clear();
        for (size_t i = 0; i < proj.size(); ++i) {
          double err =
              std::hypot(proj[i].x - pts_2d[i].x, proj[i].y - pts_2d[i].y);
          if (err < kMaxReprojErrorPx)
            inliers.push_back(static_cast<int>(i));
        }
        if (inliers.empty())
          inliers.push_back(0); // 防御，有效解下不会发生
      } else if (prior_.enabled) {
        // Do not escape the physical gate through the old flat-8px
        // fallback, including first-frame/cold and class1 TT calls.
        if (g_verbose_console)
          std::cout << "[MonoPnP][Prior] no admissible 4-point candidate\n";
        return pose;
      } else {
        // 全部候选未过相对容差 → 回退旧平坦 8px 路径（保持原行为可达）
        std::cout << "[MonoPnP] 4点候选均未过相对容差 (" << collect_tol
                  << "px), 回退平坦 8px 路径"
                  << "\n";
        try {
          cv::Mat rv = seedRvec(*seed), tv = seedTvec(*seed);
          cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(), rv, tv,
                       true, // useExtrinsicGuess
                       cv::SOLVEPNP_ITERATIVE);
          pnp_ok = !rv.empty() && !tv.empty();
          if (pnp_ok) {
            double re = meanReprojError(object_points, pts_2d, rv, tv, K_cv);
            if (!isValidSolution(rv, tv, re))
              pnp_ok = false;
          }
          if (!pnp_ok) {
            // seeded 校验失败 → 无 seed 冷解兜底
            rv.release();
            tv.release();
            cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(), rv, tv, false,
                         cv::SOLVEPNP_ITERATIVE);
            pnp_ok = !rv.empty() && !tv.empty();
          }
          if (pnp_ok) {
            rvec = rv;
            tvec = tv;
            inliers = {0, 1, 2, 3};
          }
        } catch (const cv::Exception &e) {
          std::cout << "[MonoPnP] 4点回退 ITERATIVE 异常: " << e.what() << "\n";
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
        cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(), rv, tv,
                     true, // useExtrinsicGuess
                     cv::SOLVEPNP_ITERATIVE);
        collect("SeedITER", rv, tv);
      } catch (const cv::Exception &e) {
        std::cout << "[MonoPnP] Seed ITERATIVE 异常（忽略该候选）: " << e.what()
                  << "\n";
      }
    }

    // ---- 候选A: EPnP RANSAC + inlier ITERATIVE 精化 ----
    std::vector<int> ransac_inliers;
    bool ransac_ok = false;
    try {
      cv::Mat rv, tv;
      cv::solvePnPRansac(object_points, pts_2d, K_cv, cv::Mat(), rv, tv,
                         false, // useExtrinsicGuess
                         300,   // iterationsCount
                         8.0,   // reprojectionError
                         0.99,  // confidence
                         ransac_inliers, cv::SOLVEPNP_EPNP);
      ransac_ok = !rv.empty() && !tv.empty() && ransac_inliers.size() >= 4;
      if (ransac_ok) {
        rvec = rv;
        tvec = tv;
      }
    } catch (const cv::Exception &e) {
      std::cout << "[MonoPnP] solvePnPRansac 异常: " << e.what() << "\n";
    }

    if (!ransac_ok) {
      std::cout << "[MonoPnP] RANSAC EPnP 失败（内点=" << ransac_inliers.size()
                << "）"
                << "\n";
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
        cv::solvePnP(inl_obj, inl_img, K_cv, cv::Mat(), rv, tv,
                     true, // useExtrinsicGuess
                     cv::SOLVEPNP_ITERATIVE);
        collect("ITER精化", rv, tv);
      } catch (const cv::Exception &e) {
        std::cout << "[MonoPnP] ITERATIVE 精化异常（保留 RANSAC 结果）: "
                  << e.what() << "\n";
      }
    }

    // ---- 候选B: IPPE（共面专用闭式解，返回 ≤2 个解）----
    // 单目全部策略（BC/AKAZE/TT）的 3D 点均在 z=0 平面，IPPE 适用；
    // 非共面输入会抛异常，捕获后跳过即可。
    try {
      std::vector<cv::Mat> rvecs_ippe, tvecs_ippe;
      cv::solvePnPGeneric(object_points, pts_2d, K_cv, cv::Mat(), rvecs_ippe,
                          tvecs_ippe, false, cv::SOLVEPNP_IPPE);
      for (size_t s = 0; s < rvecs_ippe.size(); ++s)
        collect("IPPE#" + std::to_string(s), rvecs_ippe[s], tvecs_ippe[s]);
    } catch (const cv::Exception &e) {
      if (g_verbose_console)
        std::cout << "[MonoPnP] IPPE 跳过: " << e.what() << "\n";
    }

    // ---- 择优: 两阶段（ε-平票 → seed 时序偏好 / 纯重投影最小）----
    if (candidates.empty()) {
      std::cout << "[MonoPnP] 无有效候选（|t|越界/t.z≤0/重投影≥"
                << kMaxReprojErrorPx << "px），输入对应关系或内参可疑"
                << "\n";
      return pose;
    }
    const PnpCandidate *best =
        selectBestCandidate(candidates, seed, tie_epsilon_px);
    rvec = best->rvec;
    tvec = best->tvec;
    pnp_ok = true;
    if (g_verbose_console)
      std::cout << "[MonoPnP] 择优: 重投影=" << best->reproj
                << "px, |t|=" << cv::norm(tvec) << "mm"
                << (seed ? " (seed平票偏好)" : "") << "\n";

    // 最终内点计数（重投影 < 阈值的点数）
    std::vector<cv::Point2f> proj;
    cv::projectPoints(object_points, rvec, tvec, K_cv, cv::Mat(), proj);
    inliers.clear();
    for (size_t i = 0; i < proj.size(); ++i) {
      double err = std::hypot(proj[i].x - pts_2d[i].x, proj[i].y - pts_2d[i].y);
      if (err < kMaxReprojErrorPx)
        inliers.push_back(static_cast<int>(i));
    }
    if (inliers.empty())
      inliers.push_back(0); // 防御，有效解下不会发生
  }

  if (!pnp_ok) {
    std::cout << "[MonoPnP] PnP 失败"
              << "\n";
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
    std::cout << "[MonoPnP] 无效：t.z = " << t(2) << " ≤ 0（相机在模板后方）"
              << "\n";
    return pose;
  }

  double t_norm = t.norm();
  if (t_norm < 10.0 || t_norm > 100000.0) {
    std::cout << "[MonoPnP] 无效：|t| = " << t_norm << " mm（超出 [10, 20000]）"
              << "\n";
    return pose;
  }

  if (!R.allFinite() || !t.allFinite()) {
    std::cout << "[MonoPnP] 无效：结果包含非有限值"
              << "\n";
    return pose;
  }
  const auto prior_rejection = planarPosePriorRejection(R, t, pts_3d, prior_);
  if (!prior_rejection.empty()) {
    last_rejection_="pose_prior_rejected:"+prior_rejection;
    if (g_verbose_console)
      std::cout << "[MonoPnP][Prior] final rejection: " << prior_rejection
                << '\n';
    return pose;
  }

  // --- 成功 ---
  pose.R = R;
  pose.t = t;
  pose.success = true;
  last_rejection_.clear();
  pose.num_points = static_cast<int>(inliers.size());

  if (g_verbose_console)
    std::cout << "[MonoPnP] 成功: inliers=" << inliers.size() << "/" << n
              << ", t=(" << t(0) << ", " << t(1) << ", " << t(2) << ") mm"
              << "\n";

  return pose;
}

} // namespace gpnp
