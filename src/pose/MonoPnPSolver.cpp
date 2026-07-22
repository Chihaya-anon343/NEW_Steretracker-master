/**
 * @file MonoPnPSolver.cpp
 * @brief 单目 EPnP 求解器实现。
 */

#include "pose/MonoPnPSolver.hpp"
#include "common/LogConfig.hpp"

#include <opencv2/calib3d.hpp>
#include <iostream>

namespace gpnp {

PoseEstimate MonoPnPSolver::solve(const std::vector<cv::Point2f>& pts_2d,
                                   const std::vector<Eigen::Vector3d>& pts_3d,
                                   const Eigen::Matrix3d& K) {
    PoseEstimate pose;

    const int n = static_cast<int>(pts_2d.size());
    if (n < 4 || pts_3d.size() != pts_2d.size()) {
        std::cerr << "[MonoPnP] 点数不足: pts_2d=" << n
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

    // --- 2. PnP 求解 ---
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool pnp_ok = false;

    if (n == 4) {
        // 4 点直接 ITERATIVE：RANSAC 无意义（最小集=全集），EPnP 有共面二义性
        try {
            cv::solvePnP(object_points, pts_2d, K_cv, cv::Mat(),
                         rvec, tvec,
                         false,
                         cv::SOLVEPNP_ITERATIVE);
            pnp_ok = !rvec.empty() && !tvec.empty();
            if (pnp_ok) inliers = {0, 1, 2, 3};
        } catch (const cv::Exception& e) {
            std::cerr << "[MonoPnP] ITERATIVE (4pts) 异常: " << e.what() << std::endl;
            return pose;
        }
    } else {
        // >4 点: RANSAC EPnP + ITERATIVE 精化
        try {
            cv::solvePnPRansac(object_points, pts_2d, K_cv, cv::Mat(),
                               rvec, tvec,
                               false,                        // useExtrinsicGuess
                               300,                          // iterationsCount
                               8.0,                          // reprojectionError
                               0.99,                         // confidence
                               inliers,
                               cv::SOLVEPNP_EPNP);
            pnp_ok = !rvec.empty() && !tvec.empty() &&
                     inliers.size() >= 4;
        } catch (const cv::Exception& e) {
            std::cerr << "[MonoPnP] solvePnPRansac 异常: " << e.what() << std::endl;
            return pose;
        }

        if (!pnp_ok) {
            std::cerr << "[MonoPnP] RANSAC EPnP 失败（内点="
                      << inliers.size() << "）" << std::endl;
            return pose;
        }

        // 用 inlier 子集做 ITERATIVE 精化
        {
            std::vector<cv::Point3f> inl_obj;
            std::vector<cv::Point2f> inl_img;
            inl_obj.reserve(inliers.size());
            inl_img.reserve(inliers.size());
            for (int idx : inliers) {
                inl_obj.push_back(object_points[idx]);
                inl_img.push_back(pts_2d[idx]);
            }

            try {
                cv::solvePnP(inl_obj, inl_img, K_cv, cv::Mat(),
                             rvec, tvec,
                             true,                          // useExtrinsicGuess
                             cv::SOLVEPNP_ITERATIVE);
            } catch (const cv::Exception& e) {
                std::cerr << "[MonoPnP] ITERATIVE 精化异常（使用 RANSAC 结果）: "
                          << e.what() << std::endl;
            }
        }
    }

    if (!pnp_ok) {
        std::cerr << "[MonoPnP] PnP 失败" << std::endl;
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

    // --- 5. 有效性校验 ---
    // t[2] > 0: 相机必须在模板平面前方
    if (t(2) <= 0.0) {
        std::cerr << "[MonoPnP] 无效：t.z = " << t(2) << " ≤ 0（相机在模板后方）" << std::endl;
        return pose;
    }

    double t_norm = t.norm();
    if (t_norm < 10.0 || t_norm > 100000.0) {
        std::cerr << "[MonoPnP] 无效：|t| = " << t_norm
                  << " mm（超出 [10, 20000]）" << std::endl;
        return pose;
    }

    if (!R.allFinite() || !t.allFinite()) {
        std::cerr << "[MonoPnP] 无效：结果包含非有限值" << std::endl;
        return pose;
    }

    // --- 成功 ---
    pose.R = R;
    pose.t = t;
    pose.success = true;
    pose.num_points = static_cast<int>(inliers.size());

    if (g_verbose_console)
        std::cout << "[MonoPnP] EPnP 成功: inliers=" << inliers.size()
                  << "/" << n << ", t=(" << t(0) << ", " << t(1) << ", " << t(2) << ") mm"
              << std::endl;

    return pose;
}

} // namespace gpnp