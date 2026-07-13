#include "pose/InitialPnPSolver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <iostream>

namespace gpnp {

InitialPnPSolver::InitialPnPSolver(int gpnp_min_pts)
    : gpnp_min_pts_(gpnp_min_pts)
{
}

// ============================================================================
// 位姿估计
// ============================================================================

PoseEstimate InitialPnPSolver::solve(
    const MatchResult& match_result,
    const std::vector<Eigen::Vector3d>& pts_3d_T,
    const Eigen::Matrix3d& K)
{
    PoseEstimate result;
    result.success = false;

    const auto& good_matches = match_result.good_matches;
    const auto& pts_left_match = match_result.pts_left_match;

    // 硬编码回退值（匹配 Python 默认值）
    const Eigen::Vector3d t_hardcoded(0.0, 0.0, 500.0); // 500mm 深度

    if (static_cast<int>(good_matches.size()) < gpnp_min_pts_) {
        std::cout << "  [InitialPnP] Skip | good_matches=" << good_matches.size()
                  << " < " << gpnp_min_pts_ << std::endl;
        result.R = Eigen::Matrix3d::Identity();
        result.t = t_hardcoded;
        return result;
    }

    // 提取 3D-2D 对应关系
    std::vector<cv::Point3d> object_pts;
    std::vector<cv::Point2d> image_pts;
    object_pts.reserve(good_matches.size());
    image_pts.reserve(good_matches.size());

    for (size_t i = 0; i < good_matches.size(); ++i) {
        int train_idx = good_matches[i].trainIdx;
        if (train_idx < 0 || train_idx >= static_cast<int>(pts_3d_T.size())) continue;
        if (i >= pts_left_match.size()) continue;

        const auto& p3d = pts_3d_T[train_idx];
        object_pts.emplace_back(p3d.x(), p3d.y(), p3d.z());
        image_pts.emplace_back(pts_left_match[i].x, pts_left_match[i].y);
    }

    int n_pts = static_cast<int>(object_pts.size());
    if (n_pts < gpnp_min_pts_) {
        std::cout << "  [InitialPnP] Skip | n_pts=" << n_pts
                  << " < " << gpnp_min_pts_ << std::endl;
        result.R = Eigen::Matrix3d::Identity();
        result.t = t_hardcoded;
        return result;
    }

    std::cout << "  [InitialPnP] " << n_pts << " 3D↔2D 对应关系" << std::endl;

    // 将 K 转换为 cv::Mat
    cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
        K(0,0), K(0,1), K(0,2),
        K(1,0), K(1,1), K(1,2),
        K(2,0), K(2,1), K(2,2));

    try {
        // ---- 步骤1: RANSAC PnP ----
        // 预初始化 rvec/tvec 为零 — OpenCV 4.5.4 的 solvePnPGeneric
        // （由 solvePnPRansac 内部调用）在使用 SOLVEPNP_ITERATIVE 时要求
        // 非空初始猜测，不同于 Python 包装器内部预分配这些值。
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat inliers;
        bool ransac_ok = cv::solvePnPRansac(
            object_pts, image_pts, K_cv, cv::noArray(),
            rvec, tvec, true,  // useExtrinsicGuess=true（ITERATIVE 必需）
            300,               // 迭代次数
            8.0,               // 重投影误差
            0.99,              // 置信度
            inliers,
            cv::SOLVEPNP_ITERATIVE);

        if (!ransac_ok || rvec.empty() || tvec.empty()) {
            throw std::runtime_error("RANSAC returned failure");
        }

        int n_inliers = inliers.rows;
        if (n_inliers < gpnp_min_pts_) {
            throw std::runtime_error(
                "RANSAC inliers insufficient: " + std::to_string(n_inliers) +
                "/" + std::to_string(n_pts) + " < " + std::to_string(gpnp_min_pts_));
        }

        // ---- 步骤2: 使用内点子集进行细化 ----
        std::vector<cv::Point3d> obj_inliers;
        std::vector<cv::Point2d> img_inliers;
        obj_inliers.reserve(n_inliers);
        img_inliers.reserve(n_inliers);

        for (int i = 0; i < n_inliers; ++i) {
            int idx = inliers.at<int>(i);
            obj_inliers.push_back(object_pts[idx]);
            img_inliers.push_back(image_pts[idx]);
        }

        // 使用 RANSAC 结果作为初始猜测进行细化
        cv::Mat rvec_ref = rvec.clone();
        cv::Mat tvec_ref = tvec.clone();
        bool ref_ok = cv::solvePnP(
            obj_inliers, img_inliers, K_cv, cv::noArray(),
            rvec_ref, tvec_ref,
            true, // useExtrinsicGuess（从 RANSAC 热启动）
            cv::SOLVEPNP_ITERATIVE);

        if (!ref_ok) {
            throw std::runtime_error("Refinement solvePnP failed");
        }

        cv::Mat t_candidate = tvec_ref.reshape(1, 3);

        // ---- 步骤3: 有效性检查 ----
        double tz = t_candidate.at<double>(2);

        // 检查1: 相机必须在模板平面之前
        if (tz < 0.0) {
            throw std::runtime_error("t[2]=" + std::to_string(tz) + " < 0 (camera behind plane)");
        }

        // 检查2: 平移向量长度范围
        double t_norm = cv::norm(t_candidate);
        if (t_norm < 10.0 || t_norm > 20000.0) {
            throw std::runtime_error("t range anomaly |t|=" + std::to_string(t_norm));
        }

        // 检查3: 有限值
        if (!std::isfinite(t_norm)) {
            throw std::runtime_error("t contains NaN/Inf");
        }

        // 检查4: 旋转矩阵有效性
        cv::Mat R_cv;
        cv::Rodrigues(rvec_ref, R_cv);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(R_cv.at<double>(r, c))) {
                    throw std::runtime_error("R contains NaN/Inf");
                }
            }
        }

        // ---- 所有检查通过 ----
        // 转换为 Eigen
        result.R << R_cv.at<double>(0,0), R_cv.at<double>(0,1), R_cv.at<double>(0,2),
                    R_cv.at<double>(1,0), R_cv.at<double>(1,1), R_cv.at<double>(1,2),
                    R_cv.at<double>(2,0), R_cv.at<double>(2,1), R_cv.at<double>(2,2);

        result.t << t_candidate.at<double>(0),
                    t_candidate.at<double>(1),
                    t_candidate.at<double>(2);

        result.success = true;
        result.num_points = n_inliers;

        std::cout << "  [InitialPnP] Success | n=" << n_pts
                  << " | inliers=" << n_inliers
                  << " | t=[" << result.t(0) << "," << result.t(1) << "," << result.t(2) << "]"
                  << " | r=[" << result.R << "]"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  [InitialPnP] Failed (" << e.what()
                  << ") → 使用硬编码初始 t=["
                  << t_hardcoded(0) << "," << t_hardcoded(1) << "," << t_hardcoded(2) << "]"
                  << std::endl;
        result.R = Eigen::Matrix3d::Identity();
        result.t = t_hardcoded;
        result.success = false;
    }

    return result;
}

} // namespace gpnp