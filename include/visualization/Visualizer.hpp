#pragma once

/**
 * @file Visualizer.hpp
 * @brief Debug visualization for stereo tracking.
 *
 * Module: visualization
 * Function: Generate 4 debug images per frame:
 *            1. Stereo matched: left+right stitched with matched-point lines
 *            2. Stereo projection: left image with projected right points
 *            3. Template match: left+template stitched with match lines
 *            4. Pose axes: 3-axis overlay on left image
 * Input:   Images, matched points, projected points, template matches, pose
 * Output:  PNG files written to disk
 * Dependencies: OpenCV highgui, imgproc, imgcodecs
 * Relations: Called by StereoTracker when visualize=true
 */

#include "common/Types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace gpnp {

class Visualizer {
public:
    /**
     * @brief Construct visualizer.
     * @param K           3×3 camera intrinsic matrix (for pose axes projection)
     * @param output_dir  Directory for output PNG files (default: current dir)
     */
    Visualizer(const Eigen::Matrix3d& K, const std::string& output_dir = ".");

    ~Visualizer() = default;

    // ========================================================================
    // 主可视化 API
    // ========================================================================

    /**
     * @brief Generate all 4 debug visualizations for a frame.
     *
     * @param left_color        Left camera color image
     * @param right_color       Right camera color image
     * @param template_color    Template color image (cached)
     * @param kp_left           All left keypoints
     * @param pts_left_used     Valid left points (after projection filter)
     * @param pts_right_used    Valid right points (after projection filter)
     * @param pts_right_proj    Projected right points on left image
     * @param good_matches      Template matches (DMatch list)
     * @param pts_left_match    Left matched points
     * @param pts_template_match Template matched points
     * @param disparity         Disparity values
     * @param prefix            Filename prefix (e.g., "_f1", "_f2")
     * @param R                 Camera rotation (template→camera)
     * @param t                 Camera translation (mm)
     * @param gpnp_success      Whether GPNP succeeded
     */
    void generateAll(const cv::Mat& left_color,
                     const cv::Mat& right_color,
                     const cv::Mat& template_color,
                     const std::vector<cv::KeyPoint>& kp_left,
                     const std::vector<cv::Point2f>& pts_left_used,
                     const std::vector<cv::Point2f>& pts_right_used,
                     const std::vector<cv::Point2f>& pts_right_proj,
                     const std::vector<cv::DMatch>& good_matches,
                     const std::vector<cv::Point2f>& pts_left_match,
                     const std::vector<cv::Point2f>& pts_template_match,
                     const std::vector<double>& disparity,
                     const std::string& prefix,
                     const Eigen::Matrix3d& R,
                     const Eigen::Vector3d& t,
                     bool gpnp_success);

private:
    Eigen::Matrix3d K_;          ///< 相机内参矩阵
    std::string output_dir_;

    // ---- 各子面板 ----

    /// 面板 1：左右拼接图，含匹配点连线
    void drawStereoMatched(const cv::Mat& left_color,
                           const cv::Mat& right_color,
                           const std::vector<cv::KeyPoint>& kp_left,
                           const std::vector<cv::Point2f>& pts_left_used,
                           const std::vector<cv::Point2f>& pts_right_used,
                           const std::vector<double>& disparity,
                           const std::string& prefix);

    /// 面板 2：左图叠加右图投影点
    void drawStereoProjection(const cv::Mat& left_color,
                              const std::vector<cv::Point2f>& pts_left_used,
                              const std::vector<cv::Point2f>& pts_right_proj,
                              const std::string& prefix);

    /// 面板 3：左图与模板拼接图，含匹配连线
    void drawTemplateMatch(const cv::Mat& left_color,
                           const cv::Mat& template_color,
                           const std::vector<cv::DMatch>& good_matches,
                           const std::vector<cv::Point2f>& pts_left_match,
                           const std::vector<cv::Point2f>& pts_template_match,
                           const std::string& prefix);

    /// 面板 4：左图叠加位姿坐标轴
    void drawPoseAxes(const cv::Mat& left_color,
                      const Eigen::Matrix3d& R,
                      const Eigen::Vector3d& t,
                      const std::string& prefix);
};

} // namespace gpnp
