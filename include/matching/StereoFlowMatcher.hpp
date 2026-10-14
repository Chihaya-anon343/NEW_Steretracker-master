#pragma once

#include "common/Types.hpp"
#include <limits>
#include <memory>

namespace gpnp {

struct FlowPointDiagnostic {
    cv::Point2f initial_right{-1, -1};
    cv::Point2f tracked_right{-1, -1};
    double fb_error{std::numeric_limits<double>::infinity()};
    double epipolar_error{std::numeric_limits<double>::infinity()};
    double ncc{-1.0};
    bool coarse_attempted{false};
    int coarse_candidates{0};
    double coarse_best_ncc{-1}, coarse_second_ncc{-1};
    double planar_error_px{std::numeric_limits<double>::quiet_NaN()};
    bool accepted{false};
    std::string reason{"not_tracked"};
};

struct FlowMatchResult {
    std::vector<cv::Point2f> left_points, right_points;
    std::vector<int> left_indices;  // Original BC indices, never renumbered after rejection.
    std::vector<FlowPointDiagnostic> diagnostics;
    bool success{false};
    std::string failure_reason;
    int coarse_attempts{0};
    double pyramid_ms{0}, lk_ms{0}, coarse_ms{0}, check_ms{0}, total_ms{0};
    double crop_ms{0}, roi_attempt_ms{0};
    cv::Rect search_roi; // Full-image coordinates, including search and LK margins.
    bool roi_used{false}, full_frame_fallback{false};
    int pyramid_builds{0}; // Number of newly built L/R pyramid pairs.
    std::string roi_failure_reason;
    std::string planar_status{"disabled"};
    int planar_rejected{0};
    double planar_ms{0};
};

/// Robust geometry of one planar target; changes diagnostic decisions, never original IDs.
void filterPlanarGeometry(const std::vector<cv::Point2f>& left_points,
                          const StereoFlowConfig& config, bool horizontal, FlowMatchResult& result);

/// Sparse, same-frame stereo matching; both arguments are full-image CV_8UC1.
class StereoFlowMatcher {
public:
    StereoFlowMatcher(const StereoCameraParams& camera, const StereoFlowConfig& config);
    FlowMatchResult match(const cv::Mat& left, const cv::Mat& right,
                          const std::vector<cv::Point2f>& left_points,
                          const std::vector<cv::Point2f>& initial_right = {},
                          const cv::Rect& target_roi = {}) const;
private:
    friend class StereoFlowFrame;
    cv::Rect searchRegion(const cv::Mat& left, const std::vector<cv::Point2f>& points,
                          const std::vector<cv::Point2f>& initial, const cv::Rect& target_roi) const;
    FlowMatchResult matchFull(const cv::Mat& left, const cv::Mat& right,
                             const std::vector<cv::Point2f>& points,
                             const std::vector<cv::Point2f>& initial,
                             const std::vector<cv::Mat>* prepared_left = nullptr,
                             const std::vector<cv::Mat>* prepared_right = nullptr) const;
    StereoCameraParams camera_;
    StereoFlowConfig config_;
    Eigen::Matrix3d fundamental_;
    bool horizontal_{false};
};

/// Scoped to one immutable stereo pair; class0/class1 share image pyramids.
class StereoFlowFrame {
public:
    StereoFlowFrame(const StereoCameraParams& camera, const StereoFlowConfig& config,
                    const cv::Mat& left, const cv::Mat& right, const cv::Rect& target_union);
    FlowMatchResult match(const std::vector<cv::Point2f>& left_points,
                          const std::vector<cv::Point2f>& initial_right = {}, int min_matches = 0);
private:
    FlowMatchResult run(bool full, const std::vector<cv::Point2f>& points,
                        const std::vector<cv::Point2f>& initial, int min_matches);
    StereoFlowConfig config_;
    cv::Mat left_, right_, cropped_left_, cropped_right_;
    cv::Rect region_;
    std::unique_ptr<StereoFlowMatcher> full_matcher_, local_matcher_;
    std::vector<cv::Mat> full_left_pyramid_, full_right_pyramid_, local_left_pyramid_, local_right_pyramid_;
    double pending_crop_ms_{0};
};

struct FlowPoseCheck {
    bool success{false};
    double mean_left_px{0}, mean_right_px{0}, max_error_px{0};
    std::string reason;
};

/// Independent pixel/cheirality validation; does not accept solver flags as evidence.
FlowPoseCheck checkFlowPose(const PoseEstimate& pose, const PipelineResult& result,
                            const std::vector<Eigen::Vector3d>& template_points,
                            const StereoCameraParams& camera, bool stereo,
                            double mean_limit_px, double point_limit_px);

} // namespace gpnp
