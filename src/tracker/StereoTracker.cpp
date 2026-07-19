/**
 * @file StereoTracker.cpp
 * @brief StereoTracker 双 ROI 策略实现。
 */

#include "tracker/StereoTracker.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <random>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace gpnp {

// ============================================================================
// 构造与初始化 —— 预创建所有三种特征提取器，加载模板
// ============================================================================

StereoTracker::StereoTracker(const Eigen::Matrix3d& K,
                               const Eigen::Matrix3d& R_rl,
                               const Eigen::Vector3d& t_rl,
                               const std::string& template_path,
                               const TrackerConfig& config,
                               const BinaryCornerExtractor::Config& binary_cfg,
                               const std::string& binary_template_dir,
                               const TinyTargetExtractor::Config& tiny_cfg,
                               const std::string& tiny_template_dir)
    : camera_(makeStereoCameraParams(K, R_rl, t_rl))
    , config_(config)
    , initial_pnp_(config.gpnp_min_pts)
    , gpnp_solver_(camera_, config.gpnp_min_pts)
    , mad_filter_(3, 3.0)
    , visualizer_(nullptr)
{
    // ---- ① 预初始化 AKAZE 提取器（主策略，模板加载较慢）----
    akaze_extractor_ = std::make_unique<AkazeGpnpExtractor>(config.scale, config.lk_params);
    akaze_extractor_->initCamera(camera_);
    akaze_extractor_->setTemplateData(template_path,
                                      config.template_real_width_mm,
                                      config.template_real_height_mm);
    template_ = akaze_extractor_->templateData();

    // ---- ①+ 预初始化双ROI专用 AKAZE 提取器（独立 scale 参数）----
    LKParams dual_lk;
        dual_lk.winSize = cv::Size(15, 15);       // 小 ROI 使用更小的窗口
        dual_lk.maxLevel = 2;                      // 更少的金字塔层数
    dual_akaze_extractor_ = std::make_unique<AkazeGpnpExtractor>(
        config.dual_roi_akaze_scale, dual_lk);
    dual_akaze_extractor_->initCamera(camera_);
    dual_akaze_extractor_->setTemplateData(template_path,
                                            config.template_real_width_mm,
                                            config.template_real_height_mm);

    // ---- ② 预初始化 BinaryCorner 提取器（加载 24 个角度模板）----
    binary_extractor_ = std::make_unique<BinaryCornerExtractor>(binary_cfg, binary_template_dir);

    // ---- ③ 预初始化 TinyTarget 提取器（加载多角度模板）----
    tiny_extractor_ = std::make_unique<TinyTargetExtractor>(tiny_cfg, tiny_template_dir);

    // ---- ④ 从 config 读取策略选择阈值 ----
    akaze_min_area_ = config.akaze_min_area;
    tiny_max_area_  = config.tiny_max_area;

    // ---- ④+ 存储 ROI padding 值（用于非 AKAZE 策略时扩大 ROI）----
    binary_roi_pad_ = binary_cfg.roi_pad_pixels;
    tiny_roi_pad_   = tiny_cfg.roi_pad_pixels;

    // ---- ⑤ 默认策略链：AKAZE → BinaryCorner → TinyTarget ----
    extractor_ = akaze_extractor_.get();
    fallback_extractors_.push_back(binary_extractor_.get());
    fallback_extractors_.push_back(tiny_extractor_.get());

    std::cout << "[StereoTracker] Pre-initialized 3 extractors: AkazeGpnp, BinaryCorner, TinyTarget"
              << std::endl;
}

StereoTracker::~StereoTracker() = default;

// ============================================================================
// 策略链配置 —— 根据 ROI 面积选择活跃的主提取器和退化链（仅修改指针，零开销）
// ============================================================================

void StereoTracker::configureStrategyChain(int roi_area) {
    fallback_extractors_.clear();

    if (roi_area >= akaze_min_area_ || roi_area == 0) {
        // Large ROI or no detection → AKAZE → BinaryCorner → TinyTarget
        extractor_ = akaze_extractor_.get();
        fallback_extractors_.push_back(binary_extractor_.get());
        fallback_extractors_.push_back(tiny_extractor_.get());

        std::cout << "[StereoTracker] Strategy chain: AkazeGpnp → BinaryCorner → TinyTarget"
                  << " (roi_area=" << roi_area << ")" << std::endl;

    } else if (roi_area > tiny_max_area_) {
        // Medium ROI → BinaryCorner → TinyTarget
        extractor_ = binary_extractor_.get();
        fallback_extractors_.push_back(tiny_extractor_.get());

        std::cout << "[StereoTracker] Strategy chain: BinaryCorner → TinyTarget"
                  << " (roi_area=" << roi_area << ")" << std::endl;

    } else {
        // Small ROI → TinyTarget only (no further fallback)
        extractor_ = tiny_extractor_.get();
        // fallback_extractors_ remains empty

        std::cout << "[StereoTracker] Strategy chain: TinyTarget only"
                  << " (roi_area=" << roi_area << ")" << std::endl;
    }
}

// ============================================================================
// ROI padding —— 非 AKAZE 策略时扩大 ROI 给角点提取提供周围上下文
// ============================================================================

void StereoTracker::applyRoiPadding(RoiRect& rl, RoiRect& rr, int roi_area,
                                    int left_cols, int left_rows,
                                    int right_cols, int right_rows) const {
    // Only pad when using non-AKAZE strategies (i.e., when roi_area != 0 and < akaze_min_area)
    if (roi_area == 0 || roi_area >= akaze_min_area_)
        return;

    int pad = (roi_area <= tiny_max_area_) ? tiny_roi_pad_
             : (roi_area < akaze_min_area_) ? binary_roi_pad_
             : 0;
    if (pad <= 0) return;

    if (rl.valid()) {
        rl = RoiRect{std::max(0, rl.x - pad),
                     std::max(0, rl.y - pad),
                     std::min(left_cols - std::max(0, rl.x - pad), rl.width  + 2 * pad),
                     std::min(left_rows - std::max(0, rl.y - pad), rl.height + 2 * pad)};
    }
    if (rr.valid()) {
        rr = RoiRect{std::max(0, rr.x - pad),
                     std::max(0, rr.y - pad),
                     std::min(right_cols - std::max(0, rr.x - pad), rr.width  + 2 * pad),
                     std::min(right_rows - std::max(0, rr.y - pad), rr.height + 2 * pad)};
    }
}

// ============================================================================
// 退化辅助函数 —— runExtraction
// ============================================================================

bool StereoTracker::runExtraction(FeatureExtractor& ext,
                                  const cv::Mat& left_gray, const cv::Mat& right_gray,
                                  const cv::Mat& left_color, const cv::Mat& right_color,
                                  const cv::Point2d& left_offset, const cv::Point2d& right_offset,
                                  const cv::Mat& left_color_orig, const cv::Mat& right_color_orig,
                                  PipelineResult& result) {
    result = ext.extract(left_gray, right_gray, left_color, right_color);

    // Restore full-image coordinates
    {
        double lx = left_offset.x, ly = left_offset.y;
        double rx = right_offset.x, ry = right_offset.y;

        if (lx != 0.0 || ly != 0.0) {
            for (auto& kp : result.kp_left) {
                kp.pt.x += static_cast<float>(lx);
                kp.pt.y += static_cast<float>(ly);
            }
            auto offset_left = [lx, ly](std::vector<cv::Point2f>& arr) {
                for (auto& p : arr) { p.x += static_cast<float>(lx); p.y += static_cast<float>(ly); }
            };
            offset_left(result.pts_left_good);
            offset_left(result.pts_left_used);
            offset_left(result.pts_left_match);
            offset_left(result.pts_right_projected);
        }
        if (rx != 0.0 || ry != 0.0) {
            auto offset_right = [rx, ry](std::vector<cv::Point2f>& arr) {
                for (auto& p : arr) { p.x += static_cast<float>(rx); p.y += static_cast<float>(ry); }
            };
            offset_right(result.pts_right_good);
            offset_right(result.pts_right_used);
        }

        result.left_color = left_color_orig;
        result.right_color = right_color_orig;
        result.left_roi_offset_x = static_cast<int>(lx);
        result.left_roi_offset_y = static_cast<int>(ly);
        result.right_roi_offset_x = static_cast<int>(rx);
        result.right_roi_offset_y = static_cast<int>(ry);
    }

    // For non-AKAZE strategies, compute full-image disparity
    bool is_akaze_ext = (ext.name() == "AkazeGpnp");
    if (!is_akaze_ext && !result.pts_left_good.empty() && !result.pts_right_good.empty()) {
        int n = std::min(static_cast<int>(result.pts_left_good.size()),
                         static_cast<int>(result.pts_right_good.size()));
        result.disparity.resize(n);
        result.dx_filtered.resize(n);
        for (int i = 0; i < n; ++i) {
            double d = static_cast<double>(result.pts_left_good[i].x -
                                            result.pts_right_good[i].x);
            result.dx_filtered[i] = d;
            result.disparity[i] = -d;  // = right.x - left.x (match AKAZE convention)
        }
        if (n > 0) {
            std::cout << "  [" << ext.name() << "] Full-image stereo: " << n
                      << " pairs, median_disp=" << computeMedian(result.disparity) << " px"
                      << std::endl;
        }
    }

    // Determine if extraction succeeded
    if (is_akaze_ext) {
        return result.n_kp_left >= 4;  // AKAZE: need minimum keypoints
    } else {
        return !result.pts_left_match.empty();  // BinaryCorner/TinyTarget: need extracted corners
    }
}

// ============================================================================
// PnP Helper: AKAZE path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runAkazePnP(PipelineResult& result, bool is_first) {
    PoseEstimate pose;
    double gpnp_timing = 0.0;

    // ---- MAD disparity filter (mono mode: skip, no right-image data) ----
    bool is_mono = result.pts_right_good.empty();

    if (!is_mono) {
        auto t_filter = std::chrono::high_resolution_clock::now();
        MadFilterResult mad_res = mad_filter_.filter(
            result.pts_left_good, result.pts_right_good, result.idx_from_filtered);

        result.pts_left_good = std::move(mad_res.pts_left_filtered);
        result.pts_right_good = std::move(mad_res.pts_right_filtered);
        result.disparity = std::move(mad_res.disparity);
        result.dx_filtered = std::move(mad_res.dx_filtered);
        result.idx_from_filtered = std::move(mad_res.idx_from_filtered);

        auto t_filter_end = std::chrono::high_resolution_clock::now();
        result.timing["filter"] = std::chrono::duration<double, std::milli>(t_filter_end - t_filter).count();

        // ---- Sync pts_left_used/pts_right_used/pts_right_projected after MAD ----
        if (!mad_res.downgraded && !mad_res.filter_mask.empty()) {
            bool all_pass = true;
            for (bool m : mad_res.filter_mask) { if (!m) { all_pass = false; break; } }

            if (!all_pass && !result.valid_mask.empty()) {
                std::vector<int> valid_indices;
                for (size_t i = 0; i < result.valid_mask.size(); ++i)
                    if (result.valid_mask[i]) valid_indices.push_back(static_cast<int>(i));

                std::vector<bool> filter_subset(valid_indices.size());
                for (size_t i = 0; i < valid_indices.size(); ++i)
                    filter_subset[i] = mad_res.filter_mask[valid_indices[i]];

                auto apply_subset = [&](std::vector<cv::Point2f>& arr) {
                    if (arr.size() != filter_subset.size()) return;
                    std::vector<cv::Point2f> filtered;
                    for (size_t i = 0; i < filter_subset.size(); ++i)
                        if (filter_subset[i]) filtered.push_back(arr[i]);
                    arr = std::move(filtered);
                };
                apply_subset(result.pts_left_used);
                apply_subset(result.pts_right_used);
                apply_subset(result.pts_right_projected);
            }
        }
    } else {
        result.timing["filter"] = 0.0;
    }

    // ---- AKAZE uses its own template pts_3d if available ----
    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;

    // ---- Pose estimation ----
    if (is_first && config_.use_initial_pnp) {
        // First frame: try InitialPnP → GPNP
        MatchResult match_res;
        match_res.good_matches = result.good_matches;
        match_res.pts_left_match = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;
        PoseEstimate init_pose = initial_pnp_.solve(match_res, pnp_pts_3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        if (init_pose.success) {
            // Warm-start GPNP with InitialPnP result
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                // GPNP failed but InitialPnP succeeded → use InitialPnP (no degradation)
                std::cout << "  [AKAZE] GPNP failed, falling back to InitialPnP result" << std::endl;
                pose = init_pose;
                pose.success = true;
            }
        } else {
            // InitialPnP failed → try GPNP with default depth
            std::cout << "  [AKAZE] InitialPnP failed, trying GPNP with default depth" << std::endl;
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t_id(0, 0, 5000);
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
        }
    } else if (is_first && !config_.use_initial_pnp) {
        std::cout << "  [InitialPnP] Skipped (use_initial_pnp=false)" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_id(0, 0, 5000);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
    } else {
        // Subsequent frame: GPNP with previous frame's pose as warm-start
        const Eigen::Matrix3d* Rp = state_.has_cache ? &state_.R_prev : nullptr;
        const Eigen::Vector3d* tp = state_.has_cache ? &state_.t_prev : nullptr;
        pose = gpnp_solver_.solve(result, pnp_pts_3d, Rp, tp, gpnp_timing);
    }

    return {pose.success, pose};
}

// ============================================================================
// PnP Helper: BinaryCorner path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runBinaryCornerPnP(PipelineResult& result, bool is_first) {
    PoseEstimate pose;
    double gpnp_timing = 0.0;

    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;

    if (is_first && config_.use_initial_pnp) {
        // BinaryCorner first frame: try InitialPnP
        MatchResult match_res;
        match_res.good_matches       = result.good_matches;
        match_res.pts_left_match     = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;
        PoseEstimate init_pose = initial_pnp_.solve(match_res, pnp_pts_3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        if (init_pose.success) {
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                // GPNP failed but InitialPnP succeeded → use InitialPnP (no degradation)
                std::cout << "  [BinaryCorner] GPNP failed, falling back to InitialPnP result" << std::endl;
                pose = init_pose;
                pose.success = true;
            }
        } else {
            // InitialPnP failed → fallback to depth-from-disparity
            std::cout << "  [BinaryCorner] InitialPnP failed, estimating depth from disparity" << std::endl;
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            double depth_from_disp = 500.0;
            if (!result.disparity.empty()) {
                std::vector<double> abs_disp;
                for (double d : result.disparity) abs_disp.push_back(std::abs(d));
                double med_disp = computeMedian(std::move(abs_disp));
                if (med_disp > 1.0) {
                    depth_from_disp = camera_.focal_length * camera_.baseline / med_disp;
                    depth_from_disp = std::clamp(depth_from_disp, 50.0, 5000.0);
                    std::cout << "  [BinaryCorner] Depth from disparity: " << static_cast<int>(depth_from_disp)
                              << "mm (median_disp=" << static_cast<int>(med_disp) << "px)" << std::endl;
                }
            }
            Eigen::Vector3d t_id(0, 0, depth_from_disp);
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
        }
    } else if (is_first && !config_.use_initial_pnp) {
        std::cout << "  [InitialPnP] Skipped (use_initial_pnp=false)" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        double depth_from_disp = 500.0;
        if (!result.disparity.empty()) {
            std::vector<double> abs_disp;
            for (double d : result.disparity) abs_disp.push_back(std::abs(d));
            double med_disp = computeMedian(std::move(abs_disp));
            if (med_disp > 1.0) {
                depth_from_disp = camera_.focal_length * camera_.baseline / med_disp;
                depth_from_disp = std::clamp(depth_from_disp, 50.0, 5000.0);
            }
        }
        Eigen::Vector3d t_id(0, 0, depth_from_disp);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
    } else {
        // Subsequent frame
        const Eigen::Matrix3d* Rp = state_.has_cache ? &state_.R_prev : nullptr;
        const Eigen::Vector3d* tp = state_.has_cache ? &state_.t_prev : nullptr;
        pose = gpnp_solver_.solve(result, pnp_pts_3d, Rp, tp, gpnp_timing);
    }

    return {pose.success, pose};
}

// ============================================================================
// PnP Helper: TinyTarget path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runTinyTargetPnP(PipelineResult& result) {
    PoseEstimate pose;

    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;

    if (pnp_pts_3d.empty() || result.pts_left_match.size() < 4)
        return {false, pose};

    std::vector<cv::Point3d> obj_pts;
    for (const auto& p : pnp_pts_3d)
        obj_pts.emplace_back(p.x(), p.y(), p.z());

    std::vector<cv::Point2d> img_pts;
    for (const auto& p : result.pts_left_match)
        img_pts.emplace_back(p.x, p.y);

    cv::Mat K_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
        K_cv.at<double>(r, c) = camera_.K(r, c);

    cv::Mat rvec, tvec;
    bool pnp_ok = cv::solvePnP(obj_pts, img_pts, K_cv, cv::Mat(),
                                rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (pnp_ok) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            pose.R(r, c) = R_cv.at<double>(r, c);
        pose.t = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        pose.num_points = 4;
        pose.success = true;

        std::vector<cv::Point2d> projected;
        cv::projectPoints(obj_pts, rvec, tvec, K_cv, cv::Mat(), projected);
        double rpe_sum = 0.0;
        for (size_t i = 0; i < projected.size(); ++i)
            rpe_sum += cv::norm(projected[i] - img_pts[i]);
        std::cout << "  [TinyTarget] solvePnP OK  n_pts=4  RE="
                  << (rpe_sum / 4.0) << "px" << std::endl;
    } else {
        std::cerr << "  [TinyTarget] solvePnP FAILED" << std::endl;
    }
    result.timing["tiny_pnp"] = 0.0;

    return {pose.success, pose};
}

// ============================================================================
// Finalize pose — merge into PipelineResult and update tracking state
// ============================================================================

void StereoTracker::finalizePose(PipelineResult& result, const PoseEstimate& pose) {
    result.R = pose.R;
    result.t = pose.t;
    result.gpnp_success = pose.success;
    result.gpnp_n_pts = pose.num_points;

    if (pose.success) {
        cv::Mat R_cv(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = pose.R(r, c);
        cv::Mat rvec;
        cv::Rodrigues(R_cv, rvec);
        std::cout << "  Pose: rvec=[" << rvec.at<double>(0) << ", "
                  << rvec.at<double>(1) << ", " << rvec.at<double>(2) << "]"
                  << "  tvec=[" << pose.t(0) << ", " << pose.t(1) << ", "
                  << pose.t(2) << "] mm  n_pts=" << pose.num_points << std::endl;

        state_.R_prev = pose.R;
        state_.t_prev = pose.t;
        state_.has_cache = true;
    }
}

// ============================================================================
// PnP dispatch helper: route to correct PnP method based on strategy type
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::dispatchPnP(FeatureExtractor* ext,
                                                           PipelineResult& result,
                                                           bool is_first) {
    switch (ext->strategyType()) {
        case StrategyType::Akaze:
            return runAkazePnP(result, is_first);
        case StrategyType::BinaryCorner:
            return runBinaryCornerPnP(result, is_first);
        case StrategyType::TinyTarget:
            return runTinyTargetPnP(result);
        default:
            return {false, PoseEstimate{}};
    }
}

// ============================================================================
// 主处理入口 —— 动态退化链条
// ============================================================================

PipelineResult StereoTracker::process(const cv::Mat& left_img,
                                        const cv::Mat& right_img,
                                        bool visualize,
                                        const RoiGroup* left_group,
                                        const RoiGroup* right_group) {
    ++state_.frame_count;

    // ---- Load images ----
    auto [left_color, left_gray] = loadImage(left_img);
    auto [right_color, right_gray] = loadImage(right_img);
    if (left_gray.empty() || right_gray.empty())
        throw std::runtime_error("Cannot read input images");

    // ---- Save originals (for visualization & coordinate restore) ----
    cv::Mat left_color_orig = left_color.clone();
    cv::Mat right_color_orig = right_color.clone();

    // ---- Resolve RoiGroup ----
    RoiGroup left_grp  = left_group  ? *left_group  : RoiGroup{};
    RoiGroup right_grp = right_group ? *right_group : RoiGroup{};

    // ---- Dual-ROI path (new strategy placeholder) ----
    if (left_grp.is_dual && right_grp.is_dual) {
        return processDualRoi(left_img, right_img, left_grp, right_grp, visualize);
    }

    // ---- Single-ROI path (existing logic, unchanged) ----
    RoiRect roi_l = left_grp.primary;
    RoiRect roi_r = right_grp.primary;

    // ---- ROI validation & strategy selection ----
    roi_l = validateRoi(roi_l.valid() ? &roi_l : nullptr, left_gray.size(), "Left ROI");
    roi_r = validateRoi(roi_r.valid() ? &roi_r : nullptr, right_gray.size(), "Right ROI");

    // Automatically select extraction strategy chain from ROI area
    int roi_area = roi_l.valid() ? roi_l.width * roi_l.height : 0;
    configureStrategyChain(roi_area);

    // Expand ROI with padding for non-AKAZE extractors (corner context)
    applyRoiPadding(roi_l, roi_r, roi_area,
                    left_gray.cols, left_gray.rows,
                    right_gray.cols, right_gray.rows);

    // ---- Cropping (with possibly expanded ROI) ----
    cv::Point2d left_offset(0.0, 0.0), right_offset(0.0, 0.0);
    cv::Mat left_cropped  = left_gray;
    cv::Mat right_cropped = right_gray;
    cv::Mat left_color_cropped  = left_color;
    cv::Mat right_color_cropped = right_color;

    if (roi_l.valid()) {
        left_cropped  = left_gray(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_color_cropped = left_color(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_offset = cv::Point2d(static_cast<double>(roi_l.x), static_cast<double>(roi_l.y));
    }
    if (roi_r.valid()) {
        right_cropped = right_gray(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_color_cropped = right_color(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_offset = cv::Point2d(static_cast<double>(roi_r.x), static_cast<double>(roi_r.y));
    }

    bool is_first = !state_.has_cache;
    PipelineResult result;
    std::string winning_strategy;
    bool fallback_used = false;
    bool pose_ok = false;
    PoseEstimate final_pose;

    // ========================================================================
    // Build the degradation chain dynamically
    // ========================================================================

    // Collect all candidates: [primary, fallback[0], fallback[1], ...]
    std::vector<FeatureExtractor*> chain;
    chain.push_back(extractor_);
    for (auto* fb : fallback_extractors_)
        chain.push_back(fb);

    // Remove duplicates: if a fallback's type matches primary, skip it
    StrategyType primary_type = extractor_->strategyType();
    {
        auto it = std::remove_if(chain.begin() + 1, chain.end(),
            [primary_type](FeatureExtractor* e) { return e->strategyType() == primary_type; });
        chain.erase(it, chain.end());
    }

    for (size_t i = 0; i < chain.size(); ++i) {
        FeatureExtractor* ext = chain[i];
        bool is_primary = (i == 0);
        bool is_fb = !is_primary;

        std::string strategy_name = ext->name();

        if (is_fb) {
            fallback_used = true;
            std::string from = chain[i - 1]->name();
            std::cout << "[Degradation] " << from << " failed → " << strategy_name << std::endl;
        }

        bool extract_ok = runExtraction(*ext, left_cropped, right_cropped,
                                        left_color_cropped, right_color_cropped,
                                        left_offset, right_offset,
                                        left_color_orig, right_color_orig, result);

        if (!extract_ok) {
            if (is_primary) {
                std::cout << "[Degradation] " << strategy_name << " extraction failed"
                          << " (n_kp=" << result.n_kp_left << ")" << std::endl;
            } else {
                std::cout << "[Degradation] " << strategy_name << " extraction failed" << std::endl;
            }
            continue;
        }

        auto [ok, pose] = dispatchPnP(ext, result, is_first);
        if (ok) {
            pose_ok = true;
            final_pose = pose;
            winning_strategy = strategy_name;
            break;
        }
    }

    if (!pose_ok) {
        std::cerr << "[Degradation] All strategies failed for frame " << state_.frame_count << std::endl;
    }

    // ---- Finalize pose (if successful) ----
    if (pose_ok) {
        finalizePose(result, final_pose);
    }

    // ---- Visualization ----
    if (visualize && pose_ok) {
        std::string prefix = "_f" + std::to_string(state_.frame_count);

        if (winning_strategy == "AkazeGpnp") {
            if (!visualizer_) visualizer_ = std::make_unique<Visualizer>(camera_.K, output_dir_);
            cv::Mat tmpl_color = template_.color_image;
            if (tmpl_color.empty()) {
                cv::cvtColor(template_.gray_image.empty()
                    ? cv::Mat(200, 200, CV_8UC1, cv::Scalar(128))
                    : template_.gray_image,
                    tmpl_color, cv::COLOR_GRAY2BGR);
            }
            visualizer_->generateAll(
                left_color_orig, right_color_orig, tmpl_color,
                result.kp_left,
                result.pts_left_used, result.pts_right_used,
                result.pts_right_projected,
                result.good_matches,
                result.pts_left_match, result.pts_template_match,
                result.disparity, prefix,
                result.R, result.t, result.gpnp_success);
        } else {
            // BinaryCorner / TinyTarget visualization
            auto expandRect = [](const RoiRect& roi, const cv::Size& imgSz) -> cv::Rect {
                int cx = roi.x + roi.width / 2;
                int cy = roi.y + roi.height / 2;
                int ew = roi.width * 5;
                int eh = roi.height * 5;
                int x = std::max(0, cx - ew / 2);
                int y = std::max(0, cy - eh / 2);
                int w = std::min(ew, imgSz.width - x);
                int h = std::min(eh, imgSz.height - y);
                return cv::Rect(x, y, w, h);
            };
            cv::Rect expand_L = expandRect(roi_l, left_color_orig.size());
            cv::Rect expand_R = expandRect(roi_r, right_color_orig.size());

            cv::Mat view_L = left_color_orig(expand_L).clone();
            cv::Mat view_R = right_color_orig(expand_R).clone();

            float elx = static_cast<float>(expand_L.x), ely = static_cast<float>(expand_L.y);
            float erx = static_cast<float>(expand_R.x), ery = static_cast<float>(expand_R.y);
            auto toView_L = [&](const cv::Point2f& p) { return cv::Point2f(p.x - elx, p.y - ely); };
            auto toView_R = [&](const cv::Point2f& p) { return cv::Point2f(p.x - erx, p.y - ery); };

            auto proj = [&](const Eigen::Vector3d& P) -> cv::Point {
                if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
                Eigen::Vector2d uv = projectToImage(P, camera_.K);
                return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
            };

            const cv::Scalar CORNER_COLORS[] = {
                {0,0,255}, {0,255,0}, {0,255,255}, {255,0,0}, {255,0,255},
                {255,255,0}, {128,0,255}, {0,128,255}, {255,128,0}, {128,255,0}
            };

            const auto& pnp_pts_3d = template_.pts_3d;
            bool is_tiny = (winning_strategy == "TinyTarget");

            // --- Panel 0: 二值图像（左 | 右），角点叠加 (BinaryCorner only) ---
            if (!is_tiny) {
                // Use pre-initialized extractor for debug state
                auto* bce = binary_extractor_.get();
                if (bce) {
                    cv::Mat bl_bgr, br_bgr;
                    cv::cvtColor(bce->lastLeftBinary(),  bl_bgr, cv::COLOR_GRAY2BGR);
                    cv::cvtColor(bce->lastRightBinary(), br_bgr, cv::COLOR_GRAY2BGR);
                    float lx_roi = static_cast<float>(left_offset.x);
                    float ly_roi = static_cast<float>(left_offset.y);
                    float rx_roi = static_cast<float>(right_offset.x);
                    float ry_roi = static_cast<float>(right_offset.y);
                    for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                        cv::Point p(static_cast<int>(result.pts_left_match[i].x - lx_roi),
                                     static_cast<int>(result.pts_left_match[i].y - ly_roi));
                        cv::circle(bl_bgr, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                    for (size_t i = 0; i < result.pts_right_good.size(); ++i) {
                        cv::Point p(static_cast<int>(result.pts_right_good[i].x - rx_roi),
                                     static_cast<int>(result.pts_right_good[i].y - ry_roi));
                        cv::circle(br_bgr, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                    cv::Mat p0;
                    cv::hconcat(bl_bgr, br_bgr, p0);
                    cv::imwrite(output_dir_ + "/binary_corner_binary" + prefix + ".png", p0);
                }
            }

            // --- Panel 0b: 旋转回正二值图 (BinaryCorner only) ---
            if (!is_tiny) {
                auto* bce = binary_extractor_.get();
                if (bce && !bce->lastUprightBinary().empty()) {
                    cv::Mat up;
                    cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                    cv::imwrite(output_dir_ + "/binary_corner_upright" + prefix + ".png", up);
                }
            }

            // --- Panel 1: 3D axes on expanded left view ---
            {
                cv::Mat p1 = view_L.clone();
                double axis_len = 100.0;
                Eigen::Vector3d o  = final_pose.R * Eigen::Vector3d(0,0,0) + final_pose.t;
                Eigen::Vector3d ax = final_pose.R * Eigen::Vector3d(axis_len,0,0) + final_pose.t;
                Eigen::Vector3d ay = final_pose.R * Eigen::Vector3d(0,axis_len,0) + final_pose.t;
                Eigen::Vector3d az = final_pose.R * Eigen::Vector3d(0,0,axis_len) + final_pose.t;
                cv::Point o_p = proj(o);  o_p.x -= expand_L.x; o_p.y -= expand_L.y;
                cv::Point ax_p = proj(ax); ax_p.x -= expand_L.x; ax_p.y -= expand_L.y;
                cv::Point ay_p = proj(ay); ay_p.x -= expand_L.x; ay_p.y -= expand_L.y;
                cv::Point az_p = proj(az); az_p.x -= expand_L.x; az_p.y -= expand_L.y;
                cv::line(p1, o_p, ax_p, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                cv::line(p1, o_p, ay_p, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                cv::line(p1, o_p, az_p, cv::Scalar(255,0,0), 2, cv::LINE_AA);
                cv::imwrite(output_dir_ + "/binary_corner_axes" + prefix + ".png", p1);
            }

            // --- Panel 2: template corner correspondence (BinaryCorner only) ---
            if (!is_tiny) {
                cv::Mat p2_l = view_L.clone();
                for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                    cv::Point2f pv = toView_L(result.pts_left_match[i]);
                    cv::circle(p2_l, cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                               1, CORNER_COLORS[i % 10], -1);
                }
                auto* bce = binary_extractor_.get();
                const TemplateData* matched_tmpl = bce ? bce->lastMatchedTemplate() : nullptr;
                cv::Mat p2_tmpl;
                if (matched_tmpl) {
                    cv::cvtColor(matched_tmpl->image, p2_tmpl, cv::COLOR_GRAY2BGR);
                } else {
                    p2_tmpl = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128,128,128));
                }
                cv::resize(p2_tmpl, p2_tmpl, p2_l.size(), 0, 0, cv::INTER_NEAREST);
                if (matched_tmpl) {
                    double dsx = static_cast<double>(p2_tmpl.cols) / matched_tmpl->image.cols;
                    double dsy = static_cast<double>(p2_tmpl.rows) / matched_tmpl->image.rows;
                    for (size_t i = 0; i < matched_tmpl->corners.size(); ++i) {
                        cv::Point p(static_cast<int>(matched_tmpl->corners[i].x * dsx),
                                    static_cast<int>(matched_tmpl->corners[i].y * dsy));
                        cv::circle(p2_tmpl, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                }
                cv::Mat p2;
                cv::hconcat(p2_l, p2_tmpl, p2);
                cv::imwrite(output_dir_ + "/binary_corner_template" + prefix + ".png", p2);
            }

            // --- Panel 3: Left-right expanded view corner correspondence ---
            {
                cv::Mat p3_l = view_L.clone(), p3_r = view_R.clone();
                for (size_t i = 0; i < result.pts_left_match.size() &&
                                i < result.pts_right_good.size(); ++i) {
                    cv::Point2f pvl = toView_L(result.pts_left_match[i]);
                    cv::Point2f pvr = toView_R(result.pts_right_good[i]);
                    cv::circle(p3_l, cv::Point(static_cast<int>(pvl.x), static_cast<int>(pvl.y)),
                               1, CORNER_COLORS[i % 10], -1);
                    cv::circle(p3_r, cv::Point(static_cast<int>(pvr.x), static_cast<int>(pvr.y)),
                               1, CORNER_COLORS[i % 10], -1);
                }
                cv::Mat p3;
                cv::hconcat(p3_l, p3_r, p3);
                cv::imwrite(output_dir_ + "/binary_corner_stereo" + prefix + ".png", p3);
            }

            // --- Panel 4: Reprojection on expanded left view ---
            if (!pnp_pts_3d.empty()) {
                cv::Mat p4 = view_L.clone();

                int max_qidx = 0;
                for (const auto& m : result.good_matches)
                    max_qidx = std::max(max_qidx, m.queryIdx);
                std::vector<int> train_lut(max_qidx + 1, -1);
                for (const auto& m : result.good_matches)
                    train_lut[m.queryIdx] = m.trainIdx;

                std::vector<int> obs_to_3d(result.pts_left_good.size(), -1);
                std::vector<int> used_3d_indices;
                for (size_t j = 0; j < result.pts_left_good.size() &&
                                    j < result.idx_from_filtered.size(); ++j) {
                    int kp_idx = result.idx_from_filtered[j];
                    if (kp_idx >= 0 && kp_idx < static_cast<int>(train_lut.size())) {
                        int tr_idx = train_lut[kp_idx];
                        if (tr_idx >= 0 && tr_idx < static_cast<int>(pnp_pts_3d.size())) {
                            obs_to_3d[j] = tr_idx;
                            used_3d_indices.push_back(tr_idx);
                        }
                    }
                }

                std::vector<cv::Point3d> obj_pts;
                obj_pts.reserve(used_3d_indices.size());
                for (int idx : used_3d_indices)
                    obj_pts.emplace_back(pnp_pts_3d[idx].x(), pnp_pts_3d[idx].y(), pnp_pts_3d[idx].z());

                std::vector<cv::Point2d> projected;
                if (!obj_pts.empty()) {
                    cv::Mat rvec4, tvec4(3, 1, CV_64F);
                    tvec4.at<double>(0) = final_pose.t(0);
                    tvec4.at<double>(1) = final_pose.t(1);
                    tvec4.at<double>(2) = final_pose.t(2);
                    cv::Mat R4(3, 3, CV_64F);
                    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                        R4.at<double>(r, c) = final_pose.R(r, c);
                    cv::Rodrigues(R4, rvec4);
                    cv::Mat K4(3, 3, CV_64F);
                    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                        K4.at<double>(r, c) = camera_.K(r, c);
                    cv::projectPoints(obj_pts, rvec4, tvec4, K4, cv::Mat(), projected);
                }

                std::unordered_map<int, size_t> proj_map;
                for (size_t k = 0; k < used_3d_indices.size(); ++k)
                    proj_map[used_3d_indices[k]] = k;

                for (size_t j = 0; j < result.pts_left_good.size(); ++j) {
                    int tr_idx = obs_to_3d[j];
                    if (tr_idx < 0) continue;
                    auto it = proj_map.find(tr_idx);
                    if (it == proj_map.end() || it->second >= projected.size()) continue;
                    cv::Point2f obs_v = toView_L(result.pts_left_good[j]);
                    cv::Point pd(static_cast<int>(projected[it->second].x - expand_L.x),
                                 static_cast<int>(projected[it->second].y - expand_L.y));
                    cv::Point pm(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    cv::circle(p4, pd, 1, cv::Scalar(0, 255, 0), -1);
                    cv::circle(p4, pm, 1, cv::Scalar(0, 0, 255), 1);
                    cv::line(p4, pd, pm, cv::Scalar(0, 255, 255), 1);
                }
                cv::imwrite(output_dir_ + "/binary_corner_reproj" + prefix + ".png", p4);
            }

            // --- Panel 5: Stereo projection (disparity-based) ---
            if (!result.pts_left_good.empty() && !result.pts_right_good.empty()) {
                cv::Mat p5 = view_L.clone();
                const double fx = camera_.focal_length;
                const double fy = camera_.K(1,1);
                const double cx = camera_.K(0,2);
                const double cy = camera_.K(1,2);
                const double b  = camera_.baseline;
                const Eigen::Matrix3d& R_rl = camera_.R_rl;
                const Eigen::Vector3d& t_rl = camera_.t_rl;

                int n_stereo_proj = std::min(static_cast<int>(result.pts_left_good.size()),
                                              static_cast<int>(result.pts_right_good.size()));

                for (int i = 0; i < n_stereo_proj; ++i) {
                    double uL = result.pts_left_good[i].x;
                    double vL = result.pts_left_good[i].y;
                    double uR = result.pts_right_good[i].x;
                    double vR = result.pts_right_good[i].y;

                    double disp = uL - uR;
                    double abs_disp = std::abs(disp);
                    if (abs_disp < 0.5) continue;

                    double depth = fx * b / abs_disp;
                    if (depth <= 0.0 || depth > 100000.0) continue;

                    double rx = (uL - cx) / fx;
                    double ry = (vL - cy) / fy;
                    double rn = std::sqrt(rx*rx + ry*ry + 1.0);
                    rx /= rn; ry /= rn;
                    double rz = 1.0 / rn;

                    double Px = rx * depth;
                    double Py = ry * depth;
                    double Pz = rz * depth;

                    double dx = Px - t_rl(0);
                    double dy = Py - t_rl(1);
                    double dz = Pz - t_rl(2);
                    double PRx = dx*R_rl(0,0) + dy*R_rl(1,0) + dz*R_rl(2,0);
                    double PRy = dx*R_rl(0,1) + dy*R_rl(1,1) + dz*R_rl(2,1);
                    double PRz = dx*R_rl(0,2) + dy*R_rl(1,2) + dz*R_rl(2,2);

                    if (std::abs(PRz) < 1e-6) continue;

                    double proj_x = fx * PRx / PRz + cx;
                    double proj_y = fy * PRy / PRz + cy;

                    cv::Point2f obs_v = toView_L(result.pts_left_good[i]);
                    cv::Point pt_L(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    cv::Point pt_R(static_cast<int>(proj_x - expand_L.x),
                                   static_cast<int>(proj_y - expand_L.y));

                    cv::circle(p5, pt_L, 4, cv::Scalar(255, 0, 0), -1);
                    cv::drawMarker(p5, pt_R, cv::Scalar(0, 0, 255),
                                   cv::MARKER_TILTED_CROSS, 10, 2);
                    cv::line(p5, pt_L, pt_R, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }
                cv::imwrite(output_dir_ + "/binary_corner_stereo_proj" + prefix + ".png", p5);
            }
        }
    }

    // ---- Logging ----
    result.is_first_frame = is_first;
    result.n_matched = static_cast<int>(result.pts_left_good.size());
    result.n_projected = static_cast<int>(result.pts_right_projected.size());
    addLogEntry(result, is_first, fallback_used);
    return result;
}

PipelineResult StereoTracker::process(const std::string& left_path,
                                        const std::string& right_path,
                                        bool visualize,
                                        const RoiGroup* left_group,
                                        const RoiGroup* right_group) {
    cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
    return process(left, right, visualize, left_group, right_group);
}

// ============================================================================
// Dual-ROI Strategy: BinaryCorner (class 0 edges) + AKAZE (class 1 center)
// ============================================================================

void StereoTracker::prepareDualBcTemplate() {
    if (dual_bc_template_ready_) return;

    // Get AKAZE template grayscale image
    const cv::Mat& tmpl_img = akaze_extractor_->templateData().gray_image;
    if (tmpl_img.empty()) {
        std::cerr << "[DualRoi] AKAZE template image empty, cannot prepare BC template" << std::endl;
        dual_bc_template_ready_ = true;  // mark ready to avoid retrying
        return;
    }

    int tw = tmpl_img.cols, th = tmpl_img.rows;

    // 1. Otsu binarization
    cv::Mat binary;
    cv::threshold(tmpl_img, binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    // 2. Find largest contour
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        std::cerr << "[DualRoi] No contours found in AKAZE template binary" << std::endl;
        dual_bc_template_ready_ = true;
        return;
    }

    auto* largest = &contours[0];
    for (auto& c : contours) {
        if (cv::contourArea(c) > cv::contourArea(*largest)) largest = &c;
    }

    // 3. approxPolyDP with binary search for exactly 10 corners
    int target_n = binary_extractor_->lastCornersBeforeReorder().empty()
        ? 10
        : static_cast<int>(binary_extractor_->lastCornersBeforeReorder().size());
    if (target_n < 3) target_n = 10;

    double peri = cv::arcLength(*largest, true);
    double lo = 0.0, hi = peri * 0.1;
    std::vector<cv::Point2f> corners;
    for (int iter = 0; iter < 25; ++iter) {
        double mid = (lo + hi) * 0.5;
        std::vector<cv::Point2f> approx;
        cv::approxPolyDP(*largest, approx, mid, true);
        int sz = static_cast<int>(approx.size());
        if (sz < target_n) hi = mid;
        else if (sz > target_n) lo = mid;
        else { corners = approx; break; }
    }
    if (corners.size() != static_cast<size_t>(target_n)) {
        double eps = (lo + hi) * 0.5;
        cv::approxPolyDP(*largest, corners, eps, true);
    }

    if (corners.empty()) {
        std::cerr << "[DualRoi] Failed to extract corners from AKAZE template" << std::endl;
        dual_bc_template_ready_ = true;
        return;
    }

    // 4. Geometric ordering (CCW from reference angle 0 = up direction)
    cv::Point2f center(tw / 2.0f, th / 2.0f);
    auto order = BinaryCornerExtractor::reorderByGeometry(corners, center, 0.0, -1.0);
    dual_bc_tmpl_corners_.reserve(order.size());
    for (int idx : order) {
        dual_bc_tmpl_corners_.push_back(corners[idx]);
    }

    // 5. Compute 3D coordinates using AKAZE template physical dimensions
    double real_w = config_.template_real_width_mm;
    double real_h = config_.template_real_height_mm;
    dual_bc_tmpl_pts3d_.reserve(dual_bc_tmpl_corners_.size());
    for (const auto& c : dual_bc_tmpl_corners_) {
        dual_bc_tmpl_pts3d_.emplace_back(
            c.x / static_cast<double>(tw) * real_w,
            c.y / static_cast<double>(th) * real_h,
            0.0);
    }

    dual_bc_template_ready_ = true;
    std::cout << "[DualRoi] BC template prepared: " << dual_bc_tmpl_corners_.size()
              << " corners on AKAZE template (" << tw << "x" << th << ")"
              << "  real_size=" << real_w << "x" << real_h << "mm"
              << std::endl;
}

PipelineResult StereoTracker::processDualRoi(const cv::Mat& left_img,
                                               const cv::Mat& right_img,
                                               const RoiGroup& left_group,
                                               const RoiGroup& right_group,
                                               bool visualize) {
    // 0. Ensure template preprocessing is done
    prepareDualBcTemplate();

    // ---- Load images ----
    auto [left_color, left_gray] = loadImage(left_img);
    auto [right_color, right_gray] = loadImage(right_img);
    if (left_gray.empty() || right_gray.empty()) {
        PipelineResult empty;
        empty.is_first_frame = !state_.has_cache;
        return empty;
    }

    cv::Mat left_color_orig = left_color.clone();
    cv::Mat right_color_orig = right_color.clone();

    int pad = config_.dual_roi_secondary_expand;

    bool is_first = !state_.has_cache;

    // 1. Expand secondary ROI
    auto expandRoi = [](const RoiRect& roi, int p, int img_w, int img_h) -> RoiRect {
        int x = std::max(0, roi.x - p);
        int y = std::max(0, roi.y - p);
        int w = std::min(img_w - x, roi.width  + 2 * p);
        int h = std::min(img_h - y, roi.height + 2 * p);
        return RoiRect{x, y, w, h};
    };

    RoiRect left_pri  = left_group.primary;
    RoiRect right_pri = right_group.primary;
    RoiRect left_sec  = expandRoi(left_group.secondary,  pad, left_img.cols,  left_img.rows);
    RoiRect right_sec = expandRoi(right_group.secondary, pad, right_img.cols, right_img.rows);

    // Offset from class 1 ROI to class 0 ROI (same for right since both expand uniformly)
    cv::Point2d sec_to_pri_offset(
        static_cast<double>(left_sec.x - left_pri.x),
        static_cast<double>(left_sec.y - left_pri.y));
    cv::Point2d right_sec_to_pri_offset(
        static_cast<double>(right_sec.x - right_pri.x),
        static_cast<double>(right_sec.y - right_pri.y));

    std::cout << "[DualRoi] pad=" << pad
              << "  primary=" << left_pri.width << "x" << left_pri.height
              << "  secondary(raw)=" << left_group.secondary.width << "x" << left_group.secondary.height
              << "  secondary(expanded)=" << left_sec.width << "x" << left_sec.height
              << "  offset=(" << sec_to_pri_offset.x << "," << sec_to_pri_offset.y << ")"
              << std::endl;

    // 2. Crop images
    cv::Mat left_c0_gray   = left_gray(cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
    cv::Mat right_c0_gray  = right_gray(cv::Rect(right_pri.x, right_pri.y, right_pri.width, right_pri.height)).clone();
    cv::Mat left_c0_color  = left_color(cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
    cv::Mat right_c0_color = right_color(cv::Rect(right_pri.x, right_pri.y, right_pri.width, right_pri.height)).clone();

    cv::Mat left_c1_gray   = left_gray(cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height)).clone();
    cv::Mat right_c1_gray  = right_gray(cv::Rect(right_sec.x, right_sec.y, right_sec.width, right_sec.height)).clone();
    cv::Mat left_c1_color  = left_color(cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height)).clone();
    cv::Mat right_c1_color = right_color(cv::Rect(right_sec.x, right_sec.y, right_sec.width, right_sec.height)).clone();

    // 3. BinaryCorner extraction on class 0 (edge corners, with rotation alignment)
    PipelineResult result_bc = binary_extractor_->extract(
        left_c0_gray, right_c0_gray, left_c0_color, right_c0_color);

    int n_bc = static_cast<int>(result_bc.pts_left_match.size());
    std::cout << "[DualRoi] BinaryCorner on class 0: " << n_bc << " corners" << std::endl;

    // 4. AKAZE extraction on class 1 (center texture features, dual-ROI params)
    PipelineResult result_ak = dual_akaze_extractor_->extract(
        left_c1_gray, right_c1_gray, left_c1_color, right_c1_color);

    int m_ak_match = static_cast<int>(result_ak.pts_left_match.size());
    std::cout << "[DualRoi] AKAZE on class 1: " << m_ak_match << " template matches"
              << " (kp=" << result_ak.n_kp_left
              << ", flow=" << result_ak.pts_left_good.size() << ")"
              << std::endl;

    // 5. Coordinate transform: class-1-ROI-local → class-0-ROI-local
    auto offsetPoints = [](std::vector<cv::Point2f>& pts, double dx, double dy) {
        float fx = static_cast<float>(dx), fy = static_cast<float>(dy);
        for (auto& p : pts) { p.x += fx; p.y += fy; }
    };
    offsetPoints(result_ak.pts_left_match,   sec_to_pri_offset.x, sec_to_pri_offset.y);
    offsetPoints(result_ak.pts_left_good,    sec_to_pri_offset.x, sec_to_pri_offset.y);
    offsetPoints(result_ak.pts_right_good,   right_sec_to_pri_offset.x, right_sec_to_pri_offset.y);
    offsetPoints(result_ak.pts_left_used,    sec_to_pri_offset.x, sec_to_pri_offset.y);
    offsetPoints(result_ak.pts_right_used,   right_sec_to_pri_offset.x, right_sec_to_pri_offset.y);
    for (auto& kp : result_ak.kp_left) {
        kp.pt.x += static_cast<float>(sec_to_pri_offset.x);
        kp.pt.y += static_cast<float>(sec_to_pri_offset.y);
    }

    // 6. Merge corner sets
    // Use pts_left_match from both extractors (these have template correspondences)
    std::vector<cv::Point2f> merged_pts_left, merged_pts_right, merged_pts_template;
    std::vector<cv::KeyPoint> merged_kp_left;
    std::vector<Eigen::Vector3d> merged_pts3d;

    // --- BC contribution (N corners, 1:1 left-right-template) ---
    int n_bc_right = static_cast<int>(result_bc.pts_right_good.size());
    int n_bc_use = std::min(n_bc, n_bc_right);

    for (int i = 0; i < n_bc_use; ++i) {
        merged_pts_left.push_back(result_bc.pts_left_match[i]);
        merged_pts_right.push_back(result_bc.pts_right_good[i]);
        merged_kp_left.emplace_back(result_bc.pts_left_match[i], 1.0f);
    }
    // BC template corners scaled to class-0-ROI (already in BC's result)
    for (int i = 0; i < n_bc_use && i < static_cast<int>(result_bc.pts_template_match.size()); ++i) {
        merged_pts_template.push_back(result_bc.pts_template_match[i]);
    }
    // BC 3D points from AKAZE-template-derived corners
    for (int i = 0; i < n_bc_use && i < static_cast<int>(dual_bc_tmpl_pts3d_.size()); ++i) {
        merged_pts3d.push_back(dual_bc_tmpl_pts3d_[i]);
    }

    int bc_total = n_bc_use;
    int bc_3d_added = std::min(n_bc_use, static_cast<int>(dual_bc_tmpl_pts3d_.size()));

    // --- AK contribution (M matched feature points) ---
    int m_ak = static_cast<int>(result_ak.pts_left_match.size());
    const auto& ak_pts3d = dual_akaze_extractor_->templateData().pts_3d;

    // For AK, pts_left_match has template match; try to find right-image match
    // Use optical-flow-tracked pts_right_good that correspond to kp_left (with template match)
    std::vector<int> ak_good_indices; // indices in kp_left that have both template AND stereo
    if (!result_ak.good_matches.empty() && !result_ak.pts_left_good.empty()) {
        // Build set of kp_left indices that passed optical flow
        std::unordered_set<int> flow_indices;
        for (int idx : result_ak.idx_from_filtered) {
            flow_indices.insert(idx);
        }
        // Filter good_matches to those with stereo
        for (const auto& m : result_ak.good_matches) {
            if (flow_indices.count(m.queryIdx) > 0) {
                ak_good_indices.push_back(static_cast<int>(m.queryIdx));
            }
        }
    }

    if (ak_good_indices.empty()) {
        // Fallback: use pts_left_match as-is, skip right-image points for AK part
        std::cout << "  [DualRoi] AK: no points with both stereo+template, using template-only"
                  << std::endl;
        for (int i = 0; i < m_ak; ++i) {
            merged_pts_left.push_back(result_ak.pts_left_match[i]);
            merged_kp_left.emplace_back(result_ak.pts_left_match[i], 1.0f);
            // No right-image point available → use zero offset as placeholder
            merged_pts_right.emplace_back(
                result_ak.pts_left_match[i].x,
                result_ak.pts_left_match[i].y);
        }
        // AK template match points (in template image coords)
        for (int i = 0; i < m_ak && i < static_cast<int>(result_ak.pts_template_match.size()); ++i) {
            merged_pts_template.push_back(result_ak.pts_template_match[i]);
        }
        // AK 3D points
        for (int i = 0; i < m_ak && i < static_cast<int>(ak_pts3d.size()); ++i) {
            merged_pts3d.push_back(ak_pts3d[i]);
        }
    } else {
        // AK contribution: build lookup from kp_left index → stereo point index
        std::unordered_map<int, int> kp_to_stereo; // kp_left idx → pts_left_good idx
        for (int j = 0; j < static_cast<int>(result_ak.idx_from_filtered.size()); ++j) {
            kp_to_stereo[result_ak.idx_from_filtered[j]] = j;
        }

        for (int i = 0; i < m_ak; ++i) {
            int match_query = result_ak.good_matches[i].queryIdx; // kp_left index
            merged_pts_left.push_back(result_ak.pts_left_match[i]);
            merged_kp_left.emplace_back(result_ak.pts_left_match[i], 1.0f);

            // Find stereo right point
            auto it = kp_to_stereo.find(match_query);
            if (it != kp_to_stereo.end() && it->second < static_cast<int>(result_ak.pts_right_good.size())) {
                merged_pts_right.push_back(result_ak.pts_right_good[it->second]);
            } else {
                merged_pts_right.emplace_back(
                    result_ak.pts_left_match[i].x,
                    result_ak.pts_left_match[i].y);
            }
        }
        // AK template + 3D points (by DMatch trainIdx)
        for (int i = 0; i < m_ak; ++i) {
            int train_idx = result_ak.good_matches[i].trainIdx;
            if (i < static_cast<int>(result_ak.pts_template_match.size())) {
                merged_pts_template.push_back(result_ak.pts_template_match[i]);
            }
            if (train_idx >= 0 && train_idx < static_cast<int>(ak_pts3d.size())) {
                merged_pts3d.push_back(ak_pts3d[train_idx]);
            }
        }
    }

    int total_pts = static_cast<int>(merged_pts_left.size());
    int total_right = static_cast<int>(merged_pts_right.size());
    int total_3d = static_cast<int>(merged_pts3d.size());

    // Sync counts: GPNP needs equal numbers
    int total_use = std::min({total_pts, total_right, total_3d});
    merged_pts_left.resize(total_use);
    merged_pts_right.resize(total_use);
    merged_pts3d.resize(total_use);
    merged_kp_left.resize(total_use);

    // Build 1:1 good_matches + idx_from_filtered
    std::vector<cv::DMatch> merged_matches(total_use);
    std::vector<int> merged_idx(total_use);
    for (int i = 0; i < total_use; ++i) {
        merged_matches[i] = cv::DMatch(i, i, 0.0f); // queryIdx=i → trainIdx=i
        merged_idx[i] = i;
    }

    std::cout << "[DualRoi] Merged: " << total_use << " total (BC=" << bc_total
              << " [3d=" << bc_3d_added << "], AK=" << m_ak_match << ")"
              << "  pts3d=" << total_3d << std::endl;

    if (total_use < 4) {
        std::cerr << "[DualRoi] Too few merged points (" << total_use << "), aborting" << std::endl;
        PipelineResult empty;
        empty.is_first_frame = is_first;
        empty.gpnp_success = false;
        return empty;
    }

    // 7. Build PipelineResult
    PipelineResult result;
    result.kp_left         = std::move(merged_kp_left);
    result.n_kp_left       = total_use;
    result.pts_left_match  = merged_pts_left;
    result.pts_left_good   = merged_pts_left;
    result.pts_right_good  = merged_pts_right;
    result.pts_left_used   = merged_pts_left;
    result.pts_right_used  = merged_pts_right;
    result.pts_template_match = merged_pts_template;
    result.good_matches    = std::move(merged_matches);
    result.idx_from_filtered = std::move(merged_idx);
    result.left_color      = left_color_orig;
    result.right_color     = right_color_orig;
    result.is_first_frame  = is_first;

    // Build valid_mask (all true, identity mapping)
    result.valid_mask.resize(total_use, true);

    // 8. Pose estimation
    PoseEstimate pose;
    double gpnp_timing = 0.0;
    auto t_pnp_start = std::chrono::high_resolution_clock::now();

    if (is_first && config_.use_initial_pnp) {
        // First frame: InitialPnP → GPNP
        MatchResult match_res;
        match_res.good_matches       = result.good_matches;
        match_res.pts_left_match     = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;

        PoseEstimate init_pose = initial_pnp_.solve(match_res, merged_pts3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        if (init_pose.success) {
            std::cout << "  [DualRoi] InitialPnP OK, warm-starting GPNP" << std::endl;
            pose = gpnp_solver_.solve(result, merged_pts3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                std::cout << "  [DualRoi] GPNP failed, using InitialPnP result" << std::endl;
                pose = init_pose;
                pose.success = true;
            }
        } else {
            std::cout << "  [DualRoi] InitialPnP failed, trying GPNP with default depth" << std::endl;
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t_id(0, 0, 5000);
            pose = gpnp_solver_.solve(result, merged_pts3d, &R_id, &t_id, gpnp_timing);
        }
    } else if (is_first && !config_.use_initial_pnp) {
        std::cout << "  [DualRoi] InitialPnP skipped" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_id(0, 0, 5000);
        pose = gpnp_solver_.solve(result, merged_pts3d, &R_id, &t_id, gpnp_timing);
    } else {
        // Subsequent frame: warm-start from previous pose
        const Eigen::Matrix3d* Rp = state_.has_cache ? &state_.R_prev : nullptr;
        const Eigen::Vector3d* tp = state_.has_cache ? &state_.t_prev : nullptr;
        pose = gpnp_solver_.solve(result, merged_pts3d, Rp, tp, gpnp_timing);
    }

    auto t_pnp_end = std::chrono::high_resolution_clock::now();
    result.timing["gpnp"] = std::chrono::duration<double, std::milli>(t_pnp_end - t_pnp_start).count();

    // 9. Restore full-image coordinates + finalize
    cv::Point2d left_off(static_cast<double>(left_pri.x), static_cast<double>(left_pri.y));
    cv::Point2d right_off(static_cast<double>(right_pri.x), static_cast<double>(right_pri.y));
    offsetResultToOriginal(result, left_off, right_off, left_color_orig, right_color_orig);

    if (pose.success) {
        finalizePose(result, pose);
    }

    // ---- Visualization (dual-ROI) ----
    if (visualize && pose.success) {
        std::string prefix = "_f" + std::to_string(state_.frame_count);

        const cv::Scalar BC_COLOR(0, 0, 255);    // red: BinaryCorner corners (class 0 edges)
        const cv::Scalar AK_COLOR(0, 255, 0);    // green: AKAZE features (class 1 center)
        int ak_count = total_use - bc_total;

        auto projPoint = [&](const Eigen::Vector3d& P) -> cv::Point {
            if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
            Eigen::Vector2d uv = projectToImage(P, camera_.K);
            return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
        };

        // --- Panel 0: Overview — ROI rectangles on original image ---
        {
            cv::Mat p0 = left_color_orig.clone();
            cv::rectangle(p0,
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height),
                cv::Scalar(255, 0, 0), 2);
            cv::rectangle(p0,
                cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height),
                cv::Scalar(0, 255, 0), 2);
            cv::putText(p0, "class0 (BC)",
                cv::Point(left_pri.x + 4, left_pri.y + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1);
            cv::putText(p0, "class1 (AK)",
                cv::Point(left_sec.x + 4, left_sec.y + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
            cv::imwrite(output_dir_ + "/dual_roi_overview" + prefix + ".png", p0);
        }

        // --- Panel 1: Class 0 ROI zoomed — BC corners (red) + AK features (green) ---
        {
            cv::Mat p1 = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            for (int i = 0; i < total_use; ++i) {
                cv::Point pt(static_cast<int>(result.pts_left_match[i].x - lx),
                             static_cast<int>(result.pts_left_match[i].y - ly));
                if (i < bc_total)
                    cv::circle(p1, pt, 3, BC_COLOR, -1);
                else
                    cv::circle(p1, pt, 3, AK_COLOR, -1);
            }
            cv::imwrite(output_dir_ + "/dual_roi_corners" + prefix + ".png", p1);
        }

        // --- Panel 2: 3D axes on original image (origin = template center) ---
        {
            cv::Mat p2 = left_color_orig.clone();
            double axis_len = 100.0;
            double cx = config_.template_real_width_mm  / 2.0;  // template center X (mm)
            double cy = config_.template_real_height_mm / 2.0;  // template center Y (mm)
            Eigen::Vector3d o  = pose.R * Eigen::Vector3d(cx,        cy,        0) + pose.t;
            Eigen::Vector3d ax = pose.R * Eigen::Vector3d(cx + axis_len, cy,     0) + pose.t;
            Eigen::Vector3d ay = pose.R * Eigen::Vector3d(cx,       cy + axis_len, 0) + pose.t;
            Eigen::Vector3d az = pose.R * Eigen::Vector3d(cx,       cy,        axis_len) + pose.t;
            cv::line(p2, projPoint(o), projPoint(ax), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::line(p2, projPoint(o), projPoint(ay), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(p2, projPoint(o), projPoint(az), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            cv::imwrite(output_dir_ + "/dual_roi_axes" + prefix + ".png", p2);
        }

        // --- Panel 3: Reprojection error on class 0 ROI zoomed ---
        if (total_use > 0 && !merged_pts3d.empty()) {
            cv::Mat p3 = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            for (int i = 0; i < total_use && i < static_cast<int>(merged_pts3d.size()); ++i) {
                Eigen::Vector3d P_cam = pose.R * merged_pts3d[i] + pose.t;
                Eigen::Vector2d uv = projectToImage(P_cam, camera_.K);
                cv::Point pd(static_cast<int>(uv.x() - lx), static_cast<int>(uv.y() - ly));
                cv::Point po(static_cast<int>(result.pts_left_match[i].x - lx),
                             static_cast<int>(result.pts_left_match[i].y - ly));
                cv::Scalar obs_color = (i < bc_total) ? BC_COLOR : AK_COLOR;
                cv::circle(p3, pd, 2, cv::Scalar(0, 255, 0), -1);          // projected = green dot
                cv::circle(p3, po, 4, obs_color, 1);                        // observed = color ring
                cv::line(p3, pd, po, cv::Scalar(0, 255, 255), 1, cv::LINE_AA); // error = yellow
            }
            cv::imwrite(output_dir_ + "/dual_roi_reproj" + prefix + ".png", p3);
        }

        // --- Panel 4: Right-image class 0 ROI with matched corners ---
        {
            cv::Mat p4 = right_color_orig(
                cv::Rect(right_pri.x, right_pri.y, right_pri.width, right_pri.height)).clone();
            float rx = static_cast<float>(right_off.x), ry = static_cast<float>(right_off.y);
            for (int i = 0; i < total_use; ++i) {
                cv::Point pt(static_cast<int>(result.pts_right_good[i].x - rx),
                             static_cast<int>(result.pts_right_good[i].y - ry));
                if (i < bc_total)
                    cv::circle(p4, pt, 3, BC_COLOR, -1);
                else
                    cv::circle(p4, pt, 3, AK_COLOR, -1);
            }
            cv::imwrite(output_dir_ + "/dual_roi_right" + prefix + ".png", p4);
        }

        // --- Panel 5: Image ↔ Template correspondence (side-by-side) ---
        {
            // Left: class-0-ROI image
            cv::Mat p5_left = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();

            // Right: AKAZE template (grayscale → BGR, resize to match left height)
            const cv::Mat& tmpl_gray = akaze_extractor_->templateData().gray_image;
            cv::Mat tmpl_color;
            if (!tmpl_gray.empty()) {
                cv::cvtColor(tmpl_gray, tmpl_color, cv::COLOR_GRAY2BGR);
            } else {
                tmpl_color = cv::Mat(p5_left.rows, p5_left.rows, CV_8UC3,
                                     cv::Scalar(128, 128, 128));
            }
            double scale_tmpl = static_cast<double>(p5_left.rows) / tmpl_color.rows;
            cv::Mat tmpl_resized;
            cv::resize(tmpl_color, tmpl_resized, cv::Size(), scale_tmpl, scale_tmpl,
                      cv::INTER_NEAREST);

            cv::Mat p5;
            cv::hconcat(p5_left, tmpl_resized, p5);

            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            int offset_x = p5_left.cols;  // horizontal boundary between image and template

            for (int i = 0; i < total_use; ++i) {
                // Image point (left side, class-0-ROI-local)
                cv::Point pt_img(static_cast<int>(result.pts_left_match[i].x - lx),
                                 static_cast<int>(result.pts_left_match[i].y - ly));

                // Template point (right side, scaled to match display size)
                cv::Point2f tmpl_pt;
                if (i < bc_total && i < static_cast<int>(dual_bc_tmpl_corners_.size())) {
                    tmpl_pt = dual_bc_tmpl_corners_[i] * scale_tmpl;
                } else {
                    int ak_idx = i - bc_total;
                    if (ak_idx >= 0 && ak_idx < static_cast<int>(result_ak.pts_template_match.size()))
                        tmpl_pt = result_ak.pts_template_match[ak_idx] * scale_tmpl;
                    else
                        continue;
                }
                cv::Point pt_tmpl(static_cast<int>(tmpl_pt.x) + offset_x,
                                  static_cast<int>(tmpl_pt.y));

                cv::Scalar color = (i < bc_total) ? BC_COLOR : AK_COLOR;
                cv::circle(p5, pt_img,  2, color, -1);
                cv::circle(p5, pt_tmpl, 2, color, -1);
                cv::line(p5, pt_img, pt_tmpl, color, 1, cv::LINE_AA);
            }
            cv::imwrite(output_dir_ + "/dual_roi_correspondence" + prefix + ".png", p5);
        }

        std::cout << "  [DualRoi] Visualized: " << bc_total << " BC + "
                  << ak_count << " AK corners" << std::endl;
    }

    result.n_matched = total_use;
    result.n_projected = total_use;
    addLogEntry(result, is_first, false);

    std::cout << "[DualRoi] Frame done: n_pts=" << total_use
              << "  GPNP=" << (pose.success ? "OK" : "FAIL")
              << "  time=" << result.total_time_ms() << "ms"
              << std::endl;

    return result;
}

// ============================================================================
// Dual-ROI Mono —— class 0 (BC) + class 1 (AK) → MonoPnPSolver
// ============================================================================

PipelineResult StereoTracker::processDualRoiMono(const cv::Mat& left_img,
                                                  const RoiGroup& left_group,
                                                  bool visualize) {
    // 0. Ensure template preprocessing is done
    prepareDualBcTemplate();

    // ---- Load image ----
    auto [left_color, left_gray] = loadImage(left_img);
    if (left_gray.empty()) {
        PipelineResult empty;
        empty.is_first_frame = !state_.has_cache;
        return empty;
    }

    cv::Mat left_color_orig = left_color.clone();
    bool is_first = !state_.has_cache;
    int pad = config_.dual_roi_secondary_expand;

    // 1. Expand secondary ROI
    auto expandRoi = [](const RoiRect& roi, int p, int img_w, int img_h) -> RoiRect {
        int x = std::max(0, roi.x - p);
        int y = std::max(0, roi.y - p);
        int w = std::min(img_w - x, roi.width  + 2 * p);
        int h = std::min(img_h - y, roi.height + 2 * p);
        return RoiRect{x, y, w, h};
    };

    RoiRect left_pri = left_group.primary;
    RoiRect left_sec = expandRoi(left_group.secondary, pad, left_img.cols, left_img.rows);

    cv::Point2d sec_to_pri_offset(
        static_cast<double>(left_sec.x - left_pri.x),
        static_cast<double>(left_sec.y - left_pri.y));

    std::cout << "[DualRoi][Mono] pad=" << pad
              << "  primary=" << left_pri.width << "x" << left_pri.height
              << "  secondary(raw)=" << left_group.secondary.width << "x" << left_group.secondary.height
              << "  secondary(expanded)=" << left_sec.width << "x" << left_sec.height
              << "  offset=(" << sec_to_pri_offset.x << "," << sec_to_pri_offset.y << ")"
              << std::endl;

    // 2. Crop images (left only)
    cv::Mat left_c0_gray  = left_gray(cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
    cv::Mat left_c0_color = left_color(cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
    cv::Mat left_c1_gray  = left_gray(cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height)).clone();
    cv::Mat left_c1_color = left_color(cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height)).clone();

    // 3. BC extraction on class 0 (mono)
    PipelineResult result_bc = binary_extractor_->extractMono(left_c0_gray, left_c0_color);
    int n_bc = static_cast<int>(result_bc.pts_left_match.size());
    std::cout << "[DualRoi][Mono] BinaryCorner on class 0: " << n_bc << " corners" << std::endl;

    // 4. AK extraction on class 1 (mono)
    PipelineResult result_ak = dual_akaze_extractor_->extractMono(left_c1_gray, left_c1_color);
    int m_ak_match = static_cast<int>(result_ak.pts_left_match.size());
    std::cout << "[DualRoi][Mono] AKAZE on class 1: " << m_ak_match << " template matches"
              << " (kp=" << result_ak.n_kp_left << ")"
              << std::endl;

    // 5. Coordinate transform: class-1-local → class-0-local
    auto offsetPoints = [](std::vector<cv::Point2f>& pts, const cv::Point2d& offset) {
        float fx = static_cast<float>(offset.x), fy = static_cast<float>(offset.y);
        for (auto& p : pts) { p.x += fx; p.y += fy; }
    };
    offsetPoints(result_ak.pts_left_match, sec_to_pri_offset);
    offsetPoints(result_ak.pts_left_good,  sec_to_pri_offset);
    for (auto& kp : result_ak.kp_left) {
        kp.pt.x += static_cast<float>(sec_to_pri_offset.x);
        kp.pt.y += static_cast<float>(sec_to_pri_offset.y);
    }

    // 6. Merge BC + AK → 2D points + 3D correspondences
    std::vector<cv::Point2f> merged_pts_2d;
    std::vector<cv::KeyPoint> merged_kp_left;
    std::vector<Eigen::Vector3d> merged_pts_3d;

    // --- BC 3D points: align ordering with BC matched template angle ---
    // BC extractFromBinary() reorders corners by matchCorners(ref_angle = matched->angle).
    // dual_bc_tmpl_pts3d_ is ordered by reorderByGeometry(ref_angle = 0°).
    // If matched angle ≠ 0°, the index-based correspondence is wrong.
    // Fix: reorder dual_bc_tmpl_pts3d_ to the same reference angle.
    std::vector<Eigen::Vector3d> bc_pts3d = dual_bc_tmpl_pts3d_;  // copy
    const TemplateData* bc_matched = binary_extractor_->lastMatchedTemplate();
    if (bc_matched && std::abs(bc_matched->angle) > 0.5
        && !dual_bc_tmpl_corners_.empty()) {
        // Build Point2f from dual_bc_tmpl_corners_ for reorderByGeometry
        std::vector<cv::Point2f> tmpl_corners_2f = dual_bc_tmpl_corners_;
        cv::Point2f px_ctr(
            static_cast<float>(akaze_extractor_->templateData().template_width  / 2.0),
            static_cast<float>(akaze_extractor_->templateData().template_height / 2.0));
        auto order = BinaryCornerExtractor::reorderByGeometry(
            tmpl_corners_2f, px_ctr, bc_matched->angle);
        bc_pts3d.clear();
        bc_pts3d.reserve(order.size());
        for (int idx : order)
            bc_pts3d.push_back(dual_bc_tmpl_pts3d_[idx]);
        std::cout << "  [DualRoi][Mono] BC pts3d reordered for angle="
                  << bc_matched->angle << "°" << std::endl;
    }

    // --- BC contribution: corners[i] ↔ bc_pts3d[i] ---
    int n_bc_3d = static_cast<int>(bc_pts3d.size());
    int n_bc_use = std::min(n_bc, n_bc_3d);
    for (int i = 0; i < n_bc_use; ++i) {
        merged_pts_2d.push_back(result_bc.pts_left_match[i]);
        merged_kp_left.emplace_back(result_bc.pts_left_match[i], 1.0f);
        merged_pts_3d.push_back(bc_pts3d[i]);
    }

    // --- AK contribution: matches[i] ↔ template_.pts_3d[good_matches[i].trainIdx] ---
    const auto& ak_pts3d = dual_akaze_extractor_->templateData().pts_3d;
    for (size_t i = 0; i < result_ak.good_matches.size(); ++i) {
        int idx = result_ak.good_matches[i].trainIdx;
        if (idx >= 0 && idx < static_cast<int>(ak_pts3d.size())) {
            merged_pts_2d.push_back(result_ak.pts_left_match[i]);
            merged_kp_left.emplace_back(result_ak.pts_left_match[i], 1.0f);
            merged_pts_3d.push_back(ak_pts3d[idx]);
        }
    }

    int total_use = static_cast<int>(merged_pts_2d.size());
    std::cout << "[DualRoi][Mono] Merged: " << total_use << " total (BC=" << n_bc_use
              << ", AK=" << m_ak_match << ")"
              << "  pts3d=" << merged_pts_3d.size() << std::endl;

    if (total_use < 4) {
        std::cerr << "[DualRoi][Mono] Too few merged points (" << total_use << "), aborting" << std::endl;
        PipelineResult empty;
        empty.is_first_frame = is_first;
        empty.gpnp_success = false;
        return empty;
    }

    // 7. Restore full-image coordinates
    cv::Point2d left_off(static_cast<double>(left_pri.x), static_cast<double>(left_pri.y));
    offsetPoints(merged_pts_2d, left_off);
    for (auto& kp : merged_kp_left) {
        kp.pt.x += static_cast<float>(left_off.x);
        kp.pt.y += static_cast<float>(left_off.y);
    }

    // 8. Pose estimation (mono EPnP)
    PoseEstimate pose = mono_pnp_.solve(merged_pts_2d, merged_pts_3d, camera_.K);

    // 9. Build PipelineResult
    PipelineResult result;
    result.kp_left         = std::move(merged_kp_left);
    result.n_kp_left       = total_use;
    result.pts_left_match  = merged_pts_2d;
    result.pts_left_good   = merged_pts_2d;
    result.left_color      = left_color_orig;
    result.is_first_frame  = is_first;

    if (pose.success) {
        finalizePose(result, pose);
    }

    result.n_matched   = total_use;
    result.n_projected = 0;
    addLogEntry(result, is_first, false);

    // ---- Visualization (simplified, left-only) ----
    if (visualize && pose.success && !output_dir_.empty()) {
        std::string prefix = "_f" + std::to_string(state_.frame_count);

        const cv::Scalar BC_COLOR(0, 0, 255);    // red: BinaryCorner corners
        const cv::Scalar AK_COLOR(0, 255, 0);    // green: AKAZE features

        // Panel 0: Overview — ROI rectangles on original image
        {
            cv::Mat p0 = left_color_orig.clone();
            cv::rectangle(p0,
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height),
                cv::Scalar(255, 0, 0), 2);
            cv::rectangle(p0,
                cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height),
                cv::Scalar(0, 255, 0), 2);
            cv::putText(p0, "class0 (BC)",
                cv::Point(left_pri.x + 4, left_pri.y + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1);
            cv::putText(p0, "class1 (AK)",
                cv::Point(left_sec.x + 4, left_sec.y + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
            cv::imwrite(output_dir_ + "/dual_roi_mono_overview" + prefix + ".png", p0);
        }

        // Panel 1: Class 0 ROI zoomed — BC (red) + AK (green) merged points
        {
            cv::Mat p1 = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            for (int i = 0; i < total_use; ++i) {
                cv::Point pt(static_cast<int>(merged_pts_2d[i].x - lx),
                             static_cast<int>(merged_pts_2d[i].y - ly));
                if (i < n_bc_use)
                    cv::circle(p1, pt, 3, BC_COLOR, -1);
                else
                    cv::circle(p1, pt, 3, AK_COLOR, -1);
            }
            cv::imwrite(output_dir_ + "/dual_roi_mono_corners" + prefix + ".png", p1);
        }

        // Panel 2: 3D axes on original image
        {
            cv::Mat p2 = left_color_orig.clone();
            double axis_len = 100.0;
            auto projPoint = [&](const Eigen::Vector3d& P) -> cv::Point {
                if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
                double fx = camera_.K(0, 0), fy = camera_.K(1, 1);
                double cx = camera_.K(0, 2), cy = camera_.K(1, 2);
                double u = fx * P.x() / P.z() + cx;
                double v = fy * P.y() / P.z() + cy;
                return cv::Point(static_cast<int>(u), static_cast<int>(v));
            };
            Eigen::Vector3d o  = pose.R * Eigen::Vector3d(0,            0,             0) + pose.t;
            Eigen::Vector3d ax = pose.R * Eigen::Vector3d(axis_len,     0,             0) + pose.t;
            Eigen::Vector3d ay = pose.R * Eigen::Vector3d(0,            axis_len,      0) + pose.t;
            Eigen::Vector3d az = pose.R * Eigen::Vector3d(0,            0,             axis_len) + pose.t;
            cv::line(p2, projPoint(o), projPoint(ax), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::line(p2, projPoint(o), projPoint(ay), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(p2, projPoint(o), projPoint(az), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            cv::imwrite(output_dir_ + "/dual_roi_mono_axes" + prefix + ".png", p2);
        }

        // Panel 3: Reprojection error on class 0 ROI zoomed
        if (total_use > 0 && !merged_pts_3d.empty()) {
            cv::Mat p3 = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            for (int i = 0; i < total_use && i < static_cast<int>(merged_pts_3d.size()); ++i) {
                Eigen::Vector3d P_cam = pose.R * merged_pts_3d[i] + pose.t;
                double fx = camera_.K(0, 0), fy = camera_.K(1, 1);
                double cx = camera_.K(0, 2), cy = camera_.K(1, 2);
                if (std::abs(P_cam.z()) < 1e-6) continue;
                double u = fx * P_cam.x() / P_cam.z() + cx;
                double v = fy * P_cam.y() / P_cam.z() + cy;
                cv::Point pd(static_cast<int>(u - lx), static_cast<int>(v - ly));
                cv::Point po(static_cast<int>(merged_pts_2d[i].x - lx),
                             static_cast<int>(merged_pts_2d[i].y - ly));
                cv::Scalar obs_color = (i < n_bc_use) ? BC_COLOR : AK_COLOR;
                cv::circle(p3, pd, 2, cv::Scalar(0, 255, 0), -1);
                cv::circle(p3, po, 4, obs_color, 1);
                cv::line(p3, pd, po, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            }
            cv::imwrite(output_dir_ + "/dual_roi_mono_reproj" + prefix + ".png", p3);
        }

        std::cout << "  [DualRoi][Mono] Visualized: " << n_bc_use << " BC + "
                  << (total_use - n_bc_use) << " AK corners" << std::endl;
    }

    std::cout << "[DualRoi][Mono] Frame done: n_pts=" << total_use
              << "  PnP=" << (pose.success ? "OK" : "FAIL")
              << std::endl;

    return result;
}

void StereoTracker::clearCache() { state_ = TrackingState{}; }

// ============================================================================
// 单目模式 —— 仅左图特征提取 + PnP 解算
// ============================================================================

PipelineResult StereoTracker::processMono(const cv::Mat& left_img,
                                           bool visualize,
                                           const RoiGroup* left_group) {
    PipelineResult result;
    result.success = false;

    if (!mono_cfg_.enabled) {
        std::cerr << "[Mono] mono mode not enabled, set MonoConfig::enabled=true" << std::endl;
        return result;
    }

    if (left_img.empty()) {
        std::cerr << "[Mono] empty left image" << std::endl;
        return result;
    }

    // ---- Dual-ROI dispatch ----
    if (left_group && left_group->is_dual) {
        result = processDualRoiMono(left_img, *left_group, visualize);
        state_.frame_count++;
        return result;
    }

    // 加载灰度图
    auto [left_color, left_gray] = loadImage(left_img);
    if (left_gray.empty()) {
        std::cerr << "[Mono] failed to load left image" << std::endl;
        return result;
    }
    result.left_color = left_color;

    // ROI 校验：无 ROI → 全图
    const RoiRect* left_roi = left_group ? &left_group->primary : nullptr;
    RoiRect roi = validateRoi(left_roi, left_img.size(), "mono_left");
    if (!roi.valid()) {
        roi = RoiRect{0, 0, left_img.cols, left_img.rows};
    }

    int roi_area = roi.width * roi.height;
    std::cout << "[Mono] ROI area=" << roi_area
              << " (" << roi.width << "x" << roi.height << ")"
              << std::endl;

    // 裁剪左图 ROI
    cv::Mat left_gray_roi  = left_gray( cv::Rect(roi.x, roi.y, roi.width, roi.height));
    cv::Mat left_color_roi = left_color(cv::Rect(roi.x, roi.y, roi.width, roi.height));
    cv::Point2d left_offset(roi.x, roi.y);

    // 策略链选择
    configureStrategyChain(roi_area);

    bool is_first = (state_.frame_count == 0);
    bool extracted = false;

    // 尝试主策略 + 退化链
    std::vector<FeatureExtractor*> chain;
    chain.push_back(extractor_);
    for (auto* fb : fallback_extractors_)
        chain.push_back(fb);

    FeatureExtractor* winning_ext = nullptr;

    for (auto* ext : chain) {
        if (!ext) continue;

        std::cout << "[Mono] Trying extractor: " << ext->name() << std::endl;

        // 单目提取（仅左图，2 参数）
        PipelineResult local = ext->extractMono(left_gray_roi, left_color_roi);

        // 将 ROI 局部坐标恢复到全图坐标系
        if (!local.pts_left_match.empty()) {
            for (auto& p : local.pts_left_match) {
                p.x += static_cast<float>(left_offset.x);
                p.y += static_cast<float>(left_offset.y);
            }
        }
        if (!local.pts_left_good.empty()) {
            for (auto& p : local.pts_left_good) {
                p.x += static_cast<float>(left_offset.x);
                p.y += static_cast<float>(left_offset.y);
            }
        }
        for (auto& kp : local.kp_left) {
            kp.pt.x += static_cast<float>(left_offset.x);
            kp.pt.y += static_cast<float>(left_offset.y);
        }

        if (local.success && local.n_kp_left >= 3) {
            result = std::move(local);
            result.left_color = left_color;
            result.left_roi_offset_x = static_cast<int>(left_offset.x);
            result.left_roi_offset_y = static_cast<int>(left_offset.y);
            winning_ext = ext;
            extracted = true;
            std::cout << "[Mono] Extractor " << ext->name()
                      << " succeeded, n_kp=" << result.n_kp_left << std::endl;
            break;
        }

        std::cout << "[Mono] Extractor " << ext->name() << " failed, degrading..." << std::endl;
    }

    if (!extracted) {
        std::cerr << "[Mono] All extractors failed" << std::endl;
        addLogEntry(result, is_first, true);
        return result;
    }

    // 单目 PnP 解算（EPnP，无 GPNP / warm-start）
    // Align 3D points with 2D matches using good_matches trainIdx
    // Use the winning extractor's pts_3d (e.g. BinaryCorner/TinyTarget),
    // falling back to AKAZE template_ only if the extractor has no pts_3d.
    const auto& pnp_pts_3d = winning_ext && !winning_ext->templateData().pts_3d.empty()
        ? winning_ext->templateData().pts_3d
        : template_.pts_3d;
    std::vector<Eigen::Vector3d> matched_pts_3d;
    matched_pts_3d.reserve(result.good_matches.size());
    for (const auto& m : result.good_matches) {
        int idx = m.trainIdx;
        if (idx >= 0 && idx < static_cast<int>(pnp_pts_3d.size())) {
            matched_pts_3d.push_back(pnp_pts_3d[idx]);
        }
    }
    PoseEstimate pose = mono_pnp_.solve(result.pts_left_match, matched_pts_3d, camera_.K);
    finalizePose(result, pose);

    result.success = pose.success;
    addLogEntry(result, is_first, false);

    // 可视化
    if (visualize && !output_dir_.empty()) {
        std::string prefix = "_f" + std::to_string(state_.frame_count);

        if (winning_ext && winning_ext->name() == "BinaryCorner" &&
            roi.width > 0 && roi.height > 0) {
            // ================================================================
            // BinaryCorner 专用多面板可视化（参照双目 BC 路径）
            // ================================================================
            auto* bce = binary_extractor_.get();
            if (bce) {
                // --- 展开 ROI 5x 获得放大视图 ---
                auto expandRect = [](const RoiRect& r, const cv::Size& imgSz) -> cv::Rect {
                    int cx = r.x + r.width / 2;
                    int cy = r.y + r.height / 2;
                    int ew = r.width * 5;
                    int eh = r.height * 5;
                    int x = std::max(0, cx - ew / 2);
                    int y = std::max(0, cy - eh / 2);
                    int w = std::min(ew, imgSz.width - x);
                    int h = std::min(eh, imgSz.height - y);
                    return cv::Rect(x, y, w, h);
                };
                cv::Rect expand_L = expandRect(roi, left_color.size());
                cv::Mat view_L = left_color(expand_L).clone();
                float elx = static_cast<float>(expand_L.x);
                float ely = static_cast<float>(expand_L.y);
                auto toView = [&](const cv::Point2f& p) {
                    return cv::Point2f(p.x - elx, p.y - ely);
                };

                auto proj = [&](const Eigen::Vector3d& P) -> cv::Point {
                    if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
                    Eigen::Vector2d uv = projectToImage(P, camera_.K);
                    return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
                };

                const cv::Scalar CORNER_COLORS[] = {
                    {0,0,255}, {0,255,0}, {0,255,255}, {255,0,0}, {255,0,255},
                    {255,255,0}, {128,0,255}, {0,128,255}, {255,128,0}, {128,255,0}
                };

                const auto& bc_pts_3d = winning_ext->templateData().pts_3d;
                float lx_roi = static_cast<float>(left_offset.x);
                float ly_roi = static_cast<float>(left_offset.y);

                // --- Panel 0: 二值化阈值图 + 角点叠加 ---
                {
                    cv::Mat bl_bgr;
                    cv::cvtColor(bce->lastLeftBinary(), bl_bgr, cv::COLOR_GRAY2BGR);
                    for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                        cv::Point p(static_cast<int>(result.pts_left_match[i].x - lx_roi),
                                     static_cast<int>(result.pts_left_match[i].y - ly_roi));
                        cv::circle(bl_bgr, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                    cv::imwrite(output_dir_ + "/binary_corner_mono_binary" + prefix + ".png", bl_bgr);
                }

                // --- Panel 1: 旋转回正二值图 ---
                if (!bce->lastUprightBinary().empty()) {
                    cv::Mat up;
                    cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                    cv::imwrite(output_dir_ + "/binary_corner_mono_upright" + prefix + ".png", up);
                }

                // --- Panel 2: 3D 坐标轴（展开左视图）---
                {
                    cv::Mat p2 = view_L.clone();
                    double axis_len = 100.0;
                    Eigen::Vector3d o  = result.R * Eigen::Vector3d(0,0,0) + result.t;
                    Eigen::Vector3d ax = result.R * Eigen::Vector3d(axis_len,0,0) + result.t;
                    Eigen::Vector3d ay = result.R * Eigen::Vector3d(0,axis_len,0) + result.t;
                    Eigen::Vector3d az = result.R * Eigen::Vector3d(0,0,axis_len) + result.t;
                    cv::Point o_p = proj(o);   o_p.x -= expand_L.x; o_p.y -= expand_L.y;
                    cv::Point ax_p = proj(ax); ax_p.x -= expand_L.x; ax_p.y -= expand_L.y;
                    cv::Point ay_p = proj(ay); ay_p.x -= expand_L.x; ay_p.y -= expand_L.y;
                    cv::Point az_p = proj(az); az_p.x -= expand_L.x; az_p.y -= expand_L.y;
                    cv::line(p2, o_p, ax_p, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                    cv::line(p2, o_p, ay_p, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                    cv::line(p2, o_p, az_p, cv::Scalar(255,0,0), 2, cv::LINE_AA);
                    cv::imwrite(output_dir_ + "/binary_corner_mono_axes" + prefix + ".png", p2);
                }

                // --- Panel 3: 模板角点对应（左视图 | 模板图）---
                {
                    cv::Mat p3_l = view_L.clone();
                    for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                        cv::Point2f pv = toView(result.pts_left_match[i]);
                        cv::circle(p3_l, cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                                   1, CORNER_COLORS[i % 10], -1);
                    }
                    const TemplateData* matched_tmpl = bce->lastMatchedTemplate();
                    cv::Mat p3_tmpl;
                    if (matched_tmpl) {
                        cv::cvtColor(matched_tmpl->image, p3_tmpl, cv::COLOR_GRAY2BGR);
                    } else {
                        p3_tmpl = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128,128,128));
                    }
                    cv::resize(p3_tmpl, p3_tmpl, p3_l.size(), 0, 0, cv::INTER_NEAREST);
                    if (matched_tmpl) {
                        double dsx = static_cast<double>(p3_tmpl.cols) / matched_tmpl->image.cols;
                        double dsy = static_cast<double>(p3_tmpl.rows) / matched_tmpl->image.rows;
                        for (size_t i = 0; i < matched_tmpl->corners.size(); ++i) {
                            cv::Point p(static_cast<int>(matched_tmpl->corners[i].x * dsx),
                                        static_cast<int>(matched_tmpl->corners[i].y * dsy));
                            cv::circle(p3_tmpl, p, 1, CORNER_COLORS[i % 10], -1);
                        }
                    }
                    cv::Mat p3;
                    cv::hconcat(p3_l, p3_tmpl, p3);
                    cv::imwrite(output_dir_ + "/binary_corner_mono_template" + prefix + ".png", p3);
                }

                // --- Panel 4: 重投影误差 ---
                if (!bc_pts_3d.empty()) {
                    cv::Mat p4 = view_L.clone();

                    std::vector<cv::Point3d> obj_pts;
                    std::vector<cv::Point2f> obs_pts;
                    for (const auto& m : result.good_matches) {
                        if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(bc_pts_3d.size()) &&
                            m.queryIdx >= 0 && m.queryIdx < static_cast<int>(result.pts_left_match.size())) {
                            obj_pts.emplace_back(bc_pts_3d[m.trainIdx].x(),
                                                bc_pts_3d[m.trainIdx].y(),
                                                bc_pts_3d[m.trainIdx].z());
                            obs_pts.push_back(result.pts_left_match[m.queryIdx]);
                        }
                    }

                    if (!obj_pts.empty()) {
                        cv::Mat rvec, tvec4(3, 1, CV_64F);
                        tvec4.at<double>(0) = result.t(0);
                        tvec4.at<double>(1) = result.t(1);
                        tvec4.at<double>(2) = result.t(2);
                        cv::Mat R4(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                R4.at<double>(r, c) = result.R(r, c);
                        cv::Rodrigues(R4, rvec);
                        cv::Mat K4(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                K4.at<double>(r, c) = camera_.K(r, c);

                        std::vector<cv::Point2d> projected;
                        cv::projectPoints(obj_pts, rvec, tvec4, K4, cv::Mat(), projected);

                        for (size_t i = 0; i < projected.size() && i < obs_pts.size(); ++i) {
                            cv::Point2f ov = toView(obs_pts[i]);
                            cv::Point pd(static_cast<int>(projected[i].x - expand_L.x),
                                         static_cast<int>(projected[i].y - expand_L.y));
                            cv::Point po(static_cast<int>(ov.x), static_cast<int>(ov.y));
                            cv::circle(p4, pd, 1, cv::Scalar(0, 255, 0), -1);
                            cv::circle(p4, po, 1, cv::Scalar(0, 0, 255), 1);
                            cv::line(p4, pd, po, cv::Scalar(0, 255, 255), 1);
                        }
                    }
                    cv::imwrite(output_dir_ + "/binary_corner_mono_reproj" + prefix + ".png", p4);
                }

                std::cout << "[Mono] BC visualization saved to " << output_dir_ << std::endl;
            }
        } else if (winning_ext && winning_ext->name() == "TinyTarget" &&
                   roi.width > 0 && roi.height > 0) {
            // ================================================================
            // TinyTarget 专用可视化（参照双目 TT 路径，无二值图）
            // ================================================================
            auto* tte = tiny_extractor_.get();
            if (tte) {
                auto expandRect = [](const RoiRect& r, const cv::Size& imgSz) -> cv::Rect {
                    int cx = r.x + r.width / 2;
                    int cy = r.y + r.height / 2;
                    int ew = r.width * 5;
                    int eh = r.height * 5;
                    int x = std::max(0, cx - ew / 2);
                    int y = std::max(0, cy - eh / 2);
                    int w = std::min(ew, imgSz.width - x);
                    int h = std::min(eh, imgSz.height - y);
                    return cv::Rect(x, y, w, h);
                };
                cv::Rect expand_L = expandRect(roi, left_color.size());
                cv::Mat view_L = left_color(expand_L).clone();
                float elx = static_cast<float>(expand_L.x);
                float ely = static_cast<float>(expand_L.y);
                auto toView = [&](const cv::Point2f& p) {
                    return cv::Point2f(p.x - elx, p.y - ely);
                };
                auto proj = [&](const Eigen::Vector3d& P) -> cv::Point {
                    if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
                    Eigen::Vector2d uv = projectToImage(P, camera_.K);
                    return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
                };
                const cv::Scalar CORNER_COLORS[] = {
                    {0,0,255}, {0,255,0}, {0,255,255}, {255,0,0}
                };
                const auto& tt_pts_3d = winning_ext->templateData().pts_3d;

                // --- Panel 0: 3D 坐标轴（展开左视图）---
                if (result.success) {
                    cv::Mat p0 = view_L.clone();
                    double axis_len = 100.0;
                    Eigen::Vector3d o  = result.R * Eigen::Vector3d(0,0,0) + result.t;
                    Eigen::Vector3d ax = result.R * Eigen::Vector3d(axis_len,0,0) + result.t;
                    Eigen::Vector3d ay = result.R * Eigen::Vector3d(0,axis_len,0) + result.t;
                    Eigen::Vector3d az = result.R * Eigen::Vector3d(0,0,axis_len) + result.t;
                    cv::Point o_p = proj(o);   o_p.x -= expand_L.x; o_p.y -= expand_L.y;
                    cv::Point ax_p = proj(ax); ax_p.x -= expand_L.x; ax_p.y -= expand_L.y;
                    cv::Point ay_p = proj(ay); ay_p.x -= expand_L.x; ay_p.y -= expand_L.y;
                    cv::Point az_p = proj(az); az_p.x -= expand_L.x; az_p.y -= expand_L.y;
                    cv::line(p0, o_p, ax_p, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                    cv::line(p0, o_p, ay_p, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                    cv::line(p0, o_p, az_p, cv::Scalar(255,0,0), 2, cv::LINE_AA);
                    cv::imwrite(output_dir_ + "/tiny_target_mono_axes" + prefix + ".png", p0);
                }

                // --- Panel 1: 模板角点对应（左视图 | 模板图）---
                {
                    cv::Mat p1_l = view_L.clone();
                    for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                        cv::Point2f pv = toView(result.pts_left_match[i]);
                        cv::circle(p1_l, cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                                   2, CORNER_COLORS[i % 4], -1);
                    }
                    const TemplateData* matched_tmpl = tte->lastMatchedTemplate();
                    cv::Mat p1_tmpl;
                    if (matched_tmpl) {
                        cv::cvtColor(matched_tmpl->image, p1_tmpl, cv::COLOR_GRAY2BGR);
                    } else {
                        p1_tmpl = cv::Mat(50, 50, CV_8UC3, cv::Scalar(128,128,128));
                    }
                    cv::resize(p1_tmpl, p1_tmpl, p1_l.size(), 0, 0, cv::INTER_NEAREST);
                    if (matched_tmpl) {
                        double dsx = static_cast<double>(p1_tmpl.cols) / matched_tmpl->image.cols;
                        double dsy = static_cast<double>(p1_tmpl.rows) / matched_tmpl->image.rows;
                        for (size_t i = 0; i < matched_tmpl->corners.size(); ++i) {
                            cv::Point p(static_cast<int>(matched_tmpl->corners[i].x * dsx),
                                        static_cast<int>(matched_tmpl->corners[i].y * dsy));
                            cv::circle(p1_tmpl, p, 2, CORNER_COLORS[i % 4], -1);
                        }
                    }
                    cv::Mat p1;
                    cv::hconcat(p1_l, p1_tmpl, p1);
                    cv::imwrite(output_dir_ + "/tiny_target_mono_template" + prefix + ".png", p1);
                }

                // --- Panel 2: 重投影误差 ---
                if (!tt_pts_3d.empty()) {
                    cv::Mat p2 = view_L.clone();
                    std::vector<cv::Point3d> obj_pts;
                    std::vector<cv::Point2f> obs_pts;
                    for (const auto& m : result.good_matches) {
                        if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(tt_pts_3d.size()) &&
                            m.queryIdx >= 0 && m.queryIdx < static_cast<int>(result.pts_left_match.size())) {
                            obj_pts.emplace_back(tt_pts_3d[m.trainIdx].x(),
                                                tt_pts_3d[m.trainIdx].y(),
                                                tt_pts_3d[m.trainIdx].z());
                            obs_pts.push_back(result.pts_left_match[m.queryIdx]);
                        }
                    }
                    if (!obj_pts.empty()) {
                        cv::Mat rvec, tvec4(3, 1, CV_64F);
                        tvec4.at<double>(0) = result.t(0);
                        tvec4.at<double>(1) = result.t(1);
                        tvec4.at<double>(2) = result.t(2);
                        cv::Mat R4(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                R4.at<double>(r, c) = result.R(r, c);
                        cv::Rodrigues(R4, rvec);
                        cv::Mat K4(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                K4.at<double>(r, c) = camera_.K(r, c);
                        std::vector<cv::Point2d> projected;
                        cv::projectPoints(obj_pts, rvec, tvec4, K4, cv::Mat(), projected);
                        for (size_t i = 0; i < projected.size() && i < obs_pts.size(); ++i) {
                            cv::Point2f ov = toView(obs_pts[i]);
                            cv::Point pd(static_cast<int>(projected[i].x - expand_L.x),
                                         static_cast<int>(projected[i].y - expand_L.y));
                            cv::Point po(static_cast<int>(ov.x), static_cast<int>(ov.y));
                            cv::circle(p2, pd, 2, cv::Scalar(0, 255, 0), -1);
                            cv::circle(p2, po, 2, cv::Scalar(0, 0, 255), 1);
                            cv::line(p2, pd, po, cv::Scalar(0, 255, 255), 1);
                        }
                    }
                    cv::imwrite(output_dir_ + "/tiny_target_mono_reproj" + prefix + ".png", p2);
                }

                std::cout << "[Mono] TT visualization saved to " << output_dir_ << std::endl;
            }
        } else if (winning_ext && winning_ext->name() == "AkazeGpnp") {
            // ================================================================
            // AKAZE 专用可视化（参照双目 Visualizer 面板）
            // ================================================================

            // 获取模板彩色图
            cv::Mat tmpl_bgr;
            if (!template_.gray_image.empty()) {
                cv::cvtColor(template_.gray_image, tmpl_bgr, cv::COLOR_GRAY2BGR);
            } else {
                tmpl_bgr = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
            }

            // --- Panel 0: 模板匹配（左图 | 模板图 + 匹配连线）---
            {
                int h1 = left_color.rows, w1 = left_color.cols;
                int ht = tmpl_bgr.rows, wt = tmpl_bgr.cols;
                int nh = std::max(h1, ht);
                cv::Mat lv(nh, w1, CV_8UC3, cv::Scalar(0, 0, 0));
                cv::Mat tv(nh, wt, CV_8UC3, cv::Scalar(0, 0, 0));
                left_color.copyTo(lv(cv::Rect(0, 0, w1, h1)));
                tmpl_bgr.copyTo(tv(cv::Rect(0, 0, wt, ht)));

                cv::Mat stitched;
                cv::hconcat(lv, tv, stitched);

                std::mt19937 rng(123);
                std::uniform_int_distribution<int> dist(0, 255);
                int n_match = static_cast<int>(result.good_matches.size());
                for (int i = 0; i < std::min(n_match, static_cast<int>(result.pts_left_match.size())); ++i) {
                    if (i >= static_cast<int>(result.pts_template_match.size())) break;
                    cv::Point pt_l(static_cast<int>(result.pts_left_match[i].x),
                                   static_cast<int>(result.pts_left_match[i].y));
                    cv::Point pt_t(static_cast<int>(result.pts_template_match[i].x + w1),
                                   static_cast<int>(result.pts_template_match[i].y));
                    cv::Scalar color(dist(rng), dist(rng), dist(rng));
                    cv::line(stitched, pt_l, pt_t, color, 1, cv::LINE_AA);
                }

                std::ostringstream oss;
                oss << "AKAZE Mono Template Match | Matches:" << n_match;
                cv::putText(stitched, oss.str(), cv::Point(10, nh - 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

                cv::imwrite(output_dir_ + "/akaze_mono_template_match" + prefix + ".png", stitched);
            }

            // --- Panel 1: 位姿坐标轴 ---
            if (result.success) {
                cv::Mat axes_img = left_color.clone();
                double axis_len = 50.0;
                auto proj = [&](const Eigen::Vector3d& P) -> cv::Point {
                    if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
                    Eigen::Vector2d uv = projectToImage(P, camera_.K);
                    return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
                };
                Eigen::Vector3d o  = result.R * Eigen::Vector3d(0, 0, 0) + result.t;
                Eigen::Vector3d ax = result.R * Eigen::Vector3d(axis_len, 0, 0) + result.t;
                Eigen::Vector3d ay = result.R * Eigen::Vector3d(0, axis_len, 0) + result.t;
                Eigen::Vector3d az = result.R * Eigen::Vector3d(0, 0, axis_len) + result.t;
                cv::Point o_p = proj(o), x_p = proj(ax), y_p = proj(ay), z_p = proj(az);
                cv::line(axes_img, o_p, x_p, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(axes_img, o_p, y_p, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::line(axes_img, o_p, z_p, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                cv::putText(axes_img, "X", x_p, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                cv::putText(axes_img, "Y", y_p, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                cv::putText(axes_img, "Z", z_p, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

                std::ostringstream oss;
                oss << "EPnP Pose | t=[" << static_cast<int>(result.t(0)) << ","
                    << static_cast<int>(result.t(1)) << "," << static_cast<int>(result.t(2)) << "]mm";
                cv::putText(axes_img, oss.str(), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

                cv::imwrite(output_dir_ + "/akaze_mono_axes" + prefix + ".png", axes_img);
            }

            std::cout << "[Mono] AKAZE visualization saved to " << output_dir_ << std::endl;

        } else {
            // ================================================================
            // 通用回退可视化
            // ================================================================
            cv::Mat vis = left_color.clone();
            for (const auto& pt : result.pts_left_match) {
                cv::drawMarker(vis, pt, cv::Scalar(0, 255, 0),
                               cv::MARKER_CROSS, 10, 2);
            }
            for (size_t i = 0; i < result.pts_left_match.size() && i < result.pts_template_match.size(); ++i) {
                cv::putText(vis, std::to_string(i), result.pts_left_match[i],
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
            }
            if (result.success) {
                std::vector<cv::Point3d> axis_pts = {{0,0,0}, {100,0,0}, {0,100,0}, {0,0,100}};
                std::vector<cv::Point2d> img_pts;
                cv::Mat K_cv = (cv::Mat_<double>(3,3) <<
                    camera_.K(0,0), camera_.K(0,1), camera_.K(0,2),
                    camera_.K(1,0), camera_.K(1,1), camera_.K(1,2),
                    camera_.K(2,0), camera_.K(2,1), camera_.K(2,2));
                cv::Mat rvec;
                cv::Mat R_cv = (cv::Mat_<double>(3,3) <<
                    result.R(0,0), result.R(0,1), result.R(0,2),
                    result.R(1,0), result.R(1,1), result.R(1,2),
                    result.R(2,0), result.R(2,1), result.R(2,2));
                cv::Rodrigues(R_cv, rvec);
                cv::Mat tvec = (cv::Mat_<double>(3,1) << result.t(0), result.t(1), result.t(2));
                cv::projectPoints(axis_pts, rvec, tvec, K_cv, cv::Mat(), img_pts);
                if (img_pts.size() == 4) {
                    cv::line(vis, img_pts[0], img_pts[1], cv::Scalar(0, 0, 255), 3);
                    cv::line(vis, img_pts[0], img_pts[2], cv::Scalar(0, 255, 0), 3);
                    cv::line(vis, img_pts[0], img_pts[3], cv::Scalar(255, 0, 0), 3);
                }
            }
            std::string mono_path = output_dir_ + "/mono_f" +
                std::to_string(state_.frame_count) + ".png";
            cv::imwrite(mono_path, vis);
            std::cout << "[Mono] Visualization saved: " << mono_path << std::endl;
        }
    }

    state_.frame_count++;
    return result;
}

// ============================================================================
// ROI 辅助函数 —— 校验、裁剪、坐标偏移
// ============================================================================

RoiRect StereoTracker::validateRoi(const RoiRect* roi, const cv::Size& img_size,
                                     const std::string& name) {
    if (roi == nullptr || !roi->valid()) return RoiRect{};
    int x = roi->x, y = roi->y, w = roi->width, h = roi->height;
    if (x < 0 || y < 0 || w <= 0 || h <= 0)
        throw std::invalid_argument(name + " invalid: x=" + std::to_string(x) +
            ",y=" + std::to_string(y) + ",w=" + std::to_string(w) + ",h=" + std::to_string(h));
    if (x + w > img_size.width || y + h > img_size.height)
        throw std::invalid_argument(name + " out of bounds (" +
            std::to_string(img_size.width) + "x" + std::to_string(img_size.height) + "): " +
            std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
    return RoiRect{x, y, w, h};
}

void StereoTracker::offsetResultToOriginal(PipelineResult& result,
                                             const cv::Point2d& left_offset,
                                             const cv::Point2d& right_offset,
                                             const cv::Mat& left_color_orig,
                                             const cv::Mat& right_color_orig) {
    if (left_offset.x == 0.0 && left_offset.y == 0.0 &&
        right_offset.x == 0.0 && right_offset.y == 0.0)
        return;

    double lx = left_offset.x, ly = left_offset.y;
    double rx = right_offset.x, ry = right_offset.y;

    // Offset keypoints
    for (auto& kp : result.kp_left) {
        kp.pt.x += static_cast<float>(lx);
        kp.pt.y += static_cast<float>(ly);
    }

    // Offset left-image coordinate arrays
    auto offset_left = [lx, ly](std::vector<cv::Point2f>& arr) {
        for (auto& p : arr) { p.x += static_cast<float>(lx); p.y += static_cast<float>(ly); }
    };
    offset_left(result.pts_left_good);
    offset_left(result.pts_left_used);
    offset_left(result.pts_left_match);
    offset_left(result.pts_right_projected);

    // Offset right-image coordinate arrays
    auto offset_right = [rx, ry](std::vector<cv::Point2f>& arr) {
        for (auto& p : arr) { p.x += static_cast<float>(rx); p.y += static_cast<float>(ry); }
    };
    offset_right(result.pts_right_good);
    offset_right(result.pts_right_used);

    // Restore original full-size color images
    result.left_color = left_color_orig;
    result.right_color = right_color_orig;
    result.left_roi_offset_x = static_cast<int>(lx);
    result.left_roi_offset_y = static_cast<int>(ly);
    result.right_roi_offset_x = static_cast<int>(rx);
    result.right_roi_offset_y = static_cast<int>(ry);
}

// ============================================================================
// 图像加载 —— 分离彩色图和灰度图
// ============================================================================

std::pair<cv::Mat, cv::Mat> StereoTracker::loadImage(const cv::Mat& img) {
    if (img.empty()) return {cv::Mat(), cv::Mat()};
    cv::Mat color, gray;
    if (img.channels() == 3) {
        color = img.clone();
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
        gray = img.clone();
    } else {
        throw std::runtime_error("Unsupported image: " + std::to_string(img.channels()) + " channels");
    }
    return {color, gray};
}

// ============================================================================
// 日志记录 —— 保存每帧的处理数据，支持表格化输出
// ============================================================================

const std::vector<LogEntry>& StereoTracker::getLogs() const { return state_.logs; }

void StereoTracker::addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used) {
    double total_time = result.total_time_ms();
    double disp_median = 0.0;
    if (!result.disparity.empty()) {
        std::vector<double> abs_disp;
        for (double d : result.disparity) abs_disp.push_back(std::abs(d));
        disp_median = computeMedian(std::move(abs_disp));
    }
    LogEntry entry;
    entry.frame = state_.frame_count;
    entry.timestamp = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count()) / 1e9;
    entry.is_first = is_first;
    entry.fallback_used = fallback_used;
    entry.n_kp_left = result.n_kp_left;
    entry.n_matched = static_cast<int>(result.pts_left_good.size());
    entry.n_projected = static_cast<int>(result.pts_right_projected.size());
    entry.n_template_match = result.n_template_match;
    entry.gpnp_success = result.gpnp_success;
    entry.gpnp_n_pts = result.gpnp_n_pts;
    entry.disparity_median = disp_median;
    entry.total_time_ms = total_time;
    entry.timing = result.timing;
    state_.logs.push_back(std::move(entry));
}

void StereoTracker::printLogs() const {
    const auto& logs = state_.logs;
    if (logs.empty()) { std::cout << "[Log is empty]" << std::endl; return; }

    const std::vector<std::string> timing_keys = {"akaze", "flow", "filter", "proj", "match_template", "gpnp"};
    const std::map<std::string, std::string> timing_labels = {
        {"akaze", "AKAZE"}, {"flow", "OptFlow"}, {"filter", "Filter"},
        {"proj", "Proj"}, {"match_template", "Match"}, {"gpnp", "PnP"}};

    std::vector<std::string> used_keys;
    for (const auto& k : timing_keys)
        for (const auto& log : logs)
            if (auto it = log.timing.find(k); it != log.timing.end() && it->second > 0.0)
                { used_keys.push_back(k); break; }

    std::vector<size_t> widths = {5, 12, 8, 8, 8, 10, 8, 10, 10};
    for (const auto& k : used_keys) widths.push_back(10);

    auto print_sep = [&](char c) {
        std::cout << "+";
        for (auto w : widths) std::cout << std::string(w, c) << "+";
        std::cout << std::endl;
    };
    auto print_row = [&](const std::vector<std::string>& cols) {
        std::cout << "|";
        for (size_t i = 0; i < cols.size() && i < widths.size(); ++i)
            std::cout << std::setw(static_cast<int>(widths[i])) << cols[i] << "|";
        std::cout << std::endl;
    };

    print_sep('-');
    std::vector<std::string> header = {"Fr#", "Timestamp", "1st", "FB", "nKp", "nMatch",
                                        "nProj", "nTmpl", "Disp(px)", "PnP"};
    for (const auto& k : used_keys) {
        auto it = timing_labels.find(k);
        header.push_back(it != timing_labels.end() ? it->second : k);
    }
    print_row(header);
    print_sep('-');

    for (const auto& log : logs) {
        std::vector<std::string> row;
        row.push_back(std::to_string(log.frame));
        std::ostringstream ts; ts << std::fixed << std::setprecision(3) << log.timestamp;
        row.push_back(ts.str());
        row.push_back(log.is_first ? "Y" : "N");
        row.push_back(log.fallback_used ? "Y" : "N");
        row.push_back(std::to_string(log.n_kp_left));
        row.push_back(std::to_string(log.n_matched));
        row.push_back(std::to_string(log.n_projected));
        row.push_back(std::to_string(log.n_template_match));
        std::ostringstream ds; ds << std::fixed << std::setprecision(1) << log.disparity_median;
        row.push_back(ds.str());
        row.push_back(log.gpnp_success ? "OK" : "FAIL");
        for (const auto& k : used_keys) {
            auto it = log.timing.find(k);
            std::ostringstream ts2;
            ts2 << std::fixed << std::setprecision(1)
                << (it != log.timing.end() ? it->second : 0.0);
            row.push_back(ts2.str());
        }
        print_row(row);
    }
    print_sep('-');
}

} // namespace gpnp
