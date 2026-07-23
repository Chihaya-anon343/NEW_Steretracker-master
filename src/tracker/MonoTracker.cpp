/**
 * @file MonoTracker.cpp
 * @brief 单目追踪器实现。
 */

#include "tracker/MonoTracker.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace gpnp {

// ============================================================
// 构造
// ============================================================

MonoTracker::MonoTracker(const Eigen::Matrix3d& K,
                           const std::string& template_path,
                           const TrackerConfig& config,
                           const BinaryCornerExtractor::Config& binary_cfg,
                           const std::string& binary_template_dir,
                           const TinyTargetExtractor::Config& tiny_cfg,
                           const std::string& tiny_template_dir)
    : mono_pnp_()
{
    Eigen::Matrix3d R_rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rl(0, 0, 0);  // 单目无基线
    camera_ = makeStereoCameraParams(K, R_rl, t_rl);
    config_ = config;

    initExtractors(template_path, binary_cfg, binary_template_dir, tiny_cfg, tiny_template_dir);

    // 双ROI专用 AKAZE 提取器
    LKParams dual_lk;
    dual_lk.winSize = cv::Size(15, 15);
    dual_lk.maxLevel = 2;
    dual_akaze_extractor_ = std::make_unique<AkazeGpnpExtractor>(config.dual_roi_akaze_scale, dual_lk);
    dual_akaze_extractor_->initCamera(camera_);
    dual_akaze_extractor_->setTemplateData(template_path,
                                            config.template_real_width_mm,
                                            config.template_real_height_mm);
}

// ============================================================
// prepareDualBcTemplate
// ============================================================

void MonoTracker::prepareDualBcTemplate() {
    if (dual_bc_template_ready_) return;

    const cv::Mat& tmpl_img = akaze_extractor_->templateData().gray_image;
    if (tmpl_img.empty()) {
        std::cerr << "[DualRoi] AKAZE template image empty, cannot prepare BC template" << std::endl;
        dual_bc_template_ready_ = true;
        return;
    }

    int tw = tmpl_img.cols, th = tmpl_img.rows;

    cv::Mat binary;
    cv::threshold(tmpl_img, binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

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

    cv::Point2f center(tw / 2.0f, th / 2.0f);
    auto order = BinaryCornerExtractor::reorderByGeometry(corners, center, 0.0, -1.0);
    dual_bc_tmpl_corners_.reserve(order.size());
    for (int idx : order) dual_bc_tmpl_corners_.push_back(corners[idx]);

    double real_w = config_.template_real_width_mm;
    double real_h = config_.template_real_height_mm;
    dual_bc_tmpl_pts3d_.reserve(dual_bc_tmpl_corners_.size());
    for (const auto& c : dual_bc_tmpl_corners_) {
        dual_bc_tmpl_pts3d_.emplace_back(
            c.x / static_cast<double>(tw) * real_w,
            c.y / static_cast<double>(th) * real_h, 0.0);
    }

    dual_bc_template_ready_ = true;
    if (verbose_console_)
        std::cout << "[DualRoi] BC template prepared: " << dual_bc_tmpl_corners_.size()
                  << " corners on AKAZE template (" << tw << "x" << th << ")"
                  << "  real_size=" << real_w << "x" << real_h << "mm"
                  << std::endl;
}

// ============================================================
// MONO DUAL-ROI METHODS INJECTION POINT
// (will be appended below via bash)
// ============================================================
PipelineResult MonoTracker::processDualRoi(const cv::Mat& left_img,
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

    if (verbose_console_)
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

    // 3+4. BC + AK extraction in parallel (mono, independent extractors, distinct image regions)
    auto fut_bc = std::async(std::launch::async, [&]() {
        return binary_extractor_->extractMono(left_c0_gray, left_c0_color);
    });
    auto fut_ak = std::async(std::launch::async, [&]() {
        return dual_akaze_extractor_->extractMono(left_c1_gray, left_c1_color);
    });

    PipelineResult result_bc = fut_bc.get();
    int n_bc = static_cast<int>(result_bc.pts_left_match.size());
    if (verbose_console_) std::cout << "[DualRoi][Mono] BinaryCorner on class 0: " << n_bc << " corners" << std::endl;

    PipelineResult result_ak = fut_ak.get();
    int m_ak_match = static_cast<int>(result_ak.pts_left_match.size());
    if (verbose_console_)
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
        if (verbose_console_)
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
    if (verbose_console_)
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
    result.strategy_name = "DualRoi";

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

        if (verbose_console_)
            std::cout << "  [DualRoi][Mono] Visualized: " << n_bc_use << " BC + "
                      << (total_use - n_bc_use) << " AK corners" << std::endl;
    }

    if (verbose_console_)
        std::cout << "[DualRoi][Mono] Frame done: n_pts=" << total_use
                  << "  PnP=" << (pose.success ? "OK" : "FAIL")
                  << std::endl;

    return result;
}

void StereoTracker::clearCache() { state_ = TrackingState{}; }

// ============================================================================
// 单目模式 —— 仅左图特征提取 + PnP 解算
// ============================================================================

PipelineResult MonoTracker::process(const cv::Mat& left_img,
                                           bool visualize,
                                           const RoiGroup* left_group) {
    PipelineResult result;
    result.success = false;

        std::cerr << "[Mono] mono mode not enabled, set MonoConfig::enabled=true" << std::endl;
        return result;
    }

    if (left_img.empty()) {
        std::cerr << "[Mono] empty left image" << std::endl;
        return result;
    }

    // ---- Dual-ROI dispatch ----
    if (left_group && left_group->is_dual) {
        result = processDualRoi(left_img, *left_group, visualize);
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
    if (verbose_console_)
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

        if (verbose_console_) std::cout << "[Mono] Trying extractor: " << ext->name() << std::endl;

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
            if (verbose_console_)
                std::cout << "[Mono] Extractor " << ext->name()
                          << " succeeded, n_kp=" << result.n_kp_left << std::endl;
            break;
        }

        if (verbose_console_) std::cout << "[Mono] Extractor " << ext->name() << " failed, degrading..." << std::endl;
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

    result.strategy_name = winning_ext ? winning_ext->name() : "Unknown";
    result.success = pose.success;
    addLogEntry(result, is_first, false);

    // 可视化 — 原图 + 特征点 + 三维坐标轴
    if (visualize && !output_dir_.empty()) {
        cv::Mat vis = left_color.clone();

        // 绘制匹配特征点
        for (const auto& pt : result.pts_left_match) {
            cv::drawMarker(vis, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 1, 1);
        }

        // 绘制三维坐标轴
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
        if (verbose_console_) std::cout << "[Mono] Visualization saved: " << mono_path << std::endl;
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


} // namespace gpnp
