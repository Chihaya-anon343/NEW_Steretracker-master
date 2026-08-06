/**
 * @file MonoTracker.cpp
 * @brief 单目追踪器实现。
 */

#include "tracker/MonoTracker.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "utils/AsyncImageSaver.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <future>
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
// processDualRoi
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
        std::string prefix = "_f" + std::to_string(current_frame_);

        const cv::Scalar BC_COLOR(0, 0, 255);    // red: BinaryCorner corners
        const cv::Scalar AK_COLOR(0, 255, 0);    // green: AKAZE features

        // ---- 诊断面板 (overview/corners/reproj) 仅 Debug 模式 ----
        if (visualize_detailed_) {

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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_overview" + prefix + ".png", p0);
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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_corners" + prefix + ".png", p1);
        }

        } // end visualize_detailed_ (overview/corners)

        // Panel 2: 3D axes on original image (始终生成)
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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_axes" + prefix + ".png", p2);
        }

        // ---- 诊断面板 (reproj) 仅 Debug 模式 ----
        if (visualize_detailed_) {

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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_reproj" + prefix + ".png", p3);
        }

        } // end visualize_detailed_ dual-ROI panels

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

PipelineResult MonoTracker::process(const cv::Mat& left_img,
                                           bool visualize,
                                           const RoiGroup* left_group) {
    PipelineResult result;
    result.success = false;


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

    // 根据输入类别选择 BC/TT 的 3D 模板尺寸 并标记用于策略链选择
    // is_class1=true：仅检测到 class1（近距离回退），使用 class1 尺寸 + class1 阈值
    bool use_c1 = (left_group != nullptr && left_group->is_class1);
    {
        binary_extractor_->setUseClass1(use_c1);
        tiny_extractor_->setUseClass1(use_c1);
    }

    // 策略链选择（class1 时使用 class1 专用面积阈值）
    configureStrategyChain(roi_area, use_c1);

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
        result.is_class1 = use_c1;
        addLogEntry(result, is_first, true);
        return result;
    }

    // 单目 PnP 解算（EPnP，无 GPNP / warm-start）
    // winners_ext 有效的 extractor 一定有自己的 pts_3d（构造时已初始化）
    const auto& pnp_pts_3d = winning_ext->templateData().pts_3d;
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
    result.is_class1 = use_c1;
    addLogEntry(result, is_first, false);

    // ---- Visualization ----
    if (visualize && !output_dir_.empty()) {
        std::string prefix = "_f" + std::to_string(current_frame_);
        std::string strategy = result.strategy_name;
        bool is_bc   = (strategy == "BinaryCorner");
        bool is_tiny = (strategy == "TinyTarget");

        // ---- Shared helpers ----
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
        auto toView_L = [&](const cv::Point2f& p) {
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

        // Pre-build camera matrices for axis projection (reused across panels)
        cv::Mat K_cv = (cv::Mat_<double>(3,3) <<
            camera_.K(0,0), camera_.K(0,1), camera_.K(0,2),
            camera_.K(1,0), camera_.K(1,1), camera_.K(1,2),
            camera_.K(2,0), camera_.K(2,1), camera_.K(2,2));
        cv::Mat rvec, tvec_cv;
        if (result.success) {
            cv::Mat R_cv = (cv::Mat_<double>(3,3) <<
                result.R(0,0), result.R(0,1), result.R(0,2),
                result.R(1,0), result.R(1,1), result.R(1,2),
                result.R(2,0), result.R(2,1), result.R(2,2));
            cv::Rodrigues(R_cv, rvec);
            tvec_cv = (cv::Mat_<double>(3,1) << result.t(0), result.t(1), result.t(2));
        }

        // ================================================================
        // Panel: Overview (always) — full image + visible markers + 3D axes
        // ================================================================
        {
            cv::Mat vis = left_color.clone();
            for (const auto& pt : result.pts_left_match) {
                cv::drawMarker(vis, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 8, 2);
            }
            if (result.success) {
                std::vector<cv::Point3d> axis_pts = {{0,0,0}, {100,0,0}, {0,100,0}, {0,0,100}};
                std::vector<cv::Point2d> img_pts;
                cv::projectPoints(axis_pts, rvec, tvec_cv, K_cv, cv::Mat(), img_pts);
                if (img_pts.size() == 4) {
                    cv::line(vis, img_pts[0], img_pts[1], cv::Scalar(0, 0, 255), 3);
                    cv::line(vis, img_pts[0], img_pts[2], cv::Scalar(0, 255, 0), 3);
                    cv::line(vis, img_pts[0], img_pts[3], cv::Scalar(255, 0, 0), 3);
                }
            }
            utils::AsyncImageSaver::write(output_dir_ + "/mono_f"
                + std::to_string(current_frame_) + ".png", vis);
        }

        // ================================================================
        // 以下 per-strategy 诊断面板仅在 Debug 模式 (visualize_detailed_) 生成
        // Normal 模式只输出 Overview 三维轴叠加图
        // ================================================================
        if (visualize_detailed_) {

        // ================================================================
        // Panel: 3D Axes on expanded view (all strategies, when pose OK)
        // ================================================================
        if (result.success) {
            cv::Mat p_axes = view_L.clone();
            double axis_len = 100.0;
            Eigen::Vector3d o  = result.R * Eigen::Vector3d(0,0,0) + result.t;
            Eigen::Vector3d ax = result.R * Eigen::Vector3d(axis_len,0,0) + result.t;
            Eigen::Vector3d ay = result.R * Eigen::Vector3d(0,axis_len,0) + result.t;
            Eigen::Vector3d az = result.R * Eigen::Vector3d(0,0,axis_len) + result.t;
            cv::Point o_p  = proj(o);  o_p.x  -= expand_L.x; o_p.y  -= expand_L.y;
            cv::Point ax_p = proj(ax); ax_p.x -= expand_L.x; ax_p.y -= expand_L.y;
            cv::Point ay_p = proj(ay); ay_p.x -= expand_L.x; ay_p.y -= expand_L.y;
            cv::Point az_p = proj(az); az_p.x -= expand_L.x; az_p.y -= expand_L.y;
            cv::line(p_axes, o_p, ax_p, cv::Scalar(0,0,255), 2, cv::LINE_AA);
            cv::line(p_axes, o_p, ay_p, cv::Scalar(0,255,0), 2, cv::LINE_AA);
            cv::line(p_axes, o_p, az_p, cv::Scalar(255,0,0), 2, cv::LINE_AA);
            // Draw matched points on axes panel
            for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                cv::Point2f pv = toView_L(result.pts_left_match[i]);
                cv::circle(p_axes,
                    cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                    4, CORNER_COLORS[i % 10], -1);
            }
            std::string axes_name = is_bc   ? "/mono_bc_axes" + prefix + ".png" :
                                    is_tiny ? "/mono_tt_axes" + prefix + ".png"
                                            : "/mono_ak_axes" + prefix + ".png";
            utils::AsyncImageSaver::write(output_dir_ + axes_name, p_axes);
        }

        // ================================================================
        // BinaryCorner-specific panels
        // ================================================================
        if (is_bc) {
            auto* bce = binary_extractor_.get();
            float lx_roi = static_cast<float>(left_offset.x);
            float ly_roi = static_cast<float>(left_offset.y);

            // -- Panel: Binary image (left only) with corner overlay --
            if (bce && !bce->lastLeftBinary().empty()) {
                cv::Mat bl_bgr;
                cv::cvtColor(bce->lastLeftBinary(), bl_bgr, cv::COLOR_GRAY2BGR);
                for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                    cv::Point p(
                        static_cast<int>(result.pts_left_match[i].x - lx_roi),
                        static_cast<int>(result.pts_left_match[i].y - ly_roi));
                    cv::circle(bl_bgr, p, 4, CORNER_COLORS[i % 10], -1);
                }
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_binary" + prefix + ".png", bl_bgr);
            }

            // -- Panel: Upright (rotated-back) binary --
            if (bce && !bce->lastUprightBinary().empty()) {
                cv::Mat up;
                cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_upright" + prefix + ".png", up);
            }

            // -- Panel: Template correspondence (left-view | matched-template) --
            {
                cv::Mat p_tmpl_l = view_L.clone();
                for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                    cv::Point2f pv = toView_L(result.pts_left_match[i]);
                    cv::circle(p_tmpl_l,
                        cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                        4, CORNER_COLORS[i % 10], -1);
                }
                const TemplateData* matched_tmpl = bce ? bce->lastMatchedTemplate() : nullptr;
                cv::Mat p_tmpl_r;
                if (matched_tmpl && !matched_tmpl->image.empty()) {
                    cv::cvtColor(matched_tmpl->image, p_tmpl_r, cv::COLOR_GRAY2BGR);
                } else {
                    p_tmpl_r = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128,128,128));
                }
                cv::resize(p_tmpl_r, p_tmpl_r, p_tmpl_l.size(), 0, 0, cv::INTER_NEAREST);
                if (matched_tmpl) {
                    double dsx = static_cast<double>(p_tmpl_r.cols)
                               / std::max(1, matched_tmpl->image.cols);
                    double dsy = static_cast<double>(p_tmpl_r.rows)
                               / std::max(1, matched_tmpl->image.rows);
                    for (const auto& c : matched_tmpl->corners) {
                        cv::circle(p_tmpl_r,
                            cv::Point(static_cast<int>(c.x * dsx),
                                      static_cast<int>(c.y * dsy)),
                            4, CORNER_COLORS[0], 1);
                    }
                }
                cv::Mat p_tmpl;
                cv::hconcat(p_tmpl_l, p_tmpl_r, p_tmpl);
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_template" + prefix + ".png", p_tmpl);
            }

            // -- Panel: Reprojection error on expanded left view --
            {
                const auto& pnp_pts_3d = winning_ext->templateData().pts_3d;
                if (!pnp_pts_3d.empty()) {
                    cv::Mat p_reproj = view_L.clone();
                    size_t n_pts = std::min(result.pts_left_match.size(), pnp_pts_3d.size());
                    for (size_t i = 0; i < n_pts; ++i) {
                        cv::Point2f obs_v = toView_L(result.pts_left_match[i]);
                        cv::Point po(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                        if (result.success) {
                            Eigen::Vector3d P_cam = result.R * pnp_pts_3d[i] + result.t;
                            if (std::abs(P_cam.z()) >= 1e-6) {
                                Eigen::Vector2d uv = projectToImage(P_cam, camera_.K);
                                cv::Point pd(static_cast<int>(uv.x() - expand_L.x),
                                            static_cast<int>(uv.y() - expand_L.y));
                                cv::circle(p_reproj, pd, 3, cv::Scalar(0, 255, 0), -1);
                                cv::circle(p_reproj, po, 5, CORNER_COLORS[i % 10], 1);
                                cv::line(p_reproj, pd, po,
                                         cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                            }
                        } else {
                            cv::circle(p_reproj, po, 4, CORNER_COLORS[i % 10], -1);
                        }
                    }
                    utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_reproj" + prefix + ".png", p_reproj);
                }
            }
        }

        // ================================================================
        // TinyTarget-specific panels
        // ================================================================
        if (is_tiny) {
            const auto& pnp_pts_3d = winning_ext->templateData().pts_3d;

            // -- Panel: Reprojection error --
            if (!pnp_pts_3d.empty()) {
                cv::Mat p_reproj = view_L.clone();
                size_t n_pts = std::min(result.pts_left_match.size(), pnp_pts_3d.size());
                for (size_t i = 0; i < n_pts; ++i) {
                    cv::Point2f obs_v = toView_L(result.pts_left_match[i]);
                    cv::Point po(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    if (result.success) {
                        Eigen::Vector3d P_cam = result.R * pnp_pts_3d[i] + result.t;
                        if (std::abs(P_cam.z()) >= 1e-6) {
                            Eigen::Vector2d uv = projectToImage(P_cam, camera_.K);
                            cv::Point pd(static_cast<int>(uv.x() - expand_L.x),
                                        static_cast<int>(uv.y() - expand_L.y));
                            cv::circle(p_reproj, pd, 3, cv::Scalar(0, 255, 0), -1);
                            cv::circle(p_reproj, po, 5, CORNER_COLORS[i % 10], 1);
                            cv::line(p_reproj, pd, po,
                                     cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                        }
                    } else {
                        cv::circle(p_reproj, po, 4, CORNER_COLORS[i % 10], -1);
                    }
                }
                utils::AsyncImageSaver::write(output_dir_ + "/mono_tt_reproj" + prefix + ".png", p_reproj);
            }
        }

        // ================================================================
        // AKAZE-specific panels
        // ================================================================
        if (!is_bc && !is_tiny) {
            // -- Panel: Matched feature points on expanded view --
            if (!result.pts_left_match.empty()) {
                cv::Mat p_match = view_L.clone();
                for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                    cv::Point2f pv = toView_L(result.pts_left_match[i]);
                    cv::circle(p_match,
                        cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                        4, CORNER_COLORS[i % 10], -1);
                }
                utils::AsyncImageSaver::write(output_dir_ + "/mono_ak_matches" + prefix + ".png", p_match);
            }
        }

        } // end visualize_detailed_ per-strategy panels

        if (verbose_console_)
            std::cout << "[Mono] Visualization saved: " << output_dir_
                      << " (strategy=" << strategy << ")" << std::endl;
    }

    state_.frame_count++;
    return result;
}

} // namespace gpnp
