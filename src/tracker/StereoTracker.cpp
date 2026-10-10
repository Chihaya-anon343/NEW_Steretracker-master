/**
 * @file StereoTracker.cpp
 * @brief StereoTracker 双 ROI 策略实现。
 */

#include "tracker/StereoTracker.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "utils/AsyncImageSaver.hpp"
#include "utils/PoseUtils.hpp"

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
#include <future>

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
    : initial_pnp_(config.gpnp_min_pts)
    , gpnp_solver_(camera_, config.gpnp_min_pts)
{
    camera_ = makeStereoCameraParams(K, R_rl, t_rl);
    config_ = config;

    initExtractors(template_path, binary_cfg, binary_template_dir, tiny_cfg, tiny_template_dir);
}

StereoTracker::~StereoTracker() = default;

// ============================================================================
// 策略链配置 —— 根据 ROI 面积选择活跃的主提取器和退化链（仅修改指针，零开销）
// ============================================================================


// ============================================================================
// ROI padding —— 非 AKAZE 策略时扩大 ROI 给角点提取提供周围上下文
// ============================================================================

void StereoTracker::applyRoiPadding(RoiRect& rl, RoiRect& rr, int roi_area,
                                    int left_cols, int left_rows,
                                    int right_cols, int right_rows) const {
    // AKAZE 链停用后所有非零面积均走 BC/TT，均需角点上下文 padding
    if (roi_area == 0)
        return;

    int pad = (roi_area <= tiny_max_area_) ? tiny_roi_pad_ : binary_roi_pad_;
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

    // Full-image disparity（BC/TT 立体角点对）
    if (!result.pts_left_good.empty() && !result.pts_right_good.empty()) {
        int n = std::min(static_cast<int>(result.pts_left_good.size()),
                         static_cast<int>(result.pts_right_good.size()));
        result.disparity.resize(n);
        result.dx_filtered.resize(n);
        for (int i = 0; i < n; ++i) {
            double d = static_cast<double>(result.pts_left_good[i].x -
                                            result.pts_right_good[i].x);
            result.dx_filtered[i] = d;
            result.disparity[i] = d;  // standard rectified-stereo convention: left.x - right.x
        }
        if (n > 0) {
            if (verbose_console_)
                std::cout << "  [" << ext.name() << "] Full-image stereo: " << n
                          << " pairs, median_disp=" << computeMedian(result.disparity) << " px"
                          << "\n";
        }
    }

    // Determine if extraction succeeded
    return !result.pts_left_match.empty();  // BinaryCorner/TinyTarget: need extracted corners
}

// ============================================================================
// PnP Helper: BinaryCorner path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runBinaryCornerPnP(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d) {
    return solveBcPnpChain(result, pnp_pts_3d);
}

// BC 策略的 PnP 链：InitialPnP → GPnP warm-start → GPnP 失败回退 InitialPnP
// → InitialPnP 失败用视差中值估深度做初值。
// Dual-ROI Tier1(BC class0) 与 class1 链的 BC 段复用此链。
std::pair<bool, PoseEstimate> StereoTracker::solveBcPnpChain(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d) {
    PoseEstimate pose;
    double gpnp_timing = 0.0;

    if (config_.use_initial_pnp) {
        // 每帧始终运行 InitialPnP（几何一致初值），避免 GPnP HYBRID 路径复用上帧缓存旋转
        MatchResult match_res;
        match_res.good_matches       = result.good_matches;
        match_res.pts_left_match     = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;
        auto t_init_pnp = std::chrono::high_resolution_clock::now();
        PoseEstimate init_pose = initial_pnp_.solve(match_res, pnp_pts_3d, camera_.K);
        result.timing["initial_pnp"] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_init_pnp).count();

        if (init_pose.success) {
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                // GPNP failed but InitialPnP succeeded → use InitialPnP (no degradation)
                if (verbose_console_) std::cout << "  [BinaryCorner] GPNP failed, falling back to InitialPnP result" << "\n";
                pose = init_pose;
                pose.success = true;
            }
        } else {
            // InitialPnP failed → fallback to depth-from-disparity
            if (verbose_console_) std::cout << "  [BinaryCorner] InitialPnP failed, estimating depth from disparity" << "\n";
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            double depth_from_disp = 500.0;
            if (!result.disparity.empty()) {
                std::vector<double> abs_disp;
                for (double d : result.disparity) abs_disp.push_back(std::abs(d));
                double med_disp = computeMedian(std::move(abs_disp));
                if (med_disp > 1.0) {
                    depth_from_disp = camera_.focal_length * camera_.baseline / med_disp;
                    depth_from_disp = std::clamp(depth_from_disp, 50.0, 5000.0);
                    if (verbose_console_)
                        std::cout << "  [BinaryCorner] Depth from disparity: " << static_cast<int>(depth_from_disp)
                                  << "mm (median_disp=" << static_cast<int>(med_disp) << "px)" << "\n";
                }
            }
            Eigen::Vector3d t_id(0, 0, depth_from_disp);
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
        }
    } else {
        if (verbose_console_) std::cout << "  [InitialPnP] Skipped (use_initial_pnp=false)" << "\n";
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
    }

    return {pose.success, pose};
}

// ============================================================================
// PnP Helper: TinyTarget path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runTinyTargetPnP(
        PipelineResult& result,
        const std::vector<Eigen::Vector3d>& pnp_pts_3d) {
    PoseEstimate pose;

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
    auto t_tiny_pnp = std::chrono::high_resolution_clock::now();
    bool pnp_ok = cv::solvePnP(obj_pts, img_pts, K_cv, cv::Mat(),
                                rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    result.timing["tiny_pnp"] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_tiny_pnp).count();

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
        if (verbose_console_)
            std::cout << "  [TinyTarget] solvePnP OK  n_pts=4  RE="
                      << (rpe_sum / 4.0) << "px" << "\n";
    } else {
        std::cerr << "  [TinyTarget] solvePnP FAILED" << "\n";
    }

    return {pose.success, pose};
}

// ============================================================================
// Finalize pose — merge into PipelineResult and update tracking state
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::dispatchPnP(FeatureExtractor* ext,
                                                           PipelineResult& result) {
    // 3D 模板按当前策略 ext 解析 (退化链中 ext 可不同于成员 extractor_)
    const auto& pnp_pts_3d = ext->templateData().pts_3d;
    switch (ext->strategyType()) {
        case StrategyType::BinaryCorner:
            return runBinaryCornerPnP(result, pnp_pts_3d);
        case StrategyType::TinyTarget:
            return runTinyTargetPnP(result, pnp_pts_3d);
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

    binary_extractor_->setDebugCapture(visualize);
    tiny_extractor_->setDebugCapture(visualize);

    // ---- Load images (P3: color 仅可视化需要; 灰度保持全图——无 ROI 全图回退路径依赖) ----
    auto [left_color, left_gray] = loadImage(left_img, visualize);
    auto [right_color, right_gray] = loadImage(right_img, visualize);
    if (left_gray.empty() || right_gray.empty())
        throw std::runtime_error("Cannot read input images");

    // ---- Save originals (for visualization & coordinate restore; P3: 非可视化共享不复制) ----
    cv::Mat left_color_orig  = visualize ? left_color.clone()  : left_color;
    cv::Mat right_color_orig = visualize ? right_color.clone() : right_color;

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

    // 无有效左 ROI → 帧失败（全图回退随 AKAZE 链一并移除；frame_count 已在入口自增）
    if (!roi_l.valid()) {
        if (verbose_console_)
            std::cout << "[Stereo] no valid left ROI, frame fails (Skip_NoRoi)" << "\n";
        PipelineResult fail;
        fail.strategy_name = "Skip_NoRoi";
        addLogEntry(fail, !state_.has_cache, false);
        return fail;
    }

    // 根据输入类别选择 BC/TT 的 3D 模板尺寸
    // is_class1=true：仅检测到 class1（近距离回退），使用 class1 尺寸参数
    bool use_c1 = left_grp.is_class1;
    {
        binary_extractor_->setUseClass1(use_c1);
        tiny_extractor_->setUseClass1(use_c1);
    }

    // Automatically select extraction strategy chain from ROI area
    int roi_area = roi_l.valid() ? roi_l.width * roi_l.height : 0;
    configureStrategyChain(roi_area, use_c1);

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
        if (!left_color.empty())
            left_color_cropped = left_color(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_offset = cv::Point2d(static_cast<double>(roi_l.x), static_cast<double>(roi_l.y));
    }
    if (roi_r.valid()) {
        right_cropped = right_gray(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        if (!right_color.empty())
            right_color_cropped = right_color(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_offset = cv::Point2d(static_cast<double>(roi_r.x), static_cast<double>(roi_r.y));
    }

    bool is_first = !state_.has_cache;
    PipelineResult result;
    std::string winning_strategy;
    FeatureExtractor* winner_ext = nullptr;
    bool fallback_used = false;
    bool pose_ok = false;
    PoseEstimate final_pose;
    double extract_ms = 0.0, pnp_ms = 0.0;   // 阶段耗时: 特征提取 / 位姿解算 (跨退化链累加)

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
            if (verbose_console_) std::cout << "[Degradation] " << from << " failed → " << strategy_name << "\n";
        }

        auto t_extract = std::chrono::steady_clock::now();
        bool extract_ok = runExtraction(*ext, left_cropped, right_cropped,
                                        left_color_cropped, right_color_cropped,
                                        left_offset, right_offset,
                                        left_color_orig, right_color_orig, result);
        extract_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_extract).count();

        if (!extract_ok) {
            if (is_primary) {
                if (verbose_console_)
                    std::cout << "[Degradation] " << strategy_name << " extraction failed"
                              << " (n_kp=" << result.n_kp_left << ")" << "\n";
            } else {
                if (verbose_console_)
                    std::cout << "[Degradation] " << strategy_name << " extraction failed" << "\n";
            }
            continue;
        }

        auto t_pnp = std::chrono::steady_clock::now();
        auto [ok, pose] = dispatchPnP(ext, result);
        pnp_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pnp).count();
        if (ok) {
            pose_ok = true;
            final_pose = pose;
            winning_strategy = strategy_name;
            winner_ext = ext;
            result.strategy_name = strategy_name;
            break;
        }
    }

    if (!pose_ok) {
        std::cerr << "[Degradation] All strategies failed for frame " << state_.frame_count << "\n";
    } else {
        // 帧末反馈: 锁定档位跟随实际胜者 (temporal 策略粘滞)
        updateStickinessFromWinner(winning_strategy);
        fillTemporalMeta(result);
    }

    // ---- Finalize pose (if successful) ----
    if (pose_ok) {
        finalizePose(result, final_pose);
        result.success = final_pose.success;  // 确保终端 FAILED / 三维轴绘制检查通过
    }

    // ---- Visualization ----
    if (visualize && pose_ok) {
        std::string prefix = "_f" + std::to_string(current_frame_);

        // Normal 模式 (visualize_detailed_=false): 仅三维坐标轴叠加图
        if (!visualize_detailed_) {
            cv::Mat vis = result.left_color.clone();
            for (const auto& pt : result.pts_left_match)
                cv::drawMarker(vis, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 1, 1);
            if (result.success) {
                if (verbose_console_)
                    std::cout << "  [Viz] Drawing 3D axes, R=" << result.R(0,0)
                              << " t=[" << result.t(0) << "," << result.t(1) << "," << result.t(2) << "]"
                              << "\n";
                cv::Mat K_cv = (cv::Mat_<double>(3,3) <<
                    camera_.K(0,0), camera_.K(0,1), camera_.K(0,2),
                    camera_.K(1,0), camera_.K(1,1), camera_.K(1,2),
                    camera_.K(2,0), camera_.K(2,1), camera_.K(2,2));
                cv::Mat rvec, R_cv = (cv::Mat_<double>(3,3) <<
                    result.R(0,0), result.R(0,1), result.R(0,2),
                    result.R(1,0), result.R(1,1), result.R(1,2),
                    result.R(2,0), result.R(2,1), result.R(2,2));
                cv::Rodrigues(R_cv, rvec);
                cv::Mat tvec = (cv::Mat_<double>(3,1) << result.t(0), result.t(1), result.t(2));
                std::vector<cv::Point3d> axis = {{0,0,0},{100,0,0},{0,100,0},{0,0,100}};
                std::vector<cv::Point2d> img_pts;
                cv::projectPoints(axis, rvec, tvec, K_cv, cv::Mat(), img_pts);
                if (img_pts.size() == 4) {
                    cv::line(vis, img_pts[0], img_pts[1], cv::Scalar(0,0,255), 3);
                    cv::line(vis, img_pts[0], img_pts[2], cv::Scalar(0,255,0), 3);
                    cv::line(vis, img_pts[0], img_pts[3], cv::Scalar(255,0,0), 3);
                }
            } else {
                if (verbose_console_)
                    std::cout << "  [Viz] No axes: result.success=false" << "\n";
            }
            utils::AsyncImageSaver::write(output_dir_ + "/stereo" + prefix + ".png", vis);
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

            const auto& pnp_pts_3d = winner_ext->templateData().pts_3d;
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
                    utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_binary" + prefix + ".png", p0);
                }
            }

            // --- Panel 0b: 旋转回正二值图 (BinaryCorner only) ---
            if (!is_tiny) {
                auto* bce = binary_extractor_.get();
                if (bce && !bce->lastUprightBinary().empty()) {
                    cv::Mat up;
                    cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                    utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_upright" + prefix + ".png", up);
                }
            }

            // --- Panel 0c: TinyTarget ROI 与二值化图 (各: 灰度 ROI | 清理前 Otsu | 清理后二值) ---
            if (is_tiny) {
                auto* tte = tiny_extractor_.get();
                float lx_roi = static_cast<float>(left_offset.x);
                float ly_roi = static_cast<float>(left_offset.y);
                float rx_roi = static_cast<float>(right_offset.x);
                float ry_roi = static_cast<float>(right_offset.y);
                auto buildTiles = [&](const TinyTargetExtractor::DebugImages& dbg,
                                      const std::vector<cv::Point2f>& pts_full,
                                      float ox, float oy) -> cv::Mat {
                    cv::Mat panel;
                    if (dbg.roi_gray.empty()) return panel;
                    auto drawPts = [&](cv::Mat& img, float k) {
                        for (size_t i = 0; i < pts_full.size(); ++i) {
                            cv::Point p(
                                static_cast<int>((pts_full[i].x - ox) * k),
                                static_cast<int>((pts_full[i].y - oy) * k));
                            cv::circle(img, p, 2, CORNER_COLORS[i % 10], -1);
                        }
                    };
                    std::vector<cv::Mat> tiles;
                    cv::Mat g;
                    cv::cvtColor(dbg.roi_gray, g, cv::COLOR_GRAY2BGR);
                    drawPts(g, 1.0f);
                    tiles.push_back(g);
                    // otsu/super 均为超分尺度：按各自宽度比绘制角点后缩回 ROI 尺寸
                    auto addTile = [&](const cv::Mat& bin) {
                        if (bin.empty()) return;
                        float k = static_cast<float>(bin.cols)
                                / std::max(1, dbg.roi_gray.cols);
                        cv::Mat t;
                        cv::cvtColor(bin, t, cv::COLOR_GRAY2BGR);
                        drawPts(t, k);
                        cv::resize(t, t, g.size(), 0, 0, cv::INTER_NEAREST);
                        tiles.push_back(t);
                    };
                    addTile(dbg.otsu_binary);
                    addTile(dbg.super_binary);
                    panel = tiles[0];
                    for (size_t i = 1; i < tiles.size(); ++i)
                        cv::hconcat(panel, tiles[i], panel);
                    return panel;
                };
                cv::Mat p_tl = buildTiles(tte->lastLeftDebug(),
                                          result.pts_left_match, lx_roi, ly_roi);
                cv::Mat p_tr = buildTiles(tte->lastRightDebug(),
                                          result.pts_right_good, rx_roi, ry_roi);
                if (!p_tl.empty())
                    utils::AsyncImageSaver::write(
                        output_dir_ + "/tiny_target_roi_binary_l" + prefix + ".png", p_tl);
                if (!p_tr.empty())
                    utils::AsyncImageSaver::write(
                        output_dir_ + "/tiny_target_roi_binary_r" + prefix + ".png", p_tr);
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
                utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_axes" + prefix + ".png", p1);
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
                utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_template" + prefix + ".png", p2);
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
                utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_stereo" + prefix + ".png", p3);
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
                utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_reproj" + prefix + ".png", p4);
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
                utils::AsyncImageSaver::write(output_dir_ + "/binary_corner_stereo_proj" + prefix + ".png", p5);
            }
        }
    }

    // ---- Logging ----
    result.is_first_frame = is_first;
    result.is_class1 = use_c1;
    result.n_matched = static_cast<int>(result.pts_left_good.size());
    result.n_projected = static_cast<int>(result.pts_right_projected.size());
    result.extract_ms = extract_ms;
    result.pnp_ms = pnp_ms;
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
// Dual-ROI Strategy: BinaryCorner (class 0 edges) + class1 退化链 (Tier3)
// ============================================================================
// prepareDualBcTemplate — 从板模板图提取 BC 角点（双目双ROI用）
// ============================================================================

void StereoTracker::prepareDualBcTemplate() {
    if (dual_bc_template_ready_) return;

    const cv::Mat& tmpl_img = template_.gray_image;
    if (tmpl_img.empty()) {
        std::cerr << "[DualRoi] board template image empty, cannot prepare BC template" << "\n";
        dual_bc_template_ready_ = true;
        return;
    }

    int tw = tmpl_img.cols, th = tmpl_img.rows;
    double real_w = config_.template_real_width_mm;
    double real_h = config_.template_real_height_mm;

    // ---- class0: 人工标注优先 (规范序: 中心=模板图中心, 与运行时 BC 约定一致) ----
    bool c0_from_anno = false;
    if (!config_.dual_roi_class0_points_path.empty()) {
        auto anno = readCorners(config_.dual_roi_class0_points_path);
        if (anno.size() >= 4) {
            cv::Point2f center(tw / 2.0f, th / 2.0f);
            auto order = BinaryCornerExtractor::reorderByGeometry(anno, center, 0.0, -1.0);
            dual_bc_tmpl_corners_.reserve(order.size());
            for (int idx : order) dual_bc_tmpl_corners_.push_back(anno[idx]);
            c0_from_anno = true;
        } else {
            std::cerr << "[DualRoi] class0 annotation unusable (" << anno.size()
                      << " pts): " << config_.dual_roi_class0_points_path
                      << ", falling back to approxPolyDP" << "\n";
        }
    }

    // ---- class0 回退: approxPolyDP 自动推导 (原逻辑) ----
    if (dual_bc_tmpl_corners_.empty()) {
        cv::Mat binary;
        cv::threshold(tmpl_img, binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) {
            std::cerr << "[DualRoi] No contours found in board template binary" << "\n";
            dual_bc_template_ready_ = true;
            return;
        }

        auto* largest = &contours[0];
        for (auto& c : contours)
            if (cv::contourArea(c) > cv::contourArea(*largest)) largest = &c;

        int target_n = binary_extractor_->lastCornersBeforeReorder().empty()
            ? 10 : static_cast<int>(binary_extractor_->lastCornersBeforeReorder().size());
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
            std::cerr << "[DualRoi] Failed to extract corners from board template" << "\n";
            dual_bc_template_ready_ = true;
            return;
        }

        cv::Point2f center(tw / 2.0f, th / 2.0f);
        auto order = BinaryCornerExtractor::reorderByGeometry(corners, center, 0.0, -1.0);
        dual_bc_tmpl_corners_.reserve(order.size());
        for (int idx : order) dual_bc_tmpl_corners_.push_back(corners[idx]);
    }

    // ---- class1: 仅人工标注 (中心=class1 点集质心), 无回退 ----
    if (!config_.dual_roi_class1_points_path.empty()) {
        auto anno = readCorners(config_.dual_roi_class1_points_path);
        if (anno.size() >= 4) {
            cv::Point2f ctr(0.0f, 0.0f);
            for (const auto& p : anno) { ctr.x += p.x; ctr.y += p.y; }
            ctr *= 1.0f / static_cast<float>(anno.size());
            auto order = BinaryCornerExtractor::reorderByGeometry(anno, ctr, 0.0, -1.0);
            dual_c1_tmpl_corners_.reserve(order.size());
            for (int idx : order) dual_c1_tmpl_corners_.push_back(anno[idx]);
        } else {
            std::cerr << "[DualRoi] class1 annotation unusable (" << anno.size()
                      << " pts): " << config_.dual_roi_class1_points_path
                      << ", Tier1b joint solve disabled" << "\n";
        }
    }

    // ---- 像素 → 板系 3D (两组共用同一换算) ----
    const double cx = real_w / 2.0, cy = real_h / 2.0;
    dual_bc_tmpl_pts3d_.reserve(dual_bc_tmpl_corners_.size());
    for (const auto& c : dual_bc_tmpl_corners_) {
        dual_bc_tmpl_pts3d_.emplace_back(
            c.x / static_cast<double>(tw) * real_w - cx,
            c.y / static_cast<double>(th) * real_h - cy, 0.0);
    }
    dual_c1_tmpl_pts3d_.reserve(dual_c1_tmpl_corners_.size());
    for (const auto& c : dual_c1_tmpl_corners_) {
        dual_c1_tmpl_pts3d_.emplace_back(
            c.x / static_cast<double>(tw) * real_w - cx,
            c.y / static_cast<double>(th) * real_h - cy, 0.0);
    }

    dual_bc_template_ready_ = true;
    if (verbose_console_)
        std::cout << "[DualRoi] BC template ready: c0=" << dual_bc_tmpl_corners_.size()
                  << " pts (" << (c0_from_anno ? "anno" : "auto")
                  << ")  c1=" << dual_c1_tmpl_corners_.size() << " pts"
                  << "  board=" << tw << "x" << th
                  << "  real_size=" << real_w << "x" << real_h << "mm"
                  << "\n";
}


PipelineResult StereoTracker::processDualRoi(const cv::Mat& left_img,
                                               const cv::Mat& right_img,
                                               const RoiGroup& left_group,
                                               const RoiGroup& right_group,
                                               bool visualize) {
    auto t_dual_start = std::chrono::steady_clock::now();
    double extract_ms = 0.0, pnp_ms = 0.0;   // 阶段耗时: 特征提取 / 位姿解算

    // 链路追踪: 逐级记录 Tier1/2/3 的结果与降级原因, 帧末输出 Route 汇总
    std::string route;
    auto routeStep = [&route](const std::string& s) {
        if (!route.empty()) route += " -> ";
        route += s;
    };
    std::string t1_reason;

    // 消融开关 (config: strategies.dual_roi.tier1_only): true = 屏蔽 Tier3(class1 链)/
    // 巨型 primary 短路, 仅保留 Tier1 (BC class0 解算)
    const bool kDualRoiTier1Only = config_.dual_roi_tier1_only;

    // BC 二值化过程条图 (成功/失败路径共用 —— 失败帧也要可视化, 排查主证据)
    auto saveBcProcessPanel = [&]() {
        const auto& stages = binary_extractor_->lastBinaryProcess();
        if (stages.empty()) return;
        const int panel_h = 240;   // 每面板统一高度, 宽度按纵横比
        std::vector<cv::Mat> panels;
        panels.reserve(stages.size());
        for (const auto& kv : stages) {
            cv::Mat m;
            if (kv.second.channels() == 1)
                cv::cvtColor(kv.second, m, cv::COLOR_GRAY2BGR);
            else
                m = kv.second.clone();
            double s = static_cast<double>(panel_h) / m.rows;
            cv::resize(m, m, cv::Size(), s, s, cv::INTER_NEAREST);
            cv::copyMakeBorder(m, m, 22, 0, 0, 0, cv::BORDER_CONSTANT,
                               cv::Scalar(40, 40, 40));
            cv::putText(m, kv.first, cv::Point(4, 16),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            panels.push_back(m);
        }
        cv::Mat strip;
        cv::hconcat(panels, strip);
        utils::AsyncImageSaver::write(
            output_dir_ + "/dual_roi_bc_process_f" + std::to_string(current_frame_) + ".png",
            strip);
    };

    // 0. Ensure template preprocessing is done
    prepareDualBcTemplate();

    // ---- Load images ----
    auto [left_color, left_gray] = loadImage(left_img, visualize);
    auto [right_color, right_gray] = loadImage(right_img, visualize);
    if (left_gray.empty() || right_gray.empty()) {
        PipelineResult empty;
        empty.is_first_frame = !state_.has_cache;
        return empty;
    }

    cv::Mat left_color_orig  = visualize ? left_color.clone() : left_color;
    cv::Mat right_color_orig = visualize ? right_color.clone() : right_color;

    int pad = config_.dual_roi_secondary_expand;

    bool is_first = !state_.has_cache;
    PoseSeed seed;
    const bool use_seed = seedActive();
    if (use_seed) { seed.R = state_.R_prev; seed.t = state_.t_prev; }

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

    if (verbose_console_)
        std::cout << "[DualRoi] pad=" << pad
                  << "  primary=" << left_pri.width << "x" << left_pri.height
                  << "  secondary(raw)=" << left_group.secondary.width << "x" << left_group.secondary.height
                  << "  secondary(expanded)=" << left_sec.width << "x" << left_sec.height
                  << "  offset=(" << sec_to_pri_offset.x << "," << sec_to_pri_offset.y << ")"
                  << "\n";

    // P1 巨型 primary 短路: 与 MonoTracker 同语义 —— primary 边长覆盖画幅 ≥
    // primary_span_ratio 时目标外角点已出视野, BC 角点必败 (Tier1 的 PnP 必失败)
    // 且全图级 BC 提取极耗时 → 跳过 Tier1 直接走 9c class1 链。Tier3 关闭时保持旧路径。
    const bool primary_spans_frame =
        !kDualRoiTier1Only &&   // 消融: 关闭短路, 巨型 primary 也走 Tier1
        config_.dual_roi_primary_span_ratio > 0.0 &&
        config_.dual_roi_class1_fallback &&
        left_pri.width  >= left_img.cols * config_.dual_roi_primary_span_ratio &&
        left_pri.height >= left_img.rows * config_.dual_roi_primary_span_ratio;

    if (primary_spans_frame && verbose_console_)
        std::cout << "[DualRoi] primary spans frame ("
                  << left_pri.width << "x" << left_pri.height << " vs "
                  << left_img.cols << "x" << left_img.rows
                  << "), skip Tier1 -> class1 chain" << "\n";
    if (primary_spans_frame)
        routeStep("SHORT-CIRCUIT(primary spans frame, skip Tier1)");

    // 2. Crop images
    cv::Mat left_c0_gray, right_c0_gray, left_c0_color, right_c0_color;
    if (!primary_spans_frame) {
        left_c0_gray   = left_gray(cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
        right_c0_gray  = right_gray(cv::Rect(right_pri.x, right_pri.y, right_pri.width, right_pri.height)).clone();
        if (!left_color.empty())
            left_c0_color = left_color(cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
        if (!right_color.empty())
            right_c0_color = right_color(cv::Rect(right_pri.x, right_pri.y, right_pri.width, right_pri.height)).clone();
    }

    cv::Mat left_c1_gray   = left_gray(cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height)).clone();
    cv::Mat right_c1_gray  = right_gray(cv::Rect(right_sec.x, right_sec.y, right_sec.width, right_sec.height)).clone();
    cv::Mat left_c1_color, right_c1_color;
    if (!left_color.empty())
        left_c1_color = left_color(cv::Rect(left_sec.x, left_sec.y, left_sec.width, left_sec.height)).clone();
    if (!right_color.empty())
        right_c1_color = right_color(cv::Rect(right_sec.x, right_sec.y, right_sec.width, right_sec.height)).clone();

    // ---- 下游共用变量（短路时保持空/失败初值, 9c 直接成为唯一路径）----
    PipelineResult result_bc;
    int n_bc = 0, n_bc_use = 0, bc_total = 0;
    int total_use = 0;
    std::vector<int> pair_tmpl_idx;   // 每个合并点对应的标注模板点索引 (c0: 0..n0-1, c1: n0..)
    bool joint_used = false;          // Tier1b 联合解算胜出
    std::vector<cv::Point2f> merged_pts_left, merged_pts_right, merged_pts_template;
    std::vector<cv::KeyPoint> merged_kp_left;
    std::vector<Eigen::Vector3d> merged_pts3d;
    std::vector<cv::DMatch> merged_matches;
    std::vector<int> merged_idx;
    cv::Point2d left_off(static_cast<double>(left_pri.x), static_cast<double>(left_pri.y));
    cv::Point2d right_off(static_cast<double>(right_pri.x), static_cast<double>(right_pri.y));
    PipelineResult result;
    PoseEstimate pose;

    if (!primary_spans_frame) {

    // 3. BC(class0) 提取 —— class1 部分不再用 AKAZE, 退化时由 Tier3 class1 链的 BC 接管
    auto t_extract = std::chrono::steady_clock::now();
    result_bc = binary_extractor_->extract(
        left_c0_gray, right_c0_gray, left_c0_color, right_c0_color);
    n_bc = static_cast<int>(result_bc.pts_left_match.size());
    if (verbose_console_) std::cout << "[DualRoi] BinaryCorner on class 0: " << n_bc << " corners" << "\n";
    extract_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_extract).count();

    // 4. BC 角点 ↔ 板系 3D 对应 (严格数量守卫, 与 MonoTracker 同语义):
    //    BC 输出已规范序 (matchCorners 模板/图像同步重排), 直接前缀索引;
    //    运行时角点数 ≠ 模板点数 → c0 侧整体剔除 (前缀 min 配对是错位对应)
    int n_bc_right = static_cast<int>(result_bc.pts_right_good.size());
    int n_bc_3d = static_cast<int>(dual_bc_tmpl_pts3d_.size());
    if (n_bc_3d > 0 && n_bc != n_bc_3d) {
        std::cerr << "[DualRoi] BC corner count mismatch (" << n_bc
                  << " vs template " << n_bc_3d << "), dropping c0 side" << "\n";
    }
    n_bc_use = (n_bc == n_bc_3d) ? std::min(n_bc, n_bc_right) : 0;

    for (int i = 0; i < n_bc_use; ++i) {
        merged_pts_left.push_back(result_bc.pts_left_match[i]);
        merged_pts_right.push_back(result_bc.pts_right_good[i]);
        merged_kp_left.emplace_back(result_bc.pts_left_match[i], 1.0f);
        pair_tmpl_idx.push_back(i);
    }
    // BC template corners scaled to class-0-ROI (already in BC's result)
    for (int i = 0; i < n_bc_use && i < static_cast<int>(result_bc.pts_template_match.size()); ++i) {
        merged_pts_template.push_back(result_bc.pts_template_match[i]);
    }
    // BC 3D points from board-template-derived corners
    for (int i = 0; i < n_bc_use && i < static_cast<int>(dual_bc_tmpl_pts3d_.size()); ++i) {
        merged_pts3d.push_back(dual_bc_tmpl_pts3d_[i]);
    }

    bc_total = n_bc_use;

    int total_pts = static_cast<int>(merged_pts_left.size());
    int total_right = static_cast<int>(merged_pts_right.size());
    int total_3d = static_cast<int>(merged_pts3d.size());

    // Sync counts: GPNP needs equal numbers
    total_use = std::min({total_pts, total_right, total_3d});
    merged_pts_left.resize(total_use);
    merged_pts_right.resize(total_use);
    merged_pts3d.resize(total_use);
    merged_kp_left.resize(total_use);
    pair_tmpl_idx.resize(total_use);

    // Build 1:1 good_matches + idx_from_filtered
    merged_matches.resize(total_use);
    merged_idx.resize(total_use);
    for (int i = 0; i < total_use; ++i) {
        merged_matches[i] = cv::DMatch(i, i, 0.0f); // queryIdx=i → trainIdx=i
        merged_idx[i] = i;
    }

    if (verbose_console_)
        std::cout << "[DualRoi] BC corners merged: " << total_use
                  << "  pts3d=" << total_3d << "\n";

    if (total_use < 4 && verbose_console_) {
        std::cout << "[DualRoi] Too few BC corners (" << total_use
                  << "), skipping Tier1, trying class1 chain" << "\n";
    }

    // 7. Build PipelineResult
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

    // 8. Restore full-image coordinates BEFORE PnP (PnP uses full-image K cx/cy)
    offsetResultToOriginal(result, left_off, right_off, left_color_orig, right_color_orig);

    // Fill disparity for logging (right.x - left.x, match AKAZE convention)
    {
        int n = std::min(static_cast<int>(result.pts_left_good.size()),
                         static_cast<int>(result.pts_right_good.size()));
        result.disparity.resize(n);
        result.dx_filtered.resize(n);
        for (int i = 0; i < n; ++i) {
            double d = static_cast<double>(result.pts_left_good[i].x -
                                           result.pts_right_good[i].x);
            result.dx_filtered[i] = d;
            result.disparity[i] = -d;  // = right.x - left.x
        }
        if (n > 0 && verbose_console_)
            std::cout << "  [DualRoi] Full-image stereo: " << n
                      << " pairs, median_disp=" << computeMedian(result.disparity) << " px"
                      << "\n";
    }

    // 9. Pose estimation
    double gpnp_timing = 0.0;
    auto t_pnp_start = std::chrono::high_resolution_clock::now();

    if (total_use < 4) {
        // 角点不足: 跳过 Tier1 解算, 由下方 class1 退化链兜底
        if (verbose_console_)
            std::cout << "  [DualRoi] Skip Tier1 PnP (" << total_use << " pts < 4)" << "\n";
        t1_reason = "pts<4";
    } else if (config_.use_initial_pnp) {
        // 每帧始终运行 InitialPnP（几何一致初值），避免 GPnP HYBRID 路径复用上帧缓存旋转
        MatchResult match_res;
        match_res.good_matches       = result.good_matches;
        match_res.pts_left_match     = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;

        PoseEstimate init_pose = initial_pnp_.solve(match_res, merged_pts3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        if (init_pose.success) {
            if (verbose_console_) std::cout << "  [DualRoi] InitialPnP OK, warm-starting GPNP" << "\n";
            pose = gpnp_solver_.solve(result, merged_pts3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                if (verbose_console_) std::cout << "  [DualRoi] GPNP failed, using InitialPnP result" << "\n";
                pose = init_pose;
                pose.success = true;
            }
        } else {
            if (verbose_console_) std::cout << "  [DualRoi] InitialPnP failed, trying GPNP with default depth" << "\n";
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t_id(0, 0, 5000);
            pose = gpnp_solver_.solve(result, merged_pts3d, &R_id, &t_id, gpnp_timing);
        }
    } else {
        if (verbose_console_) std::cout << "  [DualRoi] InitialPnP skipped" << "\n";
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_id(0, 0, 5000);
        pose = gpnp_solver_.solve(result, merged_pts3d, &R_id, &t_id, gpnp_timing);
    }

    // 运动门控 (与 MonoTracker::processDualRoi 同语义, 含陈旧放宽):
    // 解通过重投影校验但相对 seed 跳变超阈值 → 判失败, 交由退化链 (Tier3) 接管
    if (pose.success && use_seed &&
        !motionGatePass(pose, seed, nullptr, false, "DualRoi-T1", true)) {
        if (verbose_console_)
            std::cout << "  [DualRoi] Tier1 pose rejected by motion gate" << "\n";
        t1_reason = "motion-gate";
        pose = PoseEstimate{};
    }
    if (t1_reason.empty() && !pose.success) t1_reason = "pnp";
    routeStep(pose.success
        ? "Tier1(BC c0 n=" + std::to_string(bc_total) + ") OK"
        : "Tier1(BC c0 n=" + std::to_string(bc_total) + ") FAIL[" + t1_reason + "]");

    auto t_pnp_end = std::chrono::high_resolution_clock::now();
    result.timing["gpnp"] = std::chrono::duration<double, std::milli>(t_pnp_end - t_pnp_start).count();
    pnp_ms += result.timing["gpnp"];

    // ================================================================
    // 6b. Tier1b: class1 增广联合解算 (与 MonoTracker 同机制; 双目需左右成对)
    //     P0 投影全部标注 3D 点 (c0+c1), 与提取角点 (c0+c1, 左图) 贪心 NN 匹配
    //     (门限内, 双向唯一); 右图对应点按类内索引取各 class 提取结果的
    //     pts_right_good (BC 双目规范序 1:1), 右点缺失的对丢弃。
    //     匹配对多于 Tier1a 时 GPnP(P0 warm) 联合重解, 失败回退 InitialPnP+GPnP;
    //     失败时保留 Tier1a 位姿 (P0 仍成功, 不落入 Tier3)。
    // ================================================================
    if (pose.success && !dual_c1_tmpl_pts3d_.empty() &&
        !left_c1_gray.empty() && !right_c1_gray.empty()) {
        auto t_c1_ext = std::chrono::steady_clock::now();
        PipelineResult r_c1 = binary_extractor_->extract(
            left_c1_gray, right_c1_gray, left_c1_color, right_c1_color);
        extract_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_c1_ext).count();
        int n_c1 = static_cast<int>(r_c1.pts_left_match.size());

        if (n_c1 < 4) {
            routeStep("Tier1b(BCJ) SKIP[c1 corners=" + std::to_string(n_c1) + " <4]");
        } else {
            // 观测集: c0 角点 + c1 角点 (左图全图坐标), 记录类内索引供右图取点
            const int n_c1_right = static_cast<int>(r_c1.pts_right_good.size());
            std::vector<cv::Point2f> obs;
            std::vector<char> obs_is_c1;
            std::vector<int> obs_idx;    // 类内索引 (c0: result_bc 序; c1: r_c1 序)
            obs.reserve(static_cast<size_t>(n_bc) + n_c1);
            obs_is_c1.reserve(static_cast<size_t>(n_bc) + n_c1);
            obs_idx.reserve(static_cast<size_t>(n_bc) + n_c1);
            for (int i = 0; i < n_bc; ++i) {
                obs.emplace_back(result_bc.pts_left_match[i].x + static_cast<float>(left_off.x),
                                 result_bc.pts_left_match[i].y + static_cast<float>(left_off.y));
                obs_is_c1.push_back(0);
                obs_idx.push_back(i);
            }
            for (int i = 0; i < n_c1; ++i) {
                obs.emplace_back(r_c1.pts_left_match[i].x + static_cast<float>(left_sec.x),
                                 r_c1.pts_left_match[i].y + static_cast<float>(left_sec.y));
                obs_is_c1.push_back(1);
                obs_idx.push_back(i);
            }

            // 预测集: P0 投影全部标注板系 3D 点 (c0 段在前, c1 段在后)
            std::vector<cv::Point2f> pred;
            std::vector<Eigen::Vector3d> all_3d;
            pred.reserve(dual_bc_tmpl_pts3d_.size() + dual_c1_tmpl_pts3d_.size());
            auto projOf = [&](const Eigen::Vector3d& P) {
                Eigen::Vector3d Pc = pose.R * P + pose.t;
                Eigen::Vector2d uv = projectToImage(Pc, camera_.K);
                return cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
            };
            for (const auto& P : dual_bc_tmpl_pts3d_) { all_3d.push_back(P); pred.push_back(projOf(P)); }
            for (const auto& P : dual_c1_tmpl_pts3d_) { all_3d.push_back(P); pred.push_back(projOf(P)); }

            // 贪心 NN 匹配: 距离升序接受, 双向唯一
            struct Cand { double d; int obs_i; int tmpl_j; };
            std::vector<Cand> cands;
            const double gate = config_.dual_roi_joint_match_gate_px;
            for (int i = 0; i < static_cast<int>(obs.size()); ++i) {
                for (int j = 0; j < static_cast<int>(pred.size()); ++j) {
                    double dx = obs[i].x - pred[j].x;
                    double dy = obs[i].y - pred[j].y;
                    double d = std::sqrt(dx * dx + dy * dy);
                    if (d <= gate) cands.push_back({d, i, j});
                }
            }
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b) { return a.d < b.d; });
            std::vector<char> obs_used(obs.size(), 0), tmpl_used(pred.size(), 0);
            std::vector<cv::Point2f> j2d_c0l, j2d_c0r, j2d_c1l, j2d_c1r;
            std::vector<int> jidx_c0, jidx_c1;
            for (const auto& c : cands) {
                if (obs_used[c.obs_i] || tmpl_used[c.tmpl_j]) continue;
                const int oi = obs_idx[c.obs_i];
                cv::Point2f pr;
                bool has_right = true;
                if (obs_is_c1[c.obs_i]) {
                    if (oi < n_c1_right) {
                        pr = r_c1.pts_right_good[oi];
                        pr.x += static_cast<float>(right_sec.x);
                        pr.y += static_cast<float>(right_sec.y);
                    } else has_right = false;
                } else {
                    if (oi < n_bc_right) {
                        pr = result_bc.pts_right_good[oi];
                        pr.x += static_cast<float>(right_off.x);
                        pr.y += static_cast<float>(right_off.y);
                    } else has_right = false;
                }
                if (!has_right) continue;   // 右点缺失: 不占用观测/模板, 让位后续候选
                obs_used[c.obs_i] = 1;
                tmpl_used[c.tmpl_j] = 1;
                if (obs_is_c1[c.obs_i]) {
                    j2d_c1l.push_back(obs[c.obs_i]); j2d_c1r.push_back(pr);
                    jidx_c1.push_back(c.tmpl_j);
                } else {
                    j2d_c0l.push_back(obs[c.obs_i]); j2d_c0r.push_back(pr);
                    jidx_c0.push_back(c.tmpl_j);
                }
            }

            const int n_match = static_cast<int>(j2d_c0l.size() + j2d_c1l.size());
            if (n_match < 4 || n_match <= total_use) {
                routeStep("Tier1b(BCJ) SKIP[match=" + std::to_string(n_match)
                          + " pts, no gain]");
            } else {
                // 联合数组: c0 对在前, c1 对在后
                std::vector<cv::Point2f> jl = j2d_c0l, jr = j2d_c0r;
                jl.insert(jl.end(), j2d_c1l.begin(), j2d_c1l.end());
                jr.insert(jr.end(), j2d_c1r.begin(), j2d_c1r.end());
                std::vector<int> jtidx = jidx_c0;
                jtidx.insert(jtidx.end(), jidx_c1.begin(), jidx_c1.end());
                std::vector<Eigen::Vector3d> j3d;
                j3d.reserve(jtidx.size());
                for (int j : jtidx) j3d.push_back(all_3d[j]);

                // GPnP 输入载体 (左右 2D + 1:1 matches)
                PipelineResult jr_res;
                jr_res.pts_left_match = jl;
                jr_res.pts_left_good  = jl;
                jr_res.pts_left_used  = jl;
                jr_res.pts_right_good = jr;
                jr_res.pts_right_used = jr;
                jr_res.good_matches.resize(n_match);
                jr_res.idx_from_filtered.resize(n_match);
                jr_res.valid_mask.resize(n_match, true);
                const int n0c = static_cast<int>(dual_bc_tmpl_corners_.size());
                jr_res.pts_template_match.resize(n_match);
                for (int i = 0; i < n_match; ++i) {
                    jr_res.good_matches[i] = cv::DMatch(i, i, 0.0f);
                    jr_res.idx_from_filtered[i] = i;
                    jr_res.pts_template_match[i] = (jtidx[i] < n0c)
                        ? dual_bc_tmpl_corners_[jtidx[i]]
                        : dual_c1_tmpl_corners_[jtidx[i] - n0c];
                }
                jr_res.left_color  = left_color_orig;
                jr_res.right_color = right_color_orig;

                // 解算链: GPnP(P0 warm) → 失败回退 InitialPnP + GPnP (Tier1a 同链)
                auto t_j = std::chrono::steady_clock::now();
                double joint_gpnp_t = 0.0;
                PoseEstimate pj = gpnp_solver_.solve(jr_res, j3d,
                                                     &pose.R, &pose.t, joint_gpnp_t);
                bool joint_ok = pj.success;
                if (!joint_ok) {
                    MatchResult jm;
                    jm.good_matches       = jr_res.good_matches;
                    jm.pts_left_match     = jl;
                    jm.pts_template_match = jr_res.pts_template_match;
                    PoseEstimate ip = initial_pnp_.solve(jm, j3d, camera_.K);
                    if (ip.success) {
                        pj = gpnp_solver_.solve(jr_res, j3d, &ip.R, &ip.t, joint_gpnp_t);
                        if (!pj.success) pj = ip;
                    } else {
                        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
                        Eigen::Vector3d t_id(0, 0, 5000);
                        pj = gpnp_solver_.solve(jr_res, j3d, &R_id, &t_id, joint_gpnp_t);
                    }
                    joint_ok = pj.success;
                }
                // 运动门控 (与 Tier1a 同语义; 链内 InitialPnP 已是无 seed 冷解,
                // 拒绝直接判败, 不再二次冷重解)
                if (joint_ok && use_seed &&
                    !motionGatePass(pj, seed, nullptr, false, "DualRoi-BCJ", true)) {
                    joint_ok = false;
                    if (verbose_console_)
                        std::cout << "  [DualRoi] Tier1b joint pose rejected by motion gate"
                                  << "\n";
                }
                pnp_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_j).count();

                if (joint_ok) {
                    pose = pj;
                    joint_used = true;
                    result.pts_left_match       = std::move(jl);
                    result.pts_left_good        = result.pts_left_match;
                    result.pts_left_used        = result.pts_left_match;
                    result.pts_right_good       = std::move(jr);
                    result.pts_right_used       = result.pts_right_good;
                    result.pts_template_match   = jr_res.pts_template_match;
                    result.good_matches         = std::move(jr_res.good_matches);
                    result.idx_from_filtered    = std::move(jr_res.idx_from_filtered);
                    result.valid_mask           = std::move(jr_res.valid_mask);
                    result.n_kp_left            = n_match;
                    merged_kp_left.clear();
                    merged_kp_left.reserve(n_match);
                    for (const auto& p : result.pts_left_match)
                        merged_kp_left.emplace_back(p, 1.0f);
                    result.kp_left = merged_kp_left;
                    merged_pts3d   = std::move(j3d);
                    total_use      = n_match;
                    bc_total       = static_cast<int>(j2d_c0l.size());
                    pair_tmpl_idx  = std::move(jtidx);
                    // 重填视差 (口径同主链: dx=left.x-right.x, disparity=-dx)
                    result.disparity.resize(n_match);
                    result.dx_filtered.resize(n_match);
                    for (int i = 0; i < n_match; ++i) {
                        double d = static_cast<double>(result.pts_left_match[i].x -
                                                       result.pts_right_good[i].x);
                        result.dx_filtered[i] = d;
                        result.disparity[i] = -d;
                    }
                    routeStep("Tier1b(BCJ c0=" + std::to_string(bc_total)
                              + "+c1=" + std::to_string(static_cast<int>(j2d_c1l.size()))
                              + ") OK");
                } else {
                    routeStep("Tier1b(BCJ c0=" + std::to_string(static_cast<int>(j2d_c0l.size()))
                              + "+c1=" + std::to_string(static_cast<int>(j2d_c1l.size()))
                              + ") FAIL[pnp]");
                }
            }
        }
    } else if (pose.success && dual_c1_tmpl_pts3d_.empty()) {
        routeStep("Tier1b(BCJ) SKIP[no c1 anno]");
    }

    } // !primary_spans_frame（短路时 pose 保持失败, 9c 直接成为唯一路径）

    // 9c. 退化: Tier1 BC(class0) 失败 → 在 secondary ROI (class1) 上跑 BC→TT 链
    //     (class1 3D 尺寸, State 5 同机制; 冷启动, 不依赖板系位姿缓存)
    bool c1_fallback_used = false;
    std::string c1_strategy;
    if (!kDualRoiTier1Only && !pose.success && config_.dual_roi_class1_fallback) {
        if (verbose_console_)
            std::cout << "  [DualRoi] Tier1 BC solve failed, trying class1 chain (BC→TT)"
                      << "\n";
        PipelineResult c1_result;
        std::vector<Eigen::Vector3d> c1_pts3d;
        std::string c1_legs;
        auto [ok, c1_pose] = runDualRoiClass1Chain(
            left_c1_gray, right_c1_gray, left_c1_color, right_c1_color,
            left_sec, right_sec, is_first, c1_result, c1_pts3d, c1_strategy,
            extract_ms, pnp_ms, &c1_legs);
        if (ok) {
            pose = c1_pose;
            c1_result.left_color = left_color_orig;
            c1_result.right_color = right_color_orig;
            result = std::move(c1_result);
            merged_pts3d = std::move(c1_pts3d);
            total_use = static_cast<int>(result.pts_left_match.size());
            bc_total = 0;   // 可视化分色: 链胜出点全部来自 class1
            c1_fallback_used = true;
            routeStep("Tier3(class1 " + c1_legs + ") OK");
        } else {
            routeStep("Tier3(class1) FAIL[" + (c1_legs.empty() ? "no leg" : c1_legs) + "]");
        }
    } else if (!pose.success && !kDualRoiTier1Only) {
        routeStep("Tier3 SKIP[class1_fallback disabled]");
    } else if (!pose.success) {
        routeStep("Tier3 SKIP[ablation]");
    }

    if (pose.success) {
        finalizePose(result, pose);
    }
    // Tier1(BC class0) 现为主级: Tier1b 联合解算胜出为 "DualRoi_BCJ", 纯 c0 为
    // "DualRoi_BC"; class1 链胜出保留 C1 命名
    result.strategy_name = c1_fallback_used ? c1_strategy
                         : (joint_used ? "DualRoi_BCJ" : "DualRoi_BC");

    // BC 二值化过程面板: 不依赖位姿成功, 失败帧也输出 (排查主证据)
    if (visualize && visualize_detailed_ && !output_dir_.empty())
        saveBcProcessPanel();

    // ---- Visualization (dual-ROI) ----
    if (visualize && pose.success) {
        std::string prefix = "_f" + std::to_string(current_frame_);

        // Normal 模式: 仅三维坐标轴叠加图
        if (!visualize_detailed_) {
            cv::Mat vis = result.left_color.clone();
            for (const auto& pt : result.pts_left_match)
                cv::drawMarker(vis, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 1, 1);
            if (result.gpnp_success) {
                cv::Mat K_cv = (cv::Mat_<double>(3,3) <<
                    camera_.K(0,0), camera_.K(0,1), camera_.K(0,2),
                    camera_.K(1,0), camera_.K(1,1), camera_.K(1,2),
                    camera_.K(2,0), camera_.K(2,1), camera_.K(2,2));
                cv::Mat rvec, R_cv = (cv::Mat_<double>(3,3) <<
                    result.R(0,0), result.R(0,1), result.R(0,2),
                    result.R(1,0), result.R(1,1), result.R(1,2),
                    result.R(2,0), result.R(2,1), result.R(2,2));
                cv::Rodrigues(R_cv, rvec);
                cv::Mat tvec = (cv::Mat_<double>(3,1) << result.t(0), result.t(1), result.t(2));
                std::vector<cv::Point3d> axis = {{0,0,0},{100,0,0},{0,100,0},{0,0,100}};
                std::vector<cv::Point2d> img_pts;
                cv::projectPoints(axis, rvec, tvec, K_cv, cv::Mat(), img_pts);
                if (img_pts.size() == 4) {
                    cv::line(vis, img_pts[0], img_pts[1], cv::Scalar(0,0,255), 3);
                    cv::line(vis, img_pts[0], img_pts[2], cv::Scalar(0,255,0), 3);
                    cv::line(vis, img_pts[0], img_pts[3], cv::Scalar(255,0,0), 3);
                }
            }
            utils::AsyncImageSaver::write(output_dir_ + "/stereo_dualroi" + prefix + ".png", vis);
        } else {

        const cv::Scalar BC_COLOR(0, 0, 255);    // red: BinaryCorner corners (class 0 edges)
        const cv::Scalar AK_COLOR(0, 255, 0);    // green: class1 链点 (Tier3)
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
            cv::putText(p0, "class1 (BC)",
                cv::Point(left_sec.x + 4, left_sec.y + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_overview" + prefix + ".png", p0);
        }

        // --- Panel 1: Class 0 ROI zoomed — BC corners (red) + class1 链点 (green) ---
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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_corners" + prefix + ".png", p1);
        }

        // --- Panel 2: 3D axes on original image (origin = 目标中心, 3D 点已居中) ---
        {
            cv::Mat p2 = left_color_orig.clone();
            double axis_len = 100.0;
            Eigen::Vector3d o  = pose.R * Eigen::Vector3d(0, 0, 0) + pose.t;
            Eigen::Vector3d ax = pose.R * Eigen::Vector3d(axis_len, 0, 0) + pose.t;
            Eigen::Vector3d ay = pose.R * Eigen::Vector3d(0, axis_len, 0) + pose.t;
            Eigen::Vector3d az = pose.R * Eigen::Vector3d(0, 0, axis_len) + pose.t;
            cv::line(p2, projPoint(o), projPoint(ax), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::line(p2, projPoint(o), projPoint(ay), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(p2, projPoint(o), projPoint(az), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_axes" + prefix + ".png", p2);
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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_reproj" + prefix + ".png", p3);
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
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_right" + prefix + ".png", p4);
        }

        // --- Panel 5: Image ↔ Template correspondence (side-by-side) ---
        // class1 链胜出时跳过: 模板侧对应关系为板模板 BC 角点, 与链点 (BC/TT class1) 不匹配
        if (!c1_fallback_used) {
            // Left: class-0-ROI image
            cv::Mat p5_left = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();

            // Right: board template (grayscale → BGR, resize to match left height)
            const cv::Mat& tmpl_gray = template_.gray_image;
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

            // 模板侧点: pair_tmpl_idx[i] < n0c → class0 标注角点(红),
            // 否则 → class1 标注角点(绿, 索引偏移 n0c)
            const int n0c = static_cast<int>(dual_bc_tmpl_corners_.size());
            const int n1c = static_cast<int>(dual_c1_tmpl_corners_.size());
            for (int i = 0; i < total_use &&
                           i < static_cast<int>(pair_tmpl_idx.size()); ++i) {
                // Image point (left side, class-0-ROI-local)
                cv::Point pt_img(static_cast<int>(result.pts_left_match[i].x - lx),
                                 static_cast<int>(result.pts_left_match[i].y - ly));

                const int j = pair_tmpl_idx[i];
                cv::Point2f tmpl_pt;
                const cv::Scalar* color;
                if (j < n0c) {
                    tmpl_pt = dual_bc_tmpl_corners_[j];
                    color = &BC_COLOR;
                } else if (j - n0c < n1c) {
                    tmpl_pt = dual_c1_tmpl_corners_[j - n0c];
                    color = &AK_COLOR;
                } else {
                    continue;
                }
                tmpl_pt *= scale_tmpl;
                cv::Point pt_tmpl(static_cast<int>(tmpl_pt.x) + offset_x,
                                  static_cast<int>(tmpl_pt.y));

                cv::circle(p5, pt_img,  2, *color, -1);
                cv::circle(p5, pt_tmpl, 2, *color, -1);
                cv::line(p5, pt_img, pt_tmpl, *color, 1, cv::LINE_AA);
            }
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_correspondence" + prefix + ".png", p5);
        }

        if (verbose_console_)
            std::cout << "  [DualRoi] Visualized: " << bc_total << " BC + "
                      << ak_count << " C1 pts" << "\n";
        } // end else (full debug viz)
    }

    result.n_matched = total_use;
    result.n_projected = total_use;
    result.extract_ms = extract_ms;
    result.pnp_ms = pnp_ms;
    addLogEntry(result, is_first, c1_fallback_used);

    result.timing["dual_roi"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_dual_start).count();

    routeStep(pose.success ? "FRAME-OK" : "FRAME-FAIL");
    if (verbose_console_)
        std::cout << "[DualRoi] Route: " << route << "\n";

    if (verbose_console_)
        std::cout << "[DualRoi] Frame done: n_pts=" << total_use
                  << "  GPNP=" << (pose.success ? "OK" : "FAIL")
                  << (c1_fallback_used ? "  [class1-chain:" + c1_strategy + "]" : "")
                  << "  time=" << result.total_time_ms() << "ms"
                  << "\n";

    return result;
}

// ============================================================================
// Dual-ROI Tier 3: class1 退化链 —— secondary ROI 上 BC → TT
// (State 5 同款 class1 3D 尺寸机制; 参考主退化链: 提取失败或 PnP 失败均降级)
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runDualRoiClass1Chain(
        const cv::Mat& left_gray, const cv::Mat& right_gray,
        const cv::Mat& left_color, const cv::Mat& right_color,
        const RoiRect& left_sec, const RoiRect& right_sec,
        bool is_first,
        PipelineResult& out_result,
        std::vector<Eigen::Vector3d>& out_pts3d,
        std::string& out_strategy,
        double& extract_ms_acc, double& pnp_ms_acc,
        std::string* out_route_detail) {
    out_strategy.clear();
    out_pts3d.clear();
    if (out_route_detail) out_route_detail->clear();
    if (left_gray.empty() || right_gray.empty()) return {false, PoseEstimate{}};

    // 链内腿级明细: 失败时累计 "BC:原因; TT:原因", 成功时记录胜出腿 ("BC"/"TT")
    std::string fail_detail;
    auto legFail = [&](const char* leg, const char* reason) {
        if (!fail_detail.empty()) fail_detail += "; ";
        fail_detail += std::string(leg) + ":" + reason;
        if (out_route_detail) *out_route_detail = fail_detail;
    };

    // BC/TT 切到 class1 3D 尺寸; RAII 恢复, 避免 Dual-ROI 早退路径残留提取器状态
    struct Class1ScaleGuard {
        BinaryCornerExtractor* bc;
        TinyTargetExtractor* tt;
        ~Class1ScaleGuard() { bc->setUseClass1(false); tt->setUseClass1(false); }
    } guard{binary_extractor_.get(), tiny_extractor_.get()};
    binary_extractor_->setUseClass1(true);
    tiny_extractor_->setUseClass1(true);

    FeatureExtractor* chain[] = {binary_extractor_.get(), tiny_extractor_.get()};
    for (auto* ext : chain) {
        const bool is_bc = (ext->strategyType() == StrategyType::BinaryCorner);
        if (verbose_console_)
            std::cout << "[DualRoi] class1 chain: " << ext->name()
                      << " on secondary ROI" << "\n";

        PipelineResult r;
        auto t_ext = std::chrono::steady_clock::now();
        const bool extract_ok = runExtraction(
            *ext, left_gray, right_gray, left_color, right_color,
            cv::Point2d(static_cast<double>(left_sec.x), static_cast<double>(left_sec.y)),
            cv::Point2d(static_cast<double>(right_sec.x), static_cast<double>(right_sec.y)),
            left_color, right_color, r);
        extract_ms_acc += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_ext).count();
        if (!extract_ok || !(r.success && r.n_kp_left >= 3)) {
            if (verbose_console_)
                std::cout << "  [DualRoi] class1 " << ext->name()
                          << " extraction failed" << "\n";
            legFail(is_bc ? "BC" : "TT", "extract");
            continue;
        }

        const auto& pts3d = ext->templateData().pts_3d;
        auto t_c1_pnp = std::chrono::steady_clock::now();
        auto [ok, c1_pose] = is_bc ? solveBcPnpChain(r, pts3d)
                                   : runTinyTargetPnP(r, pts3d);
        pnp_ms_acc += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_c1_pnp).count();
        if (!ok) {
            if (verbose_console_)
                std::cout << "  [DualRoi] class1 " << ext->name()
                          << " PnP failed" << "\n";
            legFail(is_bc ? "BC" : "TT", "pnp");
            continue;
        }

        r.is_first_frame = is_first;
        r.is_class1 = true;
        out_result = std::move(r);
        out_pts3d = pts3d;   // 与 out_result.pts_left_match 前缀 1:1 (BC/TT 规范序)
        out_strategy = is_bc ? "DualRoi_C1BC" : "DualRoi_C1TT";
        if (out_route_detail) *out_route_detail = is_bc ? "BC" : "TT";
        return {ok, c1_pose};
    }
    return {false, PoseEstimate{}};
}

// ============================================================================
// Dual-ROI Mono —— class 0 (BC) + class 1 (AK) → MonoPnPSolver
// ============================================================================


// ============================================================================
// 单目模式 —— 仅左图特征提取 + PnP 解算
// ============================================================================


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
// 单目降级处理 —— 双目模式左右仅一侧检测到目标时调用
// ============================================================================

PipelineResult StereoTracker::processMono(const cv::Mat& img,
                                                 bool visualize,
                                                 const RoiGroup* roi_group) {
    PipelineResult result;
    result.success = false;

    if (img.empty()) {
        std::cerr << "[StereoMono] empty image" << "\n";
        return result;
    }

    auto [color, gray] = loadImage(img, visualize);
    if (gray.empty()) {
        std::cerr << "[StereoMono] failed to load image" << "\n";
        return result;
    }
    result.left_color = color;

    // ROI 校验：无 ROI → 帧失败（全图回退随 AKAZE 链一并移除）
    const RoiRect* roi_ptr = roi_group ? &roi_group->primary : nullptr;
    RoiRect roi = validateRoi(roi_ptr, img.size(), "mono_fallback");
    if (!roi.valid()) {
        if (verbose_console_)
            std::cout << "[StereoMono] no valid ROI, frame fails (Skip_NoRoi)" << "\n";
        result.strategy_name = "Skip_NoRoi";
        addLogEntry(result, state_.frame_count == 0, false);
        return result;
    }

    int roi_area = roi.width * roi.height;
    if (verbose_console_)
        std::cout << "[StereoMono] ROI area=" << roi_area
                  << " (" << roi.width << "x" << roi.height << ")"
                  << "\n";

    cv::Mat gray_roi  = gray( cv::Rect(roi.x, roi.y, roi.width, roi.height));
    cv::Mat color_roi = color.empty() ? color : color(cv::Rect(roi.x, roi.y, roi.width, roi.height));
    cv::Point2d offset(roi.x, roi.y);

    // class1 标志 + 策略链
    bool use_c1 = (roi_group != nullptr && roi_group->is_class1);
    {
        binary_extractor_->setUseClass1(use_c1);
        tiny_extractor_->setUseClass1(use_c1);
    }
    configureStrategyChain(roi_area, use_c1);

    // 时序 seed（机制A: 位姿链）—— 上帧成功位姿作为本帧 PnP 初值候选
    PoseSeed seed;
    const bool use_seed = seedActive();
    if (use_seed) { seed.R = state_.R_prev; seed.t = state_.t_prev; }

    bool is_first = (state_.frame_count == 0);

    // 提取成功且单目 PnP 成功才算胜出，PnP 失败同样触发退化
    std::vector<FeatureExtractor*> chain;
    chain.push_back(extractor_);
    for (auto* fb : fallback_extractors_)
        chain.push_back(fb);

    FeatureExtractor* winning_ext = nullptr;
    PoseEstimate pose;
    double extract_ms = 0.0, pnp_ms = 0.0;   // 阶段耗时: 特征提取 / 位姿解算 (跨退化链累加)

    for (auto* ext : chain) {
        if (!ext) continue;

        if (verbose_console_)
            std::cout << "[StereoMono] Trying extractor: " << ext->name() << "\n";

        auto t_extract = std::chrono::steady_clock::now();
        PipelineResult local = ext->extractMono(gray_roi, color_roi);
        extract_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_extract).count();

        if (!local.pts_left_match.empty()) {
            for (auto& p : local.pts_left_match) {
                p.x += static_cast<float>(offset.x);
                p.y += static_cast<float>(offset.y);
            }
        }
        if (!local.pts_left_good.empty()) {
            for (auto& p : local.pts_left_good) {
                p.x += static_cast<float>(offset.x);
                p.y += static_cast<float>(offset.y);
            }
        }
        for (auto& kp : local.kp_left) {
            kp.pt.x += static_cast<float>(offset.x);
            kp.pt.y += static_cast<float>(offset.y);
        }

        if (!(local.success && local.n_kp_left >= 3)) {
            if (verbose_console_)
                std::cout << "[StereoMono] Extractor " << ext->name() << " failed, degrading..." << "\n";
            continue;
        }

        // 单目 PnP（循环内：提取成功但解算失败继续退化链）。
        // good_matches[i] ↔ pts_left_match[i] 平行序（AKAZE/TT 等长平行；BC 近似
        // 命中时角点数可多于模板角点，尾部无匹配角点自然被跳过）
        const auto& pnp_pts_3d = ext->templateData().pts_3d;
        std::vector<cv::Point2f> pnp_2d;
        std::vector<Eigen::Vector3d> pnp_3d;
        pnp_2d.reserve(std::min(local.good_matches.size(), local.pts_left_match.size()));
        for (size_t i = 0; i < local.good_matches.size() &&
                           i < local.pts_left_match.size(); ++i) {
            int idx = local.good_matches[i].trainIdx;
            if (idx >= 0 && idx < static_cast<int>(pnp_pts_3d.size())) {
                pnp_2d.push_back(local.pts_left_match[i]);
                pnp_3d.push_back(pnp_pts_3d[idx]);
            }
        }
        // 单目 PnP: 有 seed 走位姿链候选 + 运动门控；门控拒绝 → 冷启动重解一次
        PoseEstimate p;
        GateStatus gate_st = GateStatus::NotApplicable;
        auto t_pnp = std::chrono::steady_clock::now();
        if (use_seed) {
            p = mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K, &seed,
                                config_.temporal.tie_epsilon_px);
            if (p.success) {
                if (motionGatePass(p, seed, ext, false)) {
                    gate_st = GateStatus::Pass;
                } else {
                    PoseEstimate pc = mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K, nullptr, 0.0);
                    if (pc.success) {
                        // 冷重解 = temporal-OFF 的可信路径，直接采纳（不再被陈旧 seed 锚定的门控二次审查）
                        p = pc;
                        gate_st = GateStatus::Recovered;
                        if (verbose_console_)
                            std::cout << "[Gate] warm pose rejected, cold re-solve accepted"
                                      << "\n";
                    } else {
                        p = PoseEstimate{};
                        gate_st = GateStatus::Rejected;
                        if (verbose_console_)
                            std::cout << "[Gate] pose rejected by motion gate ("
                                      << ext->name() << "), degrading..." << "\n";
                    }
                }
            }
        } else {
            p = mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K);
        }
        pnp_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pnp).count();

        result = std::move(local);
        result.left_color = color;
        result.left_roi_offset_x = static_cast<int>(offset.x);
        result.left_roi_offset_y = static_cast<int>(offset.y);
        result.warm_start_used = use_seed;
        result.gate_status = gate_st;
        fillTemporalMeta(result);
        winning_ext = ext;

        if (p.success) {
            pose = p;
            if (verbose_console_)
                std::cout << "[StereoMono] Extractor " << ext->name()
                          << " succeeded, n_kp=" << result.n_kp_left << "\n";
            break;
        }

        if (verbose_console_)
            std::cout << "[StereoMono] Extractor " << ext->name()
                      << " PnP failed, degrading..." << "\n";
    }

    if (!winning_ext) {
        std::cerr << "[StereoMono] All extractors failed" << "\n";
        result.is_class1 = use_c1;
        result.extract_ms = extract_ms;
        result.pnp_ms = pnp_ms;
        addLogEntry(result, is_first, true);
        return result;
    }

    finalizePose(result, pose);

    result.strategy_name = winning_ext->name();
    result.success = pose.success;
    result.is_class1 = use_c1;
    result.extract_ms = extract_ms;
    result.pnp_ms = pnp_ms;
    updateStickinessFromWinner(winning_ext->name());
    fillTemporalMeta(result);   // 胜者反馈后重取，switch_reason 反映本帧 realign
    addLogEntry(result, is_first, winning_ext != chain.front());

    // ---- Visualization (仅三维轴叠加图，Normal/Debug 一致) ----
    if (visualize && !output_dir_.empty()) {
        cv::Mat vis = color.clone();
        for (const auto& pt : result.pts_left_match)
            cv::drawMarker(vis, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 8, 2);
        if (result.success) {
            cv::Mat K_cv = (cv::Mat_<double>(3,3) <<
                camera_.K(0,0), camera_.K(0,1), camera_.K(0,2),
                camera_.K(1,0), camera_.K(1,1), camera_.K(1,2),
                camera_.K(2,0), camera_.K(2,1), camera_.K(2,2));
            cv::Mat rvec, R_cv = (cv::Mat_<double>(3,3) <<
                result.R(0,0), result.R(0,1), result.R(0,2),
                result.R(1,0), result.R(1,1), result.R(1,2),
                result.R(2,0), result.R(2,1), result.R(2,2));
            cv::Rodrigues(R_cv, rvec);
            cv::Mat tvec = (cv::Mat_<double>(3,1) << result.t(0), result.t(1), result.t(2));
            std::vector<cv::Point3d> axis = {{0,0,0},{100,0,0},{0,100,0},{0,0,100}};
            std::vector<cv::Point2d> img_pts;
            cv::projectPoints(axis, rvec, tvec, K_cv, cv::Mat(), img_pts);
            if (img_pts.size() == 4) {
                cv::line(vis, img_pts[0], img_pts[1], cv::Scalar(0,0,255), 3);
                cv::line(vis, img_pts[0], img_pts[2], cv::Scalar(0,255,0), 3);
                cv::line(vis, img_pts[0], img_pts[3], cv::Scalar(255,0,0), 3);
            }
        }
        std::string mono_path = output_dir_ + "/mono_f" + std::to_string(current_frame_) + ".png";
        utils::AsyncImageSaver::write(mono_path, vis);
        if (verbose_console_)
            std::cout << "[StereoMono] Visualization saved: " << mono_path << "\n";
    }

    return result;
}

// ============================================================================
// 图像加载 —— 分离彩色图和灰度图
// ============================================================================



} // namespace gpnp
