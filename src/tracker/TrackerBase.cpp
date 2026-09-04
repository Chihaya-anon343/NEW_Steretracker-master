/**
 * @file TrackerBase.cpp
 * @brief MonoTracker / StereoTracker 共享基础设施。
 */

#include "tracker/TrackerBase.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>

namespace gpnp {

// ============================================================
// initExtractors
// ============================================================

void TrackerBase::initExtractors(const std::string& template_path,
                                  const BinaryCornerExtractor::Config& binary_cfg,
                                  const std::string& binary_template_dir,
                                  const TinyTargetExtractor::Config& tiny_cfg,
                                  const std::string& tiny_template_dir) {
    akaze_extractor_ = std::make_unique<AkazeGpnpExtractor>(config_.scale, config_.lk_params);
    akaze_extractor_->initCamera(camera_);
    akaze_extractor_->setTemplateData(template_path,
                                       config_.template_real_width_mm,
                                       config_.template_real_height_mm);
    template_ = akaze_extractor_->templateData();

    binary_extractor_ = std::make_unique<BinaryCornerExtractor>(binary_cfg, binary_template_dir);
    tiny_extractor_ = std::make_unique<TinyTargetExtractor>(tiny_cfg, tiny_template_dir);

    akaze_min_area_ = config_.akaze_min_area;
    tiny_max_area_  = config_.tiny_max_area;
    akaze_min_area_class1_ = config_.akaze_min_area_class1;
    tiny_max_area_class1_  = config_.tiny_max_area_class1;
    binary_roi_pad_ = binary_cfg.roi_pad_pixels;
    tiny_roi_pad_   = tiny_cfg.roi_pad_pixels;

    extractor_ = akaze_extractor_.get();
    fallback_extractors_.push_back(binary_extractor_.get());
    fallback_extractors_.push_back(tiny_extractor_.get());

    if (verbose_console_)
        std::cout << "[TrackerBase] Pre-initialized 3 extractors: AkazeGpnp, BinaryCorner, TinyTarget"
                  << std::endl;
}

// ============================================================
// configureStrategyChain — 面积提名 + 粘滞状态机 (temporal)
// ============================================================

void TrackerBase::configureStrategyChain(int roi_area, bool is_class1) {
    const TemporalConfig& tc = config_.temporal;

    // 缓存年龄：每处理帧自增（成功帧由 finalizePose 清零）
    cache_age_ = state_.has_cache ? cache_age_ + 1 : 0;

    // class1 regime 翻转 → 阈值集合切换，粘滞复位
    if (tc.enabled && has_prev_c1_ && prev_use_c1_ != is_class1) {
        resetStickiness();
        if (verbose_console_)
            std::cout << "[Temporal] class1 regime flipped, stickiness reset" << std::endl;
    }
    prev_use_c1_ = is_class1;
    has_prev_c1_ = true;

    // 面积 EMA：仅用于提名/迟滞（抗 YOLO bbox 抖动）；越级硬切用 raw 面积不吃平滑延迟
    if (!has_ema_ || tc.area_ema_alpha <= 0.0) {
        ema_area_ = roi_area;
        has_ema_ = true;
    } else {
        ema_area_ = tc.area_ema_alpha * roi_area + (1.0 - tc.area_ema_alpha) * ema_area_;
    }

    nominated_band_ = nominateBand(roi_area, is_class1);
    int band_used = nominated_band_;
    if (tc.enabled)
        band_used = resolveLockedStrategy(nominated_band_, is_class1);
    sticky_hold_ = tc.enabled && (band_used != nominated_band_);

    applyBandChain(band_used, is_class1);

    // 粘滞压低档位时补齐向上的回退链：提名高档若失效仍可向上退化，避免 TT 单点终结
    if (tc.enabled && band_used < nominated_band_) {
        auto pushIfAbsent = [&](FeatureExtractor* e) {
            if (extractor_ == e) return;
            for (auto* x : fallback_extractors_) if (x == e) return;
            fallback_extractors_.push_back(e);
        };
        if (band_used <= 1 && nominated_band_ >= 2)
            pushIfAbsent(binary_extractor_.get());
        if (nominated_band_ == 3)
            pushIfAbsent(akaze_extractor_.get());
    }

    if (verbose_console_ && tc.enabled)
        std::cout << "[Temporal] locked=" << strategyNameFromBand(band_used)
                  << " nominated=" << strategyNameFromBand(nominated_band_)
                  << " pending=" << pending_count_ << "/" << tc.hold_frames
                  << " ema_area=" << static_cast<int>(ema_area_)
                  << " raw_area=" << roi_area
                  << " reason=" << sticky_switch_reason_
                  << (sticky_hold_ ? " [HOLD]" : "") << std::endl;
}

void TrackerBase::effectiveThresholds(bool is_class1, int& akaze_t, int& tiny_t) const {
    akaze_t = akaze_min_area_;
    tiny_t  = tiny_max_area_;
    if (is_class1) {
        if (akaze_min_area_class1_ > 0) akaze_t = akaze_min_area_class1_;
        if (tiny_max_area_class1_ > 0)  tiny_t  = tiny_max_area_class1_;
    }
}

int TrackerBase::bandFromStrategyName(const std::string& name) {
    if (name == "TinyTarget")   return 1;
    if (name == "BinaryCorner") return 2;
    if (name == "AkazeGpnp")    return 3;
    return 0;   // DualRoi 等非三策略
}

const char* TrackerBase::strategyNameFromBand(int band) {
    switch (band) {
        case 3:  return "AkazeGpnp";
        case 2:  return "BinaryCorner";
        case 1:  return "TinyTarget";
        default: return "Unknown";
    }
}

int TrackerBase::nominateBand(int roi_area, bool is_class1) const {
    int akaze_t, tiny_t;
    effectiveThresholds(is_class1, akaze_t, tiny_t);
    if (roi_area >= akaze_t || roi_area == 0) return 3;
    if (roi_area > tiny_t) return 2;
    return 1;
}

void TrackerBase::applyBandChain(int band, bool is_class1) {
    int akaze_t, tiny_t;
    effectiveThresholds(is_class1, akaze_t, tiny_t);

    fallback_extractors_.clear();

    if (band == 3) {
        extractor_ = akaze_extractor_.get();
        fallback_extractors_.push_back(binary_extractor_.get());
        fallback_extractors_.push_back(tiny_extractor_.get());
        if (verbose_console_)
            std::cout << "[TrackerBase] Strategy chain: AkazeGpnp → BinaryCorner → TinyTarget"
                      << " (akaze_thresh=" << akaze_t
                      << " is_class1=" << is_class1 << ")" << std::endl;
    } else if (band == 2) {
        extractor_ = binary_extractor_.get();
        fallback_extractors_.push_back(tiny_extractor_.get());
        if (verbose_console_)
            std::cout << "[TrackerBase] Strategy chain: BinaryCorner → TinyTarget"
                      << " (tiny_thresh=" << tiny_t
                      << " is_class1=" << is_class1 << ")" << std::endl;
    } else {
        extractor_ = tiny_extractor_.get();
        if (verbose_console_)
            std::cout << "[TrackerBase] Strategy chain: TinyTarget only"
                      << " (is_class1=" << is_class1 << ")" << std::endl;
    }
}

int TrackerBase::resolveLockedStrategy(int band_raw, bool is_class1) {
    const TemporalConfig& tc = config_.temporal;

    if (!has_sticky_state_) {
        sticky_locked_ = band_raw;
        sticky_pending_ = 0;
        pending_count_ = 0;
        locked_fail_count_ = 0;
        has_sticky_state_ = true;
        sticky_switch_reason_ = "init";
        return sticky_locked_;
    }

    // 越级硬切：raw 面积跨越两档（如 TT↔AKAZE，快速接近/远离），立即切换
    if (std::abs(band_raw - sticky_locked_) >= 2) {
        sticky_locked_ = band_raw;
        sticky_pending_ = 0;
        pending_count_ = 0;
        locked_fail_count_ = 0;
        sticky_switch_reason_ = "hard_override";
        return sticky_locked_;
    }

    // 相邻档提名：用 EMA 面积重新分类（抗 bbox 抖动）
    int band_ema = nominateBand(static_cast<int>(ema_area_ + 0.5), is_class1);
    if (band_ema == sticky_locked_) {
        pending_count_ = 0;
        return sticky_locked_;
    }

    // 迟滞余量：必须决定性地进入另一档
    int akaze_t, tiny_t;
    effectiveThresholds(is_class1, akaze_t, tiny_t);
    bool margin_ok = false;
    if (band_ema > sticky_locked_) {
        double th = (band_ema == 3) ? akaze_t : tiny_t;
        margin_ok = ema_area_ > th * tc.hysteresis_up;
    } else {
        double th = (sticky_locked_ == 3) ? akaze_t : tiny_t;
        margin_ok = ema_area_ < th * tc.hysteresis_down;
    }

    if (band_ema == sticky_pending_) ++pending_count_;
    else { sticky_pending_ = band_ema; pending_count_ = 1; }

    if (margin_ok && pending_count_ >= tc.hold_frames) {
        sticky_locked_ = band_ema;
        pending_count_ = 0;
        sticky_switch_reason_ = "debounce";
    }
    return sticky_locked_;
}

void TrackerBase::resetStickiness() {
    has_sticky_state_ = false;
    sticky_locked_ = 0;
    sticky_pending_ = 0;
    pending_count_ = 0;
    locked_fail_count_ = 0;
    sticky_switch_reason_ = "reset";
}

void TrackerBase::updateStickinessFromWinner(const std::string& winner_name) {
    const TemporalConfig& tc = config_.temporal;
    if (!tc.enabled || !has_sticky_state_) return;
    int wb = bandFromStrategyName(winner_name);
    if (wb == 0) return;   // DualRoi 等非三策略胜者不参与

    if (wb == sticky_locked_) {
        locked_fail_count_ = 0;
        return;
    }
    if (wb == nominated_band_) {
        // 面积说该切、退化链实际切成了且成功 → 立即对齐（下帧不再浪费首提取尝试）
        sticky_locked_ = wb;
        pending_count_ = 0;
        locked_fail_count_ = 0;
        sticky_switch_reason_ = "realign";
        return;
    }
    if (++locked_fail_count_ >= tc.locked_fail_limit) {
        sticky_locked_ = wb;
        pending_count_ = 0;
        locked_fail_count_ = 0;
        sticky_switch_reason_ = "realign";
    }
}

bool TrackerBase::seedActive() const {
    const TemporalConfig& tc = config_.temporal;
    if (!tc.enabled || !state_.has_cache) return false;
    if (tc.max_cache_age_frames > 0 && cache_age_ > tc.max_cache_age_frames) return false;
    return true;
}

bool TrackerBase::motionGatePass(const PoseEstimate& pose, const PoseSeed& seed,
                                 const FeatureExtractor* ext, bool widened) const {
    const TemporalConfig& tc = config_.temporal;

    double margin = 1.0;
    if (widened) margin *= tc.switch_margin;
    if (ext) {
        int band = bandFromStrategyName(ext->name());
        if (band != 0 && band != sticky_locked_)   // 策略切换帧：放宽吸收策略间系统偏差
            margin *= tc.switch_margin;
    }

    if (tc.max_trans_ratio > 0.0) {
        double t_ref = seed.t.norm();
        double t_new = pose.t.norm();
        if (t_ref > 1e-9 && t_new > 1e-9) {
            double dt = (pose.t - seed.t).norm();
            double dt_limit = tc.max_trans_ratio * t_ref * margin;
            bool dt_ok = dt <= dt_limit;
            // 径向快变判据：|t| 比值在容限内视为合法快速接近/远离（与 Δt 判据并列，任一满足即过）
            bool scale_ok = false;
            double scale = (t_new >= t_ref) ? t_new / t_ref : t_ref / t_new;
            if (tc.max_scale_ratio > 1.0)
                scale_ok = scale <= tc.max_scale_ratio * margin;
            if (!dt_ok && !scale_ok) {
                if (verbose_console_)
                    std::cout << "[Gate] |Δt|=" << dt << "mm > " << dt_limit
                              << ", scale=" << scale << " > " << tc.max_scale_ratio * margin
                              << " (|t_seed|=" << t_ref << "mm, |t_new|=" << t_new
                              << "mm, margin=" << margin
                              << ", " << (ext ? ext->name() : "?") << ")" << std::endl;
                return false;
            }
        }
    }
    if (tc.max_rot_deg > 0.0) {
        double ang = Eigen::AngleAxisd(pose.R * seed.R.transpose()).angle();
        double limit = tc.max_rot_deg * CV_PI / 180.0 * margin;
        if (ang > limit) {
            if (verbose_console_)
                std::cout << "[Gate] Δθ=" << ang * 180.0 / CV_PI << "° > "
                          << tc.max_rot_deg * margin << "° (margin=" << margin
                          << ", " << (ext ? ext->name() : "?") << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void TrackerBase::fillTemporalMeta(PipelineResult& result) const {
    result.seed_age = cache_age_;
    result.sticky_hold = sticky_hold_;
    result.nominated_strategy = strategyNameFromBand(nominated_band_);
    result.switch_reason = sticky_switch_reason_;
}

// ============================================================
// validateRoi
// ============================================================

RoiRect TrackerBase::validateRoi(const RoiRect* roi, const cv::Size& img_size,
                                  const std::string& name) {
    if (roi == nullptr || !roi->valid()) return RoiRect{};
    int x = roi->x, y = roi->y, w = roi->width, h = roi->height;
    if (x < 0 || y < 0 || w <= 0 || h <= 0)
        throw std::invalid_argument(name + " invalid");
    if (x + w > img_size.width || y + h > img_size.height)
        throw std::invalid_argument(name + " out of bounds");
    return RoiRect{x, y, w, h};
}

// ============================================================
// loadImage
// ============================================================

std::pair<cv::Mat, cv::Mat> TrackerBase::loadImage(const cv::Mat& img) {
    cv::Mat gray, color;
    if (img.channels() == 3) {
        color = img.clone();
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
        cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
    }
    return {color, gray};
}

// ============================================================
// finalizePose
// ============================================================

void TrackerBase::finalizePose(PipelineResult& result, const PoseEstimate& pose) {
    result.R = pose.R;
    result.t = pose.t;
    result.gpnp_success = pose.success;
    result.success = pose.success;   // 统一流水线整体成功标志（终端 FAILED / 三维轴绘制依赖此值）
    result.gpnp_n_pts = pose.num_points;

    if (pose.success) {
        cv::Mat R_cv(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = pose.R(r, c);
        cv::Mat rvec;
        cv::Rodrigues(R_cv, rvec);
        if (verbose_console_) {
            std::cout << "  Pose: rvec=[" << rvec.at<double>(0) << ", "
                      << rvec.at<double>(1) << ", " << rvec.at<double>(2) << "]"
                      << "  tvec=[" << pose.t(0) << ", " << pose.t(1) << ", "
                      << pose.t(2) << "] mm  n_pts=" << pose.num_points << std::endl;
        }
        state_.R_prev = pose.R;
        state_.t_prev = pose.t;
        state_.has_cache = true;
        cache_age_ = 0;
    }
}

// ============================================================
// addLogEntry
// ============================================================

void TrackerBase::addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used) {
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
    entry.strategy_name = result.strategy_name;
    entry.is_class1 = result.is_class1;
    entry.warm_start_used = result.warm_start_used;
    entry.gate_status = static_cast<int>(result.gate_status);
    entry.seed_age = result.seed_age;
    entry.sticky_hold = result.sticky_hold;
    entry.nominated_strategy = result.nominated_strategy;
    entry.switch_reason = result.switch_reason;
    if (result.gpnp_success || result.success) {
        entry.t_x = result.t.x(); entry.t_y = result.t.y(); entry.t_z = result.t.z();
        Eigen::AngleAxisd aa(result.R);
        Eigen::Vector3d rv = aa.angle() * aa.axis();
        entry.rvec_x = rv.x(); entry.rvec_y = rv.y(); entry.rvec_z = rv.z();
    }
    state_.logs.push_back(std::move(entry));
}

// ============================================================
// printLogs
// ============================================================

void TrackerBase::printLogs() const {
    const auto& logs = state_.logs;
    if (logs.empty()) {
        if (verbose_console_) std::cout << "[Log is empty]" << std::endl;
        return;
    }

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
        row.push_back(std::to_string(static_cast<int>(log.disparity_median)));
        row.push_back(log.gpnp_success ? "OK" : "FAIL");
        for (const auto& k : used_keys) {
            auto it = log.timing.find(k);
            row.push_back(it != log.timing.end() ? std::to_string(static_cast<int>(it->second)) + "ms" : "-");
        }
        print_row(row);
    }
    print_sep('=');
}

} // namespace gpnp
