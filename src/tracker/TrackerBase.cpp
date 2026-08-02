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
// configureStrategyChain
// ============================================================

void TrackerBase::configureStrategyChain(int roi_area, bool is_class1) {
    // 选择阈值：class1 专用阈值 > 0 时使用，否则回退到通用阈值
    int akaze_thresh = akaze_min_area_;
    int tiny_thresh  = tiny_max_area_;
    if (is_class1) {
        if (akaze_min_area_class1_ > 0) akaze_thresh = akaze_min_area_class1_;
        if (tiny_max_area_class1_ > 0)  tiny_thresh  = tiny_max_area_class1_;
    }

    fallback_extractors_.clear();

    if (roi_area >= akaze_thresh || roi_area == 0) {
        extractor_ = akaze_extractor_.get();
        fallback_extractors_.push_back(binary_extractor_.get());
        fallback_extractors_.push_back(tiny_extractor_.get());
        if (verbose_console_)
            std::cout << "[TrackerBase] Strategy chain: AkazeGpnp → BinaryCorner → TinyTarget"
                      << " (roi_area=" << roi_area
                      << " akaze_thresh=" << akaze_thresh
                      << " is_class1=" << is_class1 << ")" << std::endl;
    } else if (roi_area > tiny_thresh) {
        extractor_ = binary_extractor_.get();
        fallback_extractors_.push_back(tiny_extractor_.get());
        if (verbose_console_)
            std::cout << "[TrackerBase] Strategy chain: BinaryCorner → TinyTarget"
                      << " (roi_area=" << roi_area
                      << " tiny_thresh=" << tiny_thresh
                      << " is_class1=" << is_class1 << ")" << std::endl;
    } else {
        extractor_ = tiny_extractor_.get();
        if (verbose_console_)
            std::cout << "[TrackerBase] Strategy chain: TinyTarget only"
                      << " (roi_area=" << roi_area
                      << " is_class1=" << is_class1 << ")" << std::endl;
    }
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
