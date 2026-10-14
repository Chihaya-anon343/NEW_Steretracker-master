#include "tracker/StereoTracker.hpp"
#include "common/GeometryUtils.hpp"
#include "utils/AsyncImageSaver.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <fstream>

namespace gpnp {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
bool visible(cv::Point2f p, const cv::Mat& image) {
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 &&
           p.x < image.cols && p.y < image.rows;
}
cv::Scalar pointColor(int i) {
    static const cv::Scalar colors[] = {{0,0,255},{0,255,0},{0,255,255},{255,0,0},{255,0,255},
        {255,255,0},{128,0,255},{0,128,255},{255,128,0},{128,255,0}};
    return colors[std::max(0, i) % 10];
}
cv::Mat paired(const cv::Mat& left, const cv::Mat& right) {
    cv::Mat canvas(std::max(left.rows, right.rows), left.cols + right.cols, CV_8UC3, cv::Scalar(25,25,25));
    if (!left.empty()) left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
    if (!right.empty()) right.copyTo(canvas(cv::Rect(left.cols, 0, right.cols, right.rows)));
    return canvas;
}
void savePanel(const std::string& path, cv::Mat image, const std::string& label) {
    if (image.empty()) return;
    if (image.cols > 1600) cv::resize(image, image, cv::Size(1600, std::max(1, image.rows * 1600 / image.cols)));
    std::vector<std::string> lines;std::istringstream words(label);std::string word,line;
    while(words>>word) {
        const auto candidate=line.empty()?word:line+" "+word;
        if(!line.empty()&&cv::getTextSize(candidate,cv::FONT_HERSHEY_SIMPLEX,.55,1,nullptr).width>image.cols-16) {
            lines.push_back(line);line=word;
        } else line=candidate;
    }
    if(!line.empty())lines.push_back(line);
    const int header=std::max(36,static_cast<int>(lines.size())*26+10);
    cv::Mat titled(image.rows + header, image.cols, CV_8UC3, cv::Scalar(25,25,25));
    image.copyTo(titled(cv::Rect(0, header, image.cols, image.rows)));
    for(size_t i=0;i<lines.size();++i)cv::putText(titled,lines[i],cv::Point(8,25+26*static_cast<int>(i)),
        cv::FONT_HERSHEY_SIMPLEX,.55,cv::Scalar(255,255,255),1,cv::LINE_AA);
    utils::AsyncImageSaver::write(path, titled);
}
} // namespace

bool StereoTracker::prefersLeftBcFlow(const RoiGroup& group) const {
    if (!config_.stereo_flow.enabled || !group.valid() || group.is_dual || group.is_class1) return false;
    const int area = group.primary.width * group.primary.height;
    const int raw = nominateBand(area, false);
    const auto& t = config_.temporal;
    if (!t.enabled || !has_sticky_state_ || (has_prev_c1_ && prev_use_c1_)) return raw == 2;
    const double ema = (!has_ema_ || t.area_ema_alpha <= 0)
        ? area : t.area_ema_alpha * area + (1 - t.area_ema_alpha) * ema_area_;
    const int candidate = nominateBand(static_cast<int>(ema + 0.5), false);
    if (candidate == sticky_locked_) return sticky_locked_ == 2;
    const double threshold = effectiveThresholds(false);
    const bool decisive = candidate > sticky_locked_ ? ema > threshold * t.hysteresis_up
                                                     : ema < threshold * t.hysteresis_down;
    const int confirmations = candidate == sticky_pending_ ? pending_count_ + 1 : 1;
    return (decisive && confirmations >= t.hold_frames ? candidate : sticky_locked_) == 2;
}

bool StereoTracker::prefersLeftStereoFlow(const RoiGroup& group) const {
    if (config_.stereo_flow.enabled && config_.stereo_flow.dual_roi_flow &&
        group.valid() && group.is_dual && !group.is_class1 && group.secondary.valid()) return true;
    if(config_.stereo_flow.enabled && config_.stereo_flow.disparity_only &&
       config_.stereo_flow.disparity_full_chain && group.valid() && !group.is_dual) return true;
    return prefersLeftBcFlow(group);
}

bool StereoTracker::runBcFlow(const cv::Mat& left_gray, const cv::Mat& right_gray,
                              const cv::Mat& left_color, const cv::Mat& right_color,
                              const RoiRect& roi, PipelineResult& result,
                              PoseEstimate& pose, FlowMatchResult& diagnostics) {
    result = binary_extractor_->extractMono(left_gray(cv::Rect(roi.x, roi.y, roi.width, roi.height)),
                                            left_color.empty() ? cv::Mat() : left_color(cv::Rect(roi.x, roi.y, roi.width, roi.height)));
    result.success = false; // Extraction success is not a successful camera observation.
    result.stereo_method = "LK";
    result.disparity_only = config_.stereo_flow.disparity_only;
    result.strategy_name = "BinaryCorner";
    result.left_color = left_color; result.right_color = right_color;
    result.left_roi_offset_x = roi.x; result.left_roi_offset_y = roi.y;
    for (auto& p : result.pts_left_match) { p.x += roi.x; p.y += roi.y; }
    for (auto& kp : result.kp_left) { kp.pt.x += roi.x; kp.pt.y += roi.y; }
    const auto& model = binary_extractor_->templateData().pts_3d;
    if (result.pts_left_match.size() < 4 || (!result.disparity_only && (result.good_matches.size() < 4 || model.size() < 4))) {
        result.failure_reason = "bc_extraction_or_model_failed";
        return false;
    }

    const auto& cfg = config_.stereo_flow;
    PoseEstimate initial;
    bool initial_valid = false;
    if (config_.use_initial_pnp && !cfg.disparity_only) {
        MatchResult m;
        m.good_matches = result.good_matches; m.pts_left_match = result.pts_left_match;
        m.pts_template_match = result.pts_template_match;
        const auto t = Clock::now();
        initial = initial_pnp_.solve(m, model, camera_.K);
        result.timing["initial_pnp"] = elapsed(t);
        initial_valid = checkFlowPose(initial, result, model, camera_, false,
                                     cfg.initial_reprojection_px, cfg.initial_reprojection_px).success;
    }
    std::vector<cv::Point2f> initial_right;
    if (cfg.disparity_only) initial_right = result.pts_left_match; // LK without physical-scale pose prediction.
    if (initial_valid) {
        initial_right.assign(result.pts_left_match.size(), cv::Point2f(-1, -1));
        for (const auto& m : result.good_matches) {
            if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(initial_right.size()) ||
                m.trainIdx < 0 || m.trainIdx >= static_cast<int>(model.size())) continue;
            const Eigen::Vector3d p = camera_.R_rl.transpose() * (initial.R * model[m.trainIdx] + initial.t - camera_.t_rl);
            if (!p.allFinite() || p.z() <= 1e-6) continue;
            const Eigen::Vector3d uv = camera_.K * p;
            initial_right[m.queryIdx] = cv::Point2f(static_cast<float>(uv.x()/uv.z()), static_cast<float>(uv.y()/uv.z()));
        }
    }
    auto matching_config = cfg;
    matching_config.min_matches = cfg.disparity_only ? cfg.min_matches : std::max(cfg.min_matches, config_.gpnp_min_pts);
    diagnostics = StereoFlowMatcher(camera_, matching_config).match(left_gray, right_gray,
                                                                   result.pts_left_match, initial_right,
                                                                   cv::Rect(roi.x,roi.y,roi.width,roi.height));
    result.flow_roi=diagnostics.search_roi;
    result.flow_full_fallback=diagnostics.full_frame_fallback;
    result.timing["flow_crop"]=diagnostics.crop_ms;
    result.stereo_match_ms = diagnostics.total_ms;
    result.flow_coarse_attempts = diagnostics.coarse_attempts;
    result.timing["flow_pyramid"] = diagnostics.pyramid_ms;
    result.timing["flow_coarse"] = diagnostics.coarse_ms;
    result.timing["flow_lk"] = diagnostics.lk_ms;
    result.timing["flow_check"] = diagnostics.check_ms;
    result.pts_left_good = diagnostics.left_points;
    result.pts_right_good = diagnostics.right_points;
    result.idx_from_filtered = diagnostics.left_indices;
    result.pts_left_used = result.pts_left_good; result.pts_right_used = result.pts_right_good;
    result.valid_mask.resize(result.pts_left_match.size(), false);
    for (int i : result.idx_from_filtered) result.valid_mask[i] = true;
    for (size_t i = 0; i < result.pts_left_good.size(); ++i) {
        const double d = result.pts_left_good[i].x - result.pts_right_good[i].x;
        result.disparity.push_back(d); result.dx_filtered.push_back(d);
    }
    result.n_matched = static_cast<int>(result.pts_left_good.size());
    result.disparity_success = diagnostics.success;
    result.flow_planar_status=diagnostics.planar_status;
    result.flow_planar_rejected=diagnostics.planar_rejected;
    result.flow_planar_ms=diagnostics.planar_ms;
    if (cfg.disparity_only) {
        result.failure_reason = diagnostics.failure_reason;
        if (!output_dir_.empty()) {
            std::ofstream csv(output_dir_ + "/disparity_f" + std::to_string(current_frame_) + ".csv");
            csv << "left_index,left_x,left_y,right_x,right_y,disparity_px\n";
            for (size_t i=0; i<result.disparity.size(); ++i)
                csv << result.idx_from_filtered[i] << ',' << result.pts_left_good[i].x << ','
                    << result.pts_left_good[i].y << ',' << result.pts_right_good[i].x << ','
                    << result.pts_right_good[i].y << ',' << result.disparity[i] << '\n';
        }
        return true;
    }
    if (diagnostics.success) {
        double ms = 0;
        GPnPOptions options;
        options.initial_source=PoseInitializationSource::InitialPnP;
        options.max_mean_reprojection_px=cfg.max_mean_reprojection_px;
        options.max_point_reprojection_px=cfg.max_point_reprojection_px;
        pose = gpnp_solver_.solve(result, model, initial_valid ? &initial.R : nullptr,
                                  initial_valid ? &initial.t : nullptr, ms, options);
        result.timing["gpnp"] = ms;
        const auto check = checkFlowPose(pose, result, model, camera_, true,
                                         cfg.max_mean_reprojection_px, cfg.max_point_reprojection_px);
        if (pose.success && pose.source == PoseSource::StereoGpnp && check.success) {
            result.stereo_success = true;
        } else {
            result.failure_reason = pose.source == PoseSource::LeftPnpFallback
                ? "gpnp_used_left_initial_pose" : (!check.reason.empty() ? check.reason : "gpnp_failed");
            pose = PoseEstimate{};
        }
    } else result.failure_reason = diagnostics.failure_reason;
    if (!pose.success && cfg.allow_left_pnp && initial_valid) {
        pose = initial; pose.source = PoseSource::LeftPnpFallback;
    }
    return true;
}

void StereoTracker::renderBcFlow(const PipelineResult& result, const FlowMatchResult& flow,
                                 const RoiRect& roi, const cv::Mat& left,
                                 const cv::Mat& right, const FeatureExtractor* winner) const {
    if (left.empty() || output_dir_.empty()) return;
    const std::string suffix = "_f" + std::to_string(current_frame_) + ".png";
    const cv::Mat right_view = right.empty() ? cv::Mat(left.size(), CV_8UC3, cv::Scalar(25,25,25)) : right;
    const bool tiny=winner->strategyType()==StrategyType::TinyTarget;
    const std::string file_prefix=tiny?"tiny_target":"binary_corner";
    std::ostringstream title;
    title << (tiny?"TT":"BC") << " LK | " << poseSourceName(result.pose_source) << " | pairs=" << result.pts_left_good.size()
          << "/" << result.pts_left_match.size();
    if (!result.failure_reason.empty()) title << " | " << result.failure_reason;
    if (result.disparity_only) title << " | disparity only, PnP skipped";
    if(result.disparity_only)title << " | frame_match=" << (result.disparity_success?"OK":"FAIL");
    if(result.is_class1)title << " | class1";
    if (flow.roi_used) title << " | ROI " << flow.search_roi.width << 'x' << flow.search_roi.height;
    if (flow.full_frame_fallback) title << " | full-frame retry";
    cv::Mat l = left.clone(), r = right_view.clone();
    cv::rectangle(l, cv::Rect(roi.x, roi.y, roi.width, roi.height), cv::Scalar(255,200,0), 2);
    for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
        const cv::Point2f p = result.pts_left_match[i];
        if (!visible(p, l)) continue;
        const bool accepted = i < flow.diagnostics.size() && flow.diagnostics[i].accepted;
        if (accepted) cv::circle(l, p, 5, pointColor(static_cast<int>(i)), 2, cv::LINE_AA);
        else cv::drawMarker(l, p, cv::Scalar(0,0,255), cv::MARKER_TILTED_CROSS, 12, 2);
        cv::putText(l, std::to_string(i), p + cv::Point2f(5,-5), cv::FONT_HERSHEY_SIMPLEX, .55, pointColor(static_cast<int>(i)), 1, cv::LINE_AA);
        if (i < flow.diagnostics.size() && visible(flow.diagnostics[i].initial_right, r))
            cv::drawMarker(r, flow.diagnostics[i].initial_right, cv::Scalar(150,150,150), cv::MARKER_CROSS, 10, 1);
    }
    for (size_t k = 0; k < result.pts_right_good.size(); ++k) {
        const int i = result.idx_from_filtered[k];
        const auto p = result.pts_right_good[k];
        if (!visible(p, r)) continue;
        cv::circle(r, p, 5, pointColor(i), 2, cv::LINE_AA);
        cv::putText(r, std::to_string(i), p + cv::Point2f(5,-5), cv::FONT_HERSHEY_SIMPLEX, .55, pointColor(i), 1, cv::LINE_AA);
    }
    cv::Mat correspondence = paired(l, r);
    for (size_t k = 0; k < result.pts_right_good.size(); ++k)
        cv::line(correspondence, result.pts_left_good[k], result.pts_right_good[k] + cv::Point2f(static_cast<float>(left.cols), 0),
                 pointColor(result.idx_from_filtered[k]), 1, cv::LINE_AA);
    if (visualize_detailed_) {
        savePanel(output_dir_ + "/"+file_prefix+"_stereo" + suffix, correspondence, title.str());
        if(tiny&&!tiny_extractor_->lastLeftDebug().super_binary.empty()) {
            cv::Mat binary;
            cv::cvtColor(tiny_extractor_->lastLeftDebug().super_binary,binary,cv::COLOR_GRAY2BGR);
            savePanel(output_dir_+"/tiny_target_roi_binary_l"+suffix,binary,"Left TT binary (supersampled); right uses grayscale flow");
        }
        if (!tiny && !binary_extractor_->lastLeftBinary().empty()) {
            cv::Mat binary;
            cv::cvtColor(binary_extractor_->lastLeftBinary(), binary, cv::COLOR_GRAY2BGR);
            for (size_t i = 0; i < result.pts_left_match.size(); ++i)
                cv::circle(binary, result.pts_left_match[i] - cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)),
                           3, pointColor(static_cast<int>(i)), 1);
            savePanel(output_dir_ + "/binary_corner_binary" + suffix, paired(binary, r), "Left: BC binary | Right: grayscale matching (no BC)");
        }
        if (!tiny && !binary_extractor_->lastUprightBinary().empty()) {
            cv::Mat upright;
            cv::cvtColor(binary_extractor_->lastUprightBinary(), upright, cv::COLOR_GRAY2BGR);
            savePanel(output_dir_ + "/binary_corner_upright" + suffix, upright, "Left BC upright shape");
        }
        const auto* tmpl = tiny?tiny_extractor_->lastMatchedTemplate():binary_extractor_->lastMatchedTemplate();
        if (tmpl && !tmpl->image.empty()) {
            cv::Mat template_view;
            cv::cvtColor(tmpl->image, template_view, cv::COLOR_GRAY2BGR);
            cv::resize(template_view, template_view, cv::Size(400,400), 0, 0, cv::INTER_NEAREST);
            for (size_t i = 0; i < tmpl->corners.size(); ++i) {
                cv::Point2f p(tmpl->corners[i].x * 400 / tmpl->image.cols, tmpl->corners[i].y * 400 / tmpl->image.rows);
                cv::circle(template_view, p, 4, pointColor(static_cast<int>(i)), 2);
                cv::putText(template_view, std::to_string(i), p + cv::Point2f(4,-4), cv::FONT_HERSHEY_SIMPLEX, .5, pointColor(static_cast<int>(i)), 1);
            }
            savePanel(output_dir_ + "/"+file_prefix+"_template" + suffix, paired(l, template_view), "Left point indices | matched template");
        }
    }
    cv::Mat axes = left.clone();
    const auto& model = winner->templateData().pts_3d;
    if (result.success) {
        cv::Mat k = (cv::Mat_<double>(3,3) << camera_.K(0,0),camera_.K(0,1),camera_.K(0,2),
            camera_.K(1,0),camera_.K(1,1),camera_.K(1,2),camera_.K(2,0),camera_.K(2,1),camera_.K(2,2));
        cv::Mat rotation(3,3,CV_64F), rvec, tvec = (cv::Mat_<double>(3,1) << result.t.x(), result.t.y(), result.t.z());
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) rotation.at<double>(i,j)=result.R(i,j);
        cv::Rodrigues(rotation, rvec);
        std::vector<cv::Point3d> axis_model{{0,0,0},{100,0,0},{0,100,0},{0,0,100}};
        std::vector<cv::Point2d> projected;
        cv::projectPoints(axis_model, rvec, tvec, k, cv::noArray(), projected);
        for (int i=1;i<4;++i)
            if (visible(projected[0], axes) && visible(projected[i], axes))
                cv::line(axes, projected[0], projected[i], i==1 ? cv::Scalar(0,0,255) : i==2 ? cv::Scalar(0,255,0) : cv::Scalar(255,0,0), 3);
        if (visualize_detailed_) {
            cv::Mat reproj_l = l.clone(), reproj_r = r.clone();
            for (const auto& m : result.good_matches) {
                if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(result.pts_left_match.size()) ||
                    m.trainIdx < 0 || m.trainIdx >= static_cast<int>(model.size())) continue;
                const Eigen::Vector3d pc = result.R * model[m.trainIdx] + result.t;
                const Eigen::Vector3d pr = camera_.R_rl.transpose() * (pc - camera_.t_rl);
                if (pc.z() > 1e-6) {
                    const Eigen::Vector3d u = camera_.K * pc;
                    cv::Point2f p(static_cast<float>(u.x()/u.z()), static_cast<float>(u.y()/u.z()));
                    if (visible(p, reproj_l)) {
                        cv::drawMarker(reproj_l, p, cv::Scalar(255,255,255), cv::MARKER_CROSS, 12, 2);
                        cv::line(reproj_l, p, result.pts_left_match[m.queryIdx], pointColor(m.queryIdx), 1);
                    }
                }
                if (pr.z() > 1e-6) {
                    const Eigen::Vector3d u = camera_.K * pr;
                    cv::Point2f p(static_cast<float>(u.x()/u.z()), static_cast<float>(u.y()/u.z()));
                    if (visible(p, reproj_r)) cv::drawMarker(reproj_r, p, cv::Scalar(255,255,255), cv::MARKER_CROSS, 12, 2);
                }
            }
            savePanel(output_dir_ + "/binary_corner_axes" + suffix, axes, title.str());
            savePanel(output_dir_ + "/binary_corner_reproj" + suffix, paired(reproj_l,reproj_r), "Measured points: colored | pose projections: white");
            savePanel(output_dir_ + "/binary_corner_stereo_proj" + suffix, reproj_r, "Right: measured LK points and pose projections");
        }
    }
    savePanel(output_dir_ + "/stereo" + suffix, result.disparity_only ? correspondence : result.success ? axes : left.clone(), title.str());
    if (visualize_detailed_ && !flow.diagnostics.empty()) {
        cv::Mat diagnostic(static_cast<int>(flow.diagnostics.size()) * 25 + 35, 1280, CV_8UC3, cv::Scalar(25,25,25));
        cv::putText(diagnostic, "ID  decision   FB(px)   epi(px)   NCC   coarse   plane residual(px)", {8,24}, cv::FONT_HERSHEY_SIMPLEX, .55, {255,255,255}, 1);
        for (size_t i=0;i<flow.diagnostics.size();++i) {
            const auto& d=flow.diagnostics[i];
            std::ostringstream row;
            row << i << "  " << d.reason << "  fb=" << d.fb_error << " epi=" << d.epipolar_error
                << " ncc=" << d.ncc << " coarse=" << d.coarse_attempted << " plane=" << d.planar_error_px;
            if (i < result.pts_left_match.size() && visible(d.tracked_right, right_view))
                row << " disp=" << result.pts_left_match[i].x - d.tracked_right.x;
            cv::putText(diagnostic, row.str(), {8,49+static_cast<int>(i)*25}, cv::FONT_HERSHEY_SIMPLEX, .48,
                        d.accepted ? cv::Scalar(100,255,100) : cv::Scalar(100,100,255), 1);
        }
        savePanel(output_dir_ + "/"+file_prefix+"_flow_status" + suffix, diagnostic, title.str());
    }
}
} // namespace gpnp
