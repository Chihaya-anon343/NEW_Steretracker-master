/**
 * @file MonoTracker.cpp
 * @brief 单目追踪器实现。
 */

#include "tracker/MonoTracker.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "utils/AsyncImageSaver.hpp"
#include "utils/PoseUtils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace gpnp {

namespace {
int physicalCornerId(const PipelineResult &result, size_t index) {
  return index < result.good_matches.size()
             ? result.good_matches[index].trainIdx
             : (int)index;
}
void saveMonoIdentityPanel(const BinaryCornerExtractor& extractor,const PipelineResult& local,
                           const std::string& path) {
    const auto& d=extractor.lastMonoIdentity();
    if(d.status=="disabled" || d.reference.empty()) return;
    if(!d.foreground.empty()) {
        const int size=256;
        auto thumbnail=[&](const cv::Mat& input) {
            cv::Mat panel(size,size,CV_8UC3,cv::Scalar(20,20,20));
            if(input.empty()) return panel;
            cv::Mat color;if(input.channels()==1) cv::cvtColor(input,color,cv::COLOR_GRAY2BGR);else color=input;
            const double scale=std::min((double)size/color.cols,(double)size/color.rows);
            cv::Mat small;cv::resize(color,small,{},scale,scale,cv::INTER_NEAREST);
            small.copyTo(panel(cv::Rect(0,0,small.cols,small.rows)));return panel;
        };
        cv::Mat original=local.left_color.empty()?d.foreground:local.left_color;
        // Use display coordinates so labels do not cover a tiny source ROI.
        cv::Mat refinement=thumbnail(original);
        const float display_scale=std::min((float)size/original.cols,(float)size/original.rows);
        for(size_t i=0;i<d.points.size();++i) {
            cv::circle(refinement,d.before_refinement[i]*display_scale,3,{0,255,255},1);
            cv::circle(refinement,d.points[i]*display_scale,2,{0,0,255},-1);
        }
        cv::Mat numbered=d.coarse_normalized.empty()?cv::Mat::zeros(d.reference.size(),CV_8U):d.coarse_normalized;
        cv::cvtColor(numbered,numbered,cv::COLOR_GRAY2BGR);
        cv::Mat verified=d.normalized.empty()?cv::Mat::zeros(d.reference.size(),CV_8U):d.normalized;
        cv::cvtColor(verified,verified,cv::COLOR_GRAY2BGR);
        if(d.accepted && !d.image_to_template.empty()) {
            std::vector<cv::Point2f> pts;cv::perspectiveTransform(d.points,pts,d.image_to_template);
            for(size_t i=0;i<pts.size();++i) {
                cv::circle(verified,pts[i],2,{0,0,255},-1);
                const int id =
                    i < d.template_ids.size() ? d.template_ids[i] : (int)i;
                cv::putText(
                    verified, std::to_string(id),
                    {std::clamp(cvRound(pts[i].x) + 2, 0, verified.cols - 10),
                     std::clamp(cvRound(pts[i].y) - 2, 8, verified.rows - 2)},
                    cv::FONT_HERSHEY_SIMPLEX, .3, {0, 255, 0}, 1);
            }
        }
        for (size_t i = 0; i < d.rejected_points.size(); ++i) {
          const cv::Point2f point=d.rejected_points[i]*display_scale;
          cv::drawMarker(refinement, point, {0, 140, 255},
                         cv::MARKER_TILTED_CROSS, 9, 1);
          const int id = i < d.rejected_ids.size() ? d.rejected_ids[i] : -1;
          cv::putText(refinement, "reject " + std::to_string(id),
                      {std::clamp(cvRound(point.x)+5,0,size-65),std::clamp(cvRound(point.y)-5,12,size-3)},
                      cv::FONT_HERSHEY_SIMPLEX, .35, {0, 140, 255}, 1);
        }
        std::vector<cv::Mat> panels{thumbnail(original),thumbnail(d.foreground),thumbnail(numbered),
            thumbnail(d.reference),thumbnail(refinement),thumbnail(verified)};
        const char* labels[]={"ROI","selected foreground","four-way coarse","canonical model","original edge refinement","verified H / physical IDs"};
        for(int i=0;i<6;++i) {
            cv::copyMakeBorder(panels[i],panels[i],24,0,0,0,cv::BORDER_CONSTANT,{28,28,28});
            cv::putText(panels[i],labels[i],{5,17},cv::FONT_HERSHEY_SIMPLEX,.44,{230,230,230},1);
        }
        cv::Mat first,second,all;cv::hconcat(std::vector<cv::Mat>{panels[0],panels[1],panels[2]},first);
        cv::hconcat(std::vector<cv::Mat>{panels[3],panels[4],panels[5]},second);cv::vconcat(first,second,all);
        cv::copyMakeBorder(all,all,48,0,0,0,cv::BORDER_CONSTANT,{28,28,28});
        std::ostringstream line;line<<d.status<<" "<<d.path<<" IoU="<<d.iou<<" fit="<<d.fit_error_px<<"px";
        cv::putText(all,line.str(),{8,18},cv::FONT_HERSHEY_SIMPLEX,.48,d.accepted?cv::Scalar(0,255,0):cv::Scalar(0,120,255),1);
        line.str("");line<<"refined="<<d.refined_points<<"/10  normalize/corners/refine/check(ms): "
            <<d.normalization_ms<<" / "<<d.corners_ms<<" / "<<d.refinement_ms<<" / "<<d.validation_ms;
        cv::putText(all,line.str(),{8,39},cv::FONT_HERSHEY_SIMPLEX,.4,{230,230,230},1);
        utils::AsyncImageSaver::write(path,all);return;
    }
    const int size=d.reference.rows;
    std::vector<cv::Mat> panels;
    for(const auto* source:{&d.reference,&d.coarse_normalized,&d.normalized}) {
        cv::Mat gray=source->empty() ? cv::Mat::zeros(size,size,CV_8U) : *source;
        cv::Mat color;cv::cvtColor(gray,color,cv::COLOR_GRAY2BGR);panels.push_back(color);
    }
    if(d.accepted && !d.image_to_template.empty()) {
        std::vector<cv::Point2f> normalized;
        cv::perspectiveTransform(local.pts_left_match,normalized,d.image_to_template);
        for(size_t i=0;i<normalized.size();++i) {
            cv::circle(panels[2],normalized[i],2,{0,0,255},-1);
            const cv::Point label(std::clamp(cvRound(normalized[i].x)+2,0,size-10),
                                  std::clamp(cvRound(normalized[i].y)-2,8,size-2));
            const int id = i < local.good_matches.size()
                               ? local.good_matches[i].trainIdx
                               : (int)i;
            cv::putText(panels[2], std::to_string(id), label,
                        cv::FONT_HERSHEY_SIMPLEX, .3, {0, 255, 0}, 1);
        }
    }
    cv::Mat row;cv::hconcat(panels,row);cv::resize(row,row,{},3,3,cv::INTER_NEAREST);
    cv::copyMakeBorder(row,row,54,0,0,0,cv::BORDER_CONSTANT,{28,28,28});
    cv::putText(row,"reference | TT four-way | refined H / physical IDs",{8,20},cv::FONT_HERSHEY_SIMPLEX,.55,{230,230,230},1);
    std::ostringstream line;line<<d.status<<" "<<d.path<<" IoU="<<d.iou<<" fit="<<d.fit_error_px<<"px gap="<<d.score_gap;
    cv::putText(row,line.str(),{8,43},cv::FONT_HERSHEY_SIMPLEX,.5,d.accepted?cv::Scalar(0,255,0):cv::Scalar(0,120,255),1);
    utils::AsyncImageSaver::write(path,row);
}
} // namespace

PipelineResult
MonoTracker::processCandidates(const cv::Mat &image, bool visualize,
                               const std::vector<RoiCandidate> &choices) {
  const auto started = std::chrono::steady_clock::now();
  has_prepared_bc_ = has_prepared_tt_ = false;
  struct PreparedGuard {
    bool &bc;
    bool &tt;
    ~PreparedGuard() {
      bc = false;
      tt = false;
    }
  } prepared_guard{has_prepared_bc_, has_prepared_tt_};
  auto ranked = choices;
  const int roi_clock =
      current_frame_ > 0 ? current_frame_ : state_.frame_count;
  const int age = roi_clock - verified_roi_frame_;
  if (has_verified_roi_ && config_.roi_selection.history_max_age > 0 &&
      age <= config_.roi_selection.history_max_age) {
    const cv::Rect a(verified_roi_.x, verified_roi_.y, verified_roi_.width,
                     verified_roi_.height);
    for (auto &candidate : ranked) {
      const auto &r = candidate.group.primary;
      cv::Rect b(r.x, r.y, r.width, r.height);
      const double intersection = (a & b).area(),
                   uni = a.area() + b.area() - intersection;
      if (uni > 0)
        candidate.score +=
            .10 * (1. - (double)age / config_.roi_selection.history_max_age) *
            intersection / uni;
    }
    std::stable_sort(
        ranked.begin(), ranked.end(),
        [](const auto &a, const auto &b) { return a.score > b.score; });
  }
  binary_extractor_->setDebugCapture(visualize);
  tiny_extractor_->setDebugCapture(visualize);
  int checked = 0, selected = -1;
  std::vector<std::string> reasons;
  for (int i = 0;
       i < std::min((int)ranked.size(), config_.roi_selection.max_candidates);
       ++i) {
    const auto &group = ranked[i].group;
    auto roi = group.primary;
    bool class1 = group.is_class1;
    const bool short_primary =
        group.is_dual && config_.dual_roi_class1_fallback &&
        config_.dual_roi_primary_span_ratio > 0 &&
        roi.width >= image.cols * config_.dual_roi_primary_span_ratio &&
        roi.height >= image.rows * config_.dual_roi_primary_span_ratio;
    if (short_primary) {
      roi = group.secondary;
      class1 = true;
    }
    const cv::Rect rc(roi.x, roi.y, roi.width, roi.height),
        frame(0, 0, image.cols, image.rows);
    if (image.empty() || rc.area() <= 0 || (rc & frame) != rc) {
      reasons.push_back("invalid_bounds");
      continue;
    }
    ++checked;
    cv::Mat gray;
    if (image.channels() == 1)
      gray = image(rc);
    else
      cv::cvtColor(image(rc), gray, cv::COLOR_BGR2GRAY);
    binary_extractor_->setUseClass1(class1);
    tiny_extractor_->setUseClass1(class1);
    auto credible = [&](const PipelineResult &r, double shape) {
      if (!r.success || r.pts_left_match.size() < 4 || shape < .75)
        return false;
      float x0 = r.pts_left_match[0].x, x1 = x0, y0 = r.pts_left_match[0].y,
            y1 = y0;
      for (auto p : r.pts_left_match) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
          return false;
        x0 = std::min(x0, p.x);
        x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y);
        y1 = std::max(y1, p.y);
      }
      return std::min(x1 - x0, y1 - y0) >=
                 config_.roi_selection.min_observed_span_px &&
             x1 - x0 >= roi.width * config_.roi_selection.min_component_ratio &&
             y1 - y0 >= roi.height * config_.roi_selection.min_component_ratio;
    };
    const int tiny = class1 && tiny_max_area_class1_ > 0 ? tiny_max_area_class1_
                                                         : tiny_max_area_;
    bool good = false;
    if (roi.width * roi.height > tiny) {
      prepared_bc_ = binary_extractor_->extractMono(gray, image(rc));
      has_prepared_bc_ = !short_primary;
      good = credible(prepared_bc_, binary_extractor_->lastMatchOverlap());
    }
    if (!good) {
      prepared_tt_ = tiny_extractor_->extractMono(gray, image(rc));
      has_prepared_tt_ = !short_primary;
      good = credible(prepared_tt_, tiny_extractor_->lastMatchOverlap());
    }
    if (good) {
      selected = i;
      break;
    }
    has_prepared_bc_ = has_prepared_tt_ = false;
    reasons.push_back("appearance_or_observation_invalid");
  }
  binary_extractor_->setUseClass1(false);
  tiny_extractor_->setUseClass1(false);
  const double verify_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
  PipelineResult result;
  if (selected >= 0) {
    result = process(image, visualize, &ranked[selected].group);
    result.roi_containment = ranked[selected].containment;
    if (result.success) {
      has_verified_roi_ = true;
      verified_roi_ = ranked[selected].group.primary;
      verified_roi_frame_ = roi_clock;
    }
  } else {
    result.strategy_name = "Skip_InvalidRoi";
    result.failure_reason = "roi_appearance_rejected";
    onFrameSkipped();
    addLogEntry(result, state_.frame_count == 0, false);
    ++state_.frame_count;
  }
  has_prepared_bc_ = has_prepared_tt_ = false;
  result.roi_candidates = (int)ranked.size();
  result.roi_verified = checked;
  result.roi_selected = selected;
  result.roi_selection_reason =
      selected >= 0 ? "geometry_and_appearance" : "no_verified_target";
  result.timing["roi_verify"] = verify_ms;
  result.extract_ms += verify_ms;
  // process() creates one frame log before ROI selection's enclosing work is
  // known. Include that work once in its aggregate columns.
  if (!state_.logs.empty()) {
    auto &log = state_.logs.back();
    log.extract_ms = result.extract_ms;
    log.pnp_ms = result.pnp_ms;
    log.total_time_ms = result.extract_ms + result.pnp_ms;
    log.timing = result.timing;
  }
  if (verbose_console_)
    std::cout << "[MonoROI] candidates=" << ranked.size()
              << " verified=" << checked << " selected=" << selected
              << " reason=" << result.roi_selection_reason << "\n";
  if (visualize && !output_dir_.empty() && !image.empty()) {
    cv::Mat canvas = image.clone();
    for (int i = 0; i < (int)ranked.size(); ++i) {
      const auto &r = ranked[i].group.primary;
      const cv::Scalar color =
          i == selected ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 100, 255);
      cv::rectangle(canvas, {r.x, r.y, r.width, r.height}, color, 2);
      cv::putText(canvas,
                  "ROI " + std::to_string(i) +
                      (i == selected ? " selected" : " rejected/untried"),
                  {r.x, std::max(15, r.y - 5)}, cv::FONT_HERSHEY_SIMPLEX, .5,
                  color, 1);
    }
    utils::AsyncImageSaver::write(output_dir_ + "/mono_roi_selection_f" +
                                      std::to_string(current_frame_) + ".png",
                                  canvas);
  }
  return result;
}

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
    : mono_pnp_(config.mono_pose_prior)
{
    Eigen::Matrix3d R_rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rl(0, 0, 0);  // 单目无基线
    camera_ = makeStereoCameraParams(K, R_rl, t_rl);
    config_ = config;

    initExtractors(template_path, binary_cfg, binary_template_dir, tiny_cfg, tiny_template_dir);
    binary_extractor_->setMonoContext(true);
}

// ============================================================
// prepareDualBcTemplate
// ============================================================

void MonoTracker::prepareDualBcTemplate() {
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

    // Rebuild each group in its local physical frame, then place it once.
    // The annotation's previous board-space center must not be added twice.
    placeTemplateInClass0(dual_bc_tmpl_pts3d_, false, config_.template_geometry);
    placeTemplateInClass0(dual_c1_tmpl_pts3d_, true, config_.template_geometry);
    dual_bc_template_ready_ = true;
    if (verbose_console_)
        std::cout << "[DualRoi] BC template ready: c0=" << dual_bc_tmpl_corners_.size()
                  << " pts (" << (c0_from_anno ? "anno" : "auto")
                  << ")  c1=" << dual_c1_tmpl_corners_.size() << " pts"
                  << "  board=" << tw << "x" << th
                  << "  real_size=" << real_w << "x" << real_h << "mm"
                  << "\n";
}

// ============================================================
// processDualRoi
// ============================================================

PipelineResult MonoTracker::processDualRoi(const cv::Mat& left_img,
                                                  const RoiGroup& left_group,
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
    bool t1_cold_recovered = false;

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
            output_dir_ + "/dual_roi_mono_bc_process_f" + std::to_string(current_frame_) + ".png",
            strip);
    };

    // 0. Ensure template preprocessing is done
    prepareDualBcTemplate();

    // ---- Load image (P3: 不做全图转换/复制, 灰度在裁剪处按 ROI 转换) ----
    if (left_img.empty()) {
        PipelineResult empty;
        empty.is_first_frame = !state_.has_cache;
        return empty;
    }

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

    // P1 巨型 primary 短路: primary 边长覆盖画幅 ≥ primary_span_ratio 时目标外角点已出
    // 视野, BC 角点必败 (Tier1 的 PnP 必失败) 且全图级 BC 提取极耗时 → 跳过 Tier1
    // 直接走 Tier3 class1 链。Tier3 开关关闭时保持旧路径。
    const bool primary_spans_frame =
        !kDualRoiTier1Only &&   // 消融: 关闭短路, 巨型 primary 也走 Tier1
        config_.dual_roi_primary_span_ratio > 0.0 &&
        config_.dual_roi_class1_fallback &&
        left_pri.width  >= left_img.cols * config_.dual_roi_primary_span_ratio &&
        left_pri.height >= left_img.rows * config_.dual_roi_primary_span_ratio;

    cv::Point2d sec_to_pri_offset(
        static_cast<double>(left_sec.x - left_pri.x),
        static_cast<double>(left_sec.y - left_pri.y));

    if (verbose_console_)
        std::cout << "[DualRoi][Mono] pad=" << pad
                  << "  primary=" << left_pri.width << "x" << left_pri.height
                  << "  secondary(raw)=" << left_group.secondary.width << "x" << left_group.secondary.height
                  << "  secondary(expanded)=" << left_sec.width << "x" << left_sec.height
                  << "  offset=(" << sec_to_pri_offset.x << "," << sec_to_pri_offset.y << ")"
                  << "\n";

    if (primary_spans_frame && verbose_console_)
        std::cout << "[DualRoi][Mono] primary spans frame ("
                  << left_pri.width << "x" << left_pri.height << " vs "
                  << left_img.cols << "x" << left_img.rows
                  << "), skip Tier1 -> class1 chain" << "\n";
    if (primary_spans_frame)
        routeStep("SHORT-CIRCUIT(primary spans frame, skip Tier1)");

    // 2. Crop images (left only) —— 灰度按 ROI 转换 (P3); 彩色引用共享 (画图面板均先 clone)
    cv::Mat left_c0_gray, left_c0_color;
    if (!primary_spans_frame) {
        cv::Rect pri_rc(left_pri.x, left_pri.y, left_pri.width, left_pri.height);
        cv::cvtColor(left_img(pri_rc), left_c0_gray, cv::COLOR_BGR2GRAY);
        left_c0_color = left_img(pri_rc);
    }
    cv::Rect sec_rc(left_sec.x, left_sec.y, left_sec.width, left_sec.height);
    cv::Mat left_c1_gray;
    cv::cvtColor(left_img(sec_rc), left_c1_gray, cv::COLOR_BGR2GRAY);
    cv::Mat left_c1_color = left_img(sec_rc);

    // 可视化/面板用全帧引用（不复制；所有画图处均先 clone）
    cv::Mat left_color_orig = left_img;

    // ---- 提取/解算累加器与下游共用变量（短路时部分保持空/失败初值）----
    PipelineResult result_bc;
    int n_bc = 0, n_bc_use = 0, total_use = 0;
    std::vector<int> pair_tmpl_idx;   // 每个合并点对应的标注模板点索引 (c0: 0..n0-1, c1: n0..)
    bool joint_used = false;          // Tier1b 联合解算胜出
    std::vector<cv::Point2f> merged_pts_2d;
    std::vector<cv::KeyPoint> merged_kp_left;
    std::vector<Eigen::Vector3d> merged_pts_3d;
    cv::Point2d left_off(static_cast<double>(left_pri.x), static_cast<double>(left_pri.y));
    PoseSeed seed;
    const bool use_seed = seedActive();
    if (use_seed) { seed.R = state_.R_prev; seed.t = state_.t_prev; }
    PoseEstimate pose;                 // 短路时保持失败 → 自然落入 8c class1 链

    if (!primary_spans_frame) {

    // 3. BC(class0) 提取 —— class1 部分不再用 AKAZE, 退化时由 Tier3 class1 链的 BC 接管
    auto t_extract = std::chrono::steady_clock::now();
    if (has_prepared_bc_) {
      result_bc = std::move(prepared_bc_);
      has_prepared_bc_ = false;
    } else
      result_bc = binary_extractor_->extractMono(left_c0_gray, left_c0_color);
    if(visualize && !output_dir_.empty()) saveMonoIdentityPanel(*binary_extractor_,result_bc,
        output_dir_+"/mono_identity_c0_f"+std::to_string(current_frame_)+".png");
    n_bc = static_cast<int>(result_bc.pts_left_match.size());
    if (verbose_console_) std::cout << "[DualRoi][Mono] BinaryCorner on class 0: " << n_bc << " corners" << "\n";
    extract_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_extract).count();

    auto offsetPoints = [](std::vector<cv::Point2f>& pts, const cv::Point2d& offset) {
        float fx = static_cast<float>(offset.x), fy = static_cast<float>(offset.y);
        for (auto& p : pts) { p.x += fx; p.y += fy; }
    };

    // 4. BC 角点 ↔ 板系 3D 对应
    //    BC 输出 pts_left_match[k] 已是规范序（matchCorners 用同一参考角 θ+CPSAGL 同步
    //    重排模板/图像两侧，起点=基准角点，与匹配角无关），直接索引 dual_bc_tmpl_pts3d_ 即可。
    //    严格数量守卫: 运行时角点数 ≠ 模板点数 → c0 侧整体剔除（前缀 min 配对在
    //    数量不一致时是错位对应, 4-vs-10 类垃圾解的根源）
    int n_bc_3d = static_cast<int>(dual_bc_tmpl_pts3d_.size());
    const bool explicit_ids = result_bc.bc_detected_count > 0;
    if (!explicit_ids && n_bc_3d > 0 && n_bc != n_bc_3d) {
      std::cerr << "[DualRoi][Mono] BC corner count mismatch (" << n_bc
                << " vs template " << n_bc_3d << "), dropping c0 side"
                << "\n";
    }
    n_bc_use = explicit_ids ? (int)result_bc.good_matches.size()
                            : ((n_bc == n_bc_3d) ? n_bc : 0);
    for (int i = 0; i < n_bc_use; ++i) {
      const int id = explicit_ids ? result_bc.good_matches[i].trainIdx : i;
      if (id < 0 || id >= n_bc_3d)
        continue;
      merged_pts_2d.push_back(result_bc.pts_left_match[i]);
      merged_kp_left.emplace_back(result_bc.pts_left_match[i], 1.0f);
      merged_pts_3d.push_back(dual_bc_tmpl_pts3d_[id]);
      pair_tmpl_idx.push_back(id);
    }
    n_bc_use = (int)merged_pts_3d.size();
    total_use = n_bc_use;

    // 4b. 角点不足 → 跳过 Tier1, 落入 class1 链
    //     (BC 面板/Route 输出由函数尾部统一处理)
    if (total_use < 4) {
        std::cerr << "[DualRoi][Mono] Too few BC corners (" << total_use
                  << "), skipping Tier1, trying class1 chain" << "\n";
        routeStep("Tier1(BC) SKIP[pts=" + std::to_string(total_use) + " <4]");
    }

    // 5~6. 仅角点可解时执行: 坐标还原 + Tier1 BC(class0) 解算
    // (角点<4 或短路时 pose 保持失败 → 8c class1 链成为唯一路径)
    if (total_use >= 4) {

    // 5. Restore full-image coordinates
    offsetPoints(merged_pts_2d, left_off);
    for (auto& kp : merged_kp_left) {
        kp.pt.x += static_cast<float>(left_off.x);
        kp.pt.y += static_cast<float>(left_off.y);
    }

    // 6. Tier1: BC(class0)-only 解算 (原 Tier2 BC-only 提升为主级; MonoPnP 多候选择优)
    auto t_pnp = std::chrono::steady_clock::now();
    pose = use_seed ? mono_pnp_.solve(merged_pts_2d, merged_pts_3d, camera_.K,
                                      &seed, config_.temporal.tie_epsilon_px,
                                      result_bc.pose_min_inliers)
                    : mono_pnp_.solve(merged_pts_2d, merged_pts_3d, camera_.K,
                                      nullptr, 0, result_bc.pose_min_inliers);
    pnp_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_pnp).count();

    // 6a. 运动门控 (方案C): Tier1 解通过重投影校验但相对 seed 跳变超阈值
    //     (典型: 近距离 BC 角点误锁 class1 内图案, ~43px 跨度配 class0 尺度 3D,
    //      15x 尺度偏差仍能过 8px 平坦重投影) → 判失败, 交由退化链 (Tier3) 接管。
    //     陈旧 seed (class1 帧不解入缓存时连续多帧不刷新) 放宽阈值, 让恢复帧的合理解通过。
    if (pose.success && use_seed &&
        !motionGatePass(pose, seed, nullptr, false, "DualRoi-T1", true)) {
        // 冷启动复核（与普通链 processMono 同语义）: 陈旧 seed 误伤恢复帧时，
        // 无 seed 重解一次；冷解成功 = 纯重投影择优的可信路径，按「seed 过期」
        // 采纳，不再被陈旧 seed 锚定的门控二次审查。冷解也失败才真正判
        // motion-gate 交由退化链 (Tier3) 接管
        auto t_cold = std::chrono::steady_clock::now();
        PoseEstimate pc =
            mono_pnp_.solve(merged_pts_2d, merged_pts_3d, camera_.K, nullptr,
                            0.0, result_bc.pose_min_inliers);
        pnp_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_cold).count();
        if (pc.success) {
            pose = pc;
            t1_cold_recovered = true;
            if (verbose_console_)
                std::cout << "  [DualRoi][Mono] Tier1 warm pose rejected, "
                             "cold re-solve accepted" << "\n";
        } else {
            if (verbose_console_)
                std::cout << "  [DualRoi][Mono] Tier1 pose rejected by motion gate" << "\n";
            t1_reason = "motion-gate";
            pose = PoseEstimate{};
        }
    }
    if (t1_reason.empty() && !pose.success) t1_reason = "mono-pnp";
    routeStep(pose.success
        ? "Tier1(BC c0 n=" + std::to_string(n_bc_use) + ") OK"
        : "Tier1(BC c0 n=" + std::to_string(n_bc_use) + ") FAIL[" + t1_reason + "]");

    // ================================================================
    // 6b. Tier1b: class1 增广联合解算
    //     用 P0 把全部标注 3D 点 (c0+c1) 投影到图像, 提取角点 (c0+c1)
    //     与投影点做贪心最近邻匹配 (门限内, 观测/3D 双向唯一) —— 顺序无关,
    //     角点数波动/多余角点由门限与唯一性自然剔除。匹配对多于 Tier1a
    //     时联合重解 (P0 作 seed), 否则保留 P0。
    // ================================================================
    if (pose.success && !dual_c1_tmpl_pts3d_.empty() && !left_c1_gray.empty()) {
        auto t_c1_ext = std::chrono::steady_clock::now();
        // Secondary observations are class1 even though their 3D coordinates
        // come from the explicit joint model below. Restore class0 immediately
        // so later primary frames and visualization keep their original scale.
        binary_extractor_->setUseClass1(true);
        PipelineResult r_c1 = binary_extractor_->extractMono(left_c1_gray, left_c1_color);
        binary_extractor_->setUseClass1(false);
        if(visualize && !output_dir_.empty()) saveMonoIdentityPanel(*binary_extractor_,r_c1,
            output_dir_+"/mono_identity_joint_c1_f"+std::to_string(current_frame_)+".png");
        extract_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_c1_ext).count();
        int n_c1 = static_cast<int>(r_c1.pts_left_match.size());

        if (n_c1 < 4) {
            routeStep("Tier1b(BCJ) SKIP[c1 corners=" + std::to_string(n_c1) + " <4]");
        } else {
            // 观测集: c0 角点 + c1 角点 (全图坐标)
            std::vector<cv::Point2f> obs;
            std::vector<char> obs_is_c1;
            obs.reserve(static_cast<size_t>(n_bc) + n_c1);
            obs_is_c1.reserve(static_cast<size_t>(n_bc) + n_c1);
            for (const auto& p : result_bc.pts_left_match) {
                obs.emplace_back(p.x + static_cast<float>(left_off.x),
                                 p.y + static_cast<float>(left_off.y));
                obs_is_c1.push_back(0);
            }
            const float sox = static_cast<float>(left_sec.x);
            const float soy = static_cast<float>(left_sec.y);
            for (const auto& p : r_c1.pts_left_match) {
                obs.emplace_back(p.x + sox, p.y + soy);
                obs_is_c1.push_back(1);
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
                  if ((obs_is_c1[i] != 0) !=
                      (j >= (int)dual_bc_tmpl_pts3d_.size()))
                    continue;
                  if (!obs_is_c1[i] && explicit_ids &&
                      i < (int)result_bc.good_matches.size() &&
                      j != result_bc.good_matches[i].trainIdx)
                    continue;
                  double dx = obs[i].x - pred[j].x;
                  double dy = obs[i].y - pred[j].y;
                  double d = std::sqrt(dx * dx + dy * dy);
                  if (d <= gate)
                    cands.push_back({d, i, j});
                }
            }
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b) { return a.d < b.d; });
            std::vector<char> obs_used(obs.size(), 0), tmpl_used(pred.size(), 0);
            std::vector<cv::Point2f> j2d_c0, j2d_c1;
            std::vector<int> jidx_c0, jidx_c1;
            for (const auto& c : cands) {
                if (obs_used[c.obs_i] || tmpl_used[c.tmpl_j]) continue;
                obs_used[c.obs_i] = 1;
                tmpl_used[c.tmpl_j] = 1;
                if (obs_is_c1[c.obs_i]) { j2d_c1.push_back(obs[c.obs_i]); jidx_c1.push_back(c.tmpl_j); }
                else                    { j2d_c0.push_back(obs[c.obs_i]); jidx_c0.push_back(c.tmpl_j); }
            }

            const int n_match = static_cast<int>(j2d_c0.size() + j2d_c1.size());
            if (n_match < 4 || n_match <= total_use) {
                routeStep("Tier1b(BCJ) SKIP[match=" + std::to_string(n_match)
                          + " pts, no gain]");
            } else {
                // 联合 2D/3D (c0 对在前, c1 对在后)
                std::vector<cv::Point2f> jp2d = j2d_c0;
                jp2d.insert(jp2d.end(), j2d_c1.begin(), j2d_c1.end());
                std::vector<Eigen::Vector3d> jp3d;
                jp3d.reserve(jp2d.size());
                for (int j : jidx_c0) jp3d.push_back(all_3d[j]);
                for (int j : jidx_c1) jp3d.push_back(all_3d[j]);

                // P0 作 seed 联合重解; 运动门控与 Tier1a 同语义 (拒绝→冷重解→判败)
                PoseSeed p0_seed;
                p0_seed.R = pose.R;
                p0_seed.t = pose.t;
                auto t_j = std::chrono::steady_clock::now();
                PoseEstimate pj = mono_pnp_.solve(jp2d, jp3d, camera_.K, &p0_seed,
                                                  config_.temporal.tie_epsilon_px);
                bool joint_ok = pj.success;
                if (joint_ok && use_seed &&
                    !motionGatePass(pj, seed, nullptr, false, "DualRoi-BCJ", true)) {
                    PoseEstimate pc = mono_pnp_.solve(jp2d, jp3d, camera_.K, nullptr, 0.0);
                    if (pc.success) {
                        pj = pc;
                    } else {
                        joint_ok = false;
                    }
                }
                pnp_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_j).count();

                if (joint_ok) {
                    pose = pj;
                    joint_used = true;
                    merged_pts_2d = std::move(jp2d);
                    merged_pts_3d = std::move(jp3d);
                    merged_kp_left.clear();
                    merged_kp_left.reserve(merged_pts_2d.size());
                    for (const auto& p : merged_pts_2d)
                        merged_kp_left.emplace_back(p, 1.0f);
                    pair_tmpl_idx.clear();
                    pair_tmpl_idx.insert(pair_tmpl_idx.end(), jidx_c0.begin(), jidx_c0.end());
                    pair_tmpl_idx.insert(pair_tmpl_idx.end(), jidx_c1.begin(), jidx_c1.end());
                    n_bc_use = static_cast<int>(j2d_c0.size());
                    total_use = n_match;
                    routeStep("Tier1b(BCJ c0=" + std::to_string(n_bc_use)
                              + "+c1=" + std::to_string(static_cast<int>(j2d_c1.size()))
                              + ") OK");
                } else {
                    routeStep("Tier1b(BCJ c0=" + std::to_string(static_cast<int>(j2d_c0.size()))
                              + "+c1=" + std::to_string(static_cast<int>(j2d_c1.size()))
                              + ") FAIL[pnp]");
                }
            }
        }
    } else if (pose.success && dual_c1_tmpl_pts3d_.empty()) {
        routeStep("Tier1b(BCJ) SKIP[no c1 anno]");
    }

    } // total_use >= 4 (角点可解才执行 Tier1)
    } // !primary_spans_frame（短路或角点不足时 pose 保持失败, 8c 直接成为唯一路径）

    // 8c. 退化: Tier1 BC(class0) 失败 → 在 secondary ROI (class1) 上跑 BC→TT 链
    //     (class1 3D 尺寸, State 5 同机制; 注入板系 seed —— class1 图案与板共面无
    //      相对旋转, R seed 直接有效, 仅 t 差同心偏移, IPPE 二义性由 ε-平票偏 seed 消解)
    bool c1_fallback_used = false;
    std::string c1_strategy;
    PipelineResult c1_result;
    std::vector<Eigen::Vector3d> c1_pts3d;
    if (!kDualRoiTier1Only && !pose.success && config_.dual_roi_class1_fallback) {
        if (verbose_console_)
            std::cout << "  [DualRoi][Mono] Tier1 BC solve failed, trying class1 chain (BC→TT)"
                      << "\n";
        std::string c1_legs;
        auto [ok, c1_pose] = runDualRoiClass1Chain(left_c1_gray, left_c1_color,
                                                   left_sec, is_first,
                                                   c1_result, c1_pts3d, c1_strategy,
                                                   extract_ms, pnp_ms,
                                                   use_seed ? &seed : nullptr,
                                                   &c1_legs);
        if (ok) {
            pose = c1_pose;
            c1_result.left_color = left_color_orig;
            total_use = static_cast<int>(c1_result.pts_left_match.size());
            n_bc_use = 0;   // 可视化分色: 链胜出点全部来自 class1
            merged_pts_2d = c1_result.pts_left_match;
            merged_kp_left.clear();
            merged_pts_3d = std::move(c1_pts3d);
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

    // 9. Build PipelineResult
    PipelineResult result;
    if (c1_fallback_used) {
        result = std::move(c1_result);
        result.n_kp_left = total_use;
        result.pts_left_match = merged_pts_2d;
        result.pts_left_good = merged_pts_2d;
    } else {
        result.kp_left         = std::move(merged_kp_left);
        result.n_kp_left       = total_use;
        result.pts_left_match  = merged_pts_2d;
        result.pts_left_good   = merged_pts_2d;
    }
    result.n_template_match=(int)result.good_matches.size();
    result.left_color      = left_color_orig;
    result.is_first_frame  = is_first;

    if (!c1_fallback_used) {
      result.pose_model_points = dual_bc_tmpl_pts3d_;
      result.pose_model_points.insert(result.pose_model_points.end(),
                                      dual_c1_tmpl_pts3d_.begin(),
                                      dual_c1_tmpl_pts3d_.end());
      for (int i = 0; i < (int)pair_tmpl_idx.size(); ++i)
        result.good_matches.emplace_back(i, pair_tmpl_idx[i], 0);
      result.bc_detected_count = result_bc.bc_detected_count;
      result.bc_rejected_count = result_bc.bc_rejected_count;
      result.bc_rejected_points = result_bc.bc_rejected_points;
      result.bc_rejected_ids = result_bc.bc_rejected_ids;
      result.bc_residuals_px = result_bc.bc_residuals_px;
      for (auto &p : result.bc_rejected_points) {
        p.x += left_pri.x;
        p.y += left_pri.y;
      }
    }
    if (pose.success) {
        finalizePose(result, pose);
    }
    result.warm_start_used = use_seed && pose.success && !c1_fallback_used;
    if (t1_cold_recovered) result.gate_status = GateStatus::Recovered;
    // Tier1(BC class0) 现为主级: Tier1b 联合解算胜出为 "DualRoi_BCJ", 纯 c0 为
    // "DualRoi_BC"; class1 链胜出保留 C1 命名
    result.strategy_name = c1_fallback_used ? c1_strategy
                         : (joint_used ? "DualRoi_BCJ" : "DualRoi_BC");
    if(!pose.success)result.failure_reason="dual_roi_chain_failed";

    result.n_matched   = total_use;
    result.n_projected = 0;
    result.extract_ms = extract_ms;
    result.pnp_ms = pnp_ms;
    addLogEntry(result, is_first, c1_fallback_used);

    // ---- Visualization (simplified, left-only) ----
    // PnP 失败时仍输出：Panel 0/1 不依赖位姿；Panel 2/3 内部各自按 pose.success 降级
    if (visualize && !output_dir_.empty()) {
        std::string prefix = "_f" + std::to_string(current_frame_);

        const cv::Scalar BC_COLOR(0, 0, 255);    // red: BinaryCorner corners
        const cv::Scalar AK_COLOR(0, 255, 0);    // green: class1 链点 (Tier3)

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
            cv::putText(p0, "class1 (BC)",
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

        // Panel 2: 3D axes on original image (始终生成; PnP 失败时降级为特征点 + FAIL 标注)
        {
            cv::Mat p2 = left_color_orig.clone();
            if (pose.success) {
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
            } else {
                // 无有效位姿不画轴（单位阵 pose 画轴无意义），叠加合并特征点便于排查
                for (int i = 0; i < total_use; ++i) {
                    cv::Point pt(static_cast<int>(merged_pts_2d[i].x),
                                 static_cast<int>(merged_pts_2d[i].y));
                    cv::circle(p2, pt, 3, (i < n_bc_use) ? BC_COLOR : AK_COLOR, -1);
                }
                std::ostringstream oss;
                oss << "PnP FAIL | n=" << total_use << " (BC=" << n_bc_use
                    << ", AK=" << (total_use - n_bc_use) << ")";
                cv::putText(p2, oss.str(), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_axes" + prefix + ".png", p2);
        }

        // ---- 诊断面板 (reproj) 仅 Debug 模式 ----
        if (visualize_detailed_) {

        // Panel 3: Reprojection error on class 0 ROI zoomed
        // (PnP 失败时退化为仅观测点，与主路径 BC/TT reproj 的 obs-only 分支一致)
        if (total_use > 0 && !merged_pts_3d.empty()) {
            cv::Mat p3 = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();
            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            for (int i = 0; i < total_use && i < static_cast<int>(merged_pts_3d.size()); ++i) {
                cv::Point po(static_cast<int>(merged_pts_2d[i].x - lx),
                             static_cast<int>(merged_pts_2d[i].y - ly));
                cv::Scalar obs_color = (i < n_bc_use) ? BC_COLOR : AK_COLOR;
                if (pose.success) {
                    Eigen::Vector3d P_cam = pose.R * merged_pts_3d[i] + pose.t;
                    double fx = camera_.K(0, 0), fy = camera_.K(1, 1);
                    double cx = camera_.K(0, 2), cy = camera_.K(1, 2);
                    if (std::abs(P_cam.z()) < 1e-6) continue;
                    double u = fx * P_cam.x() / P_cam.z() + cx;
                    double v = fy * P_cam.y() / P_cam.z() + cy;
                    cv::Point pd(static_cast<int>(u - lx), static_cast<int>(v - ly));
                    cv::circle(p3, pd, 2, cv::Scalar(0, 255, 0), -1);
                    cv::circle(p3, po, 4, obs_color, 1);
                    cv::line(p3, pd, po, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                } else {
                    cv::circle(p3, po, 4, obs_color, -1);
                }
            }
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_reproj" + prefix + ".png", p3);
        }

        // Panel 4: Image ↔ Template correspondence (side-by-side, 对齐 StereoTracker 的 dual_roi_correspondence)
        // class1 链胜出时跳过: 模板侧对应关系为板模板 BC 角点, 与链点 (BC/TT class1) 不匹配
        if (!c1_fallback_used && total_use > 0) {
            // Left: class-0-ROI image
            cv::Mat p4_left = left_color_orig(
                cv::Rect(left_pri.x, left_pri.y, left_pri.width, left_pri.height)).clone();

            // Right: AKAZE template (grayscale → BGR, resize to match left height)
            const cv::Mat& tmpl_gray = template_.gray_image;
            cv::Mat tmpl_color;
            if (!tmpl_gray.empty()) {
                cv::cvtColor(tmpl_gray, tmpl_color, cv::COLOR_GRAY2BGR);
            } else {
                tmpl_color = cv::Mat(p4_left.rows, p4_left.rows, CV_8UC3,
                                     cv::Scalar(128, 128, 128));
            }
            double scale_tmpl = static_cast<double>(p4_left.rows) / tmpl_color.rows;
            cv::Mat tmpl_resized;
            cv::resize(tmpl_color, tmpl_resized, cv::Size(), scale_tmpl, scale_tmpl,
                       cv::INTER_NEAREST);

            cv::Mat p4;
            cv::hconcat(p4_left, tmpl_resized, p4);

            float lx = static_cast<float>(left_off.x), ly = static_cast<float>(left_off.y);
            int offset_x = p4_left.cols;  // 图像与模板的水平分界

            // 模板侧点: pair_tmpl_idx[i] < n0c → class0 标注角点(红),
            // 否则 → class1 标注角点(绿, 索引偏移 n0c)
            const int n0c = static_cast<int>(dual_bc_tmpl_corners_.size());
            const int n1c = static_cast<int>(dual_c1_tmpl_corners_.size());
            for (int i = 0; i < total_use &&
                           i < static_cast<int>(pair_tmpl_idx.size()); ++i) {
                // 图像点 (class-0-ROI 局部坐标)
                cv::Point pt_img(static_cast<int>(merged_pts_2d[i].x - lx),
                                 static_cast<int>(merged_pts_2d[i].y - ly));

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

                cv::circle(p4, pt_img,  2, *color, -1);
                cv::circle(p4, pt_tmpl, 2, *color, -1);
                cv::line(p4, pt_img, pt_tmpl, *color, 1, cv::LINE_AA);
            }
            utils::AsyncImageSaver::write(output_dir_ + "/dual_roi_mono_correspondence" + prefix + ".png", p4);
        }

        // Panel 5: BC 二值化过程条图 — class0 裁剪图在 BinaryCorner 内部
        // 逐阶段的二值化结果 (otsu → largest → filled → smoothed → 旋转链),
        // 排查 keepLargestRegion/keepRegionFromCenter 选域问题用。
        // 注: Tier3 class1 链胜出时快照对应的是最后一次 BC 调用 (可能是 class1 裁剪)。
        saveBcProcessPanel();

        } // end visualize_detailed_ dual-ROI panels

        if (verbose_console_)
            std::cout << "  [DualRoi][Mono] Visualized: " << n_bc_use << " BC + "
                      << (total_use - n_bc_use) << " C1 pts" << "\n";
    }

    result.timing["dual_roi"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_dual_start).count();

    routeStep(pose.success ? "FRAME-OK" : "FRAME-FAIL");
    if (verbose_console_)
        std::cout << "[DualRoi][Mono] Route: " << route << "\n";

    if (verbose_console_)
        std::cout << "[DualRoi][Mono] Frame done: n_pts=" << total_use
                  << "  PnP=" << (pose.success ? "OK" : "FAIL")
                  << (c1_fallback_used ? "  [class1-chain:" + c1_strategy + "]" : "")
                  << "\n";

    return result;
}

// ============================================================================
// Dual-ROI Tier 3: class1 退化链 —— secondary ROI 上 BC → TT
// (State 5 同款 class1 3D 尺寸机制; 提取失败或 PnP 失败均降级, 与主链同语义)
// ============================================================================

std::pair<bool, PoseEstimate> MonoTracker::runDualRoiClass1Chain(
        const cv::Mat& left_gray, const cv::Mat& left_color,
        const RoiRect& left_sec,
        bool is_first,
        PipelineResult& out_result,
        std::vector<Eigen::Vector3d>& out_pts3d,
        std::string& out_strategy,
        double& extract_ms_acc, double& pnp_ms_acc,
        const PoseSeed* seed,
        std::string* out_route_detail) {
    out_strategy.clear();
    out_pts3d.clear();
    if (out_route_detail) out_route_detail->clear();
    if (left_gray.empty()) return {false, PoseEstimate{}};

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
            std::cout << "[DualRoi][Mono] class1 chain: " << ext->name()
                      << " on secondary ROI" << "\n";

        auto t_ext = std::chrono::steady_clock::now();
        PipelineResult local = ext->extractMono(left_gray, left_color);
        extract_ms_acc += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_ext).count();
        if (!(local.success && local.n_kp_left >= 3)) {
            if (verbose_console_)
                std::cout << "  [DualRoi][Mono] class1 " << ext->name()
                          << " extraction failed" << "\n";
            legFail(is_bc ? "BC" : "TT", "extract");
            continue;
        }

        // ROI 局部坐标 → 全图坐标 (PnP 用全图内参)
        const float sox = static_cast<float>(left_sec.x);
        const float soy = static_cast<float>(left_sec.y);
        for (auto& p : local.pts_left_match) { p.x += sox; p.y += soy; }
        for (auto& p : local.pts_left_good)  { p.x += sox; p.y += soy; }
        for (auto& kp : local.kp_left) { kp.pt.x += sox; kp.pt.y += soy; }

        // 2D/3D 配对: good_matches[i].trainIdx → class1 3D 模板 (与 process 主链同规则)
        const auto& pts3d = ext->templateData().pts_3d;
        std::vector<cv::Point2f> pnp_2d;
        std::vector<Eigen::Vector3d> pnp_3d;
        pnp_2d.reserve(std::min(local.good_matches.size(), local.pts_left_match.size()));
        for (size_t i = 0; i < local.good_matches.size() &&
                           i < local.pts_left_match.size(); ++i) {
            int idx = local.good_matches[i].trainIdx;
            if (idx >= 0 && idx < static_cast<int>(pts3d.size())) {
                pnp_2d.push_back(local.pts_left_match[i]);
                pnp_3d.push_back(pts3d[idx]);
            }
        }
        if (pnp_2d.size() < 4) {
            if (verbose_console_)
                std::cout << "  [DualRoi][Mono] class1 " << ext->name()
                          << " matched pairs < 4 (" << pnp_2d.size() << ")" << "\n";
            legFail(is_bc ? "BC" : "TT", "pts<4");
            continue;
        }

        // MonoPnP 多候选择优; class1 与板共面无相对旋转, 板系 R seed 直接有效
        // (ε-平票偏 seed 消解 IPPE 二义性), 仅 t 差同心偏移
        auto t_c1_pnp = std::chrono::steady_clock::now();
        PoseEstimate p = seed
            ? mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K, seed,
                              config_.temporal.tie_epsilon_px)
            : mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K);
        pnp_ms_acc += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_c1_pnp).count();
        if (!p.success) {
            if (verbose_console_)
                std::cout << "  [DualRoi][Mono] class1 " << ext->name()
                          << " PnP failed" << "\n";
            legFail(is_bc ? "BC" : "TT", "pnp");
            continue;
        }

        // 截断为 2D/3D 严格 1:1 的配对子集 (BC 近似角点多于模板时剔除尾部)
        const int n_used = static_cast<int>(pnp_2d.size());
        local.n_kp_left = n_used;
        local.pts_left_match = pnp_2d;
        if (static_cast<int>(local.pts_left_good.size()) > n_used)
            local.pts_left_good.resize(n_used);
        if (static_cast<int>(local.kp_left.size()) > n_used)
            local.kp_left.resize(n_used);
        local.good_matches.resize(n_used);
        local.n_template_match=n_used;
        for (int i = 0; i < n_used; ++i)
          local.good_matches[i].queryIdx = i;
        local.pose_model_points = pts3d;
        local.is_first_frame = is_first;
        local.is_class1 = true;

        out_result = std::move(local);
        out_pts3d = std::move(pnp_3d);
        out_strategy = is_bc ? "DualRoi_C1BC" : "DualRoi_C1TT";
        if (out_route_detail) *out_route_detail = is_bc ? "BC" : "TT";
        return {true, p};
    }
    return {false, PoseEstimate{}};
}

PipelineResult MonoTracker::process(const cv::Mat& left_img,
                                           bool visualize,
                                           const RoiGroup* left_group) {
    PipelineResult result;
    result.success = false;

    binary_extractor_->setDebugCapture(visualize);
    tiny_extractor_->setDebugCapture(visualize);

    if (left_img.empty()) {
        std::cerr << "[Mono] empty left image" << "\n";
        return result;
    }

    // ---- Dual-ROI dispatch ----
    if (left_group && left_group->is_dual) {
        result = processDualRoi(left_img, *left_group, visualize);
        state_.frame_count++;
        return result;
    }

    // ROI 校验：无 ROI → 帧失败（全图回退随 AKAZE 链一并移除）
    const RoiRect* left_roi = left_group ? &left_group->primary : nullptr;
    RoiRect roi = validateRoi(left_roi, left_img.size(), "mono_left");
    if (!roi.valid()) {
        if (verbose_console_)
            std::cout << "[Mono] no valid ROI, frame fails (Skip_NoRoi)" << "\n";
        result.strategy_name = "Skip_NoRoi";
        addLogEntry(result, state_.frame_count == 0, false);
        state_.frame_count++;
        return result;
    }

    int roi_area = roi.width * roi.height;
    if (verbose_console_)
        std::cout << "[Mono] ROI area=" << roi_area
                  << " (" << roi.width << "x" << roi.height << ")"
                  << "\n";

    // P3: 灰度仅转 ROI（全图回退时等价旧全图转换）；彩色引用共享不复制
    // （提取器不消费彩色像素，仅可视化面板透传；画图前均已 clone）
    if (left_img.empty()) {
        std::cerr << "[Mono] failed to load left image" << "\n";
        return result;
    }
    cv::Rect left_rc(roi.x, roi.y, roi.width, roi.height);
    cv::Mat left_gray_roi;
    cv::cvtColor(left_img(left_rc), left_gray_roi, cv::COLOR_BGR2GRAY);
    cv::Mat left_color     = left_img;
    cv::Mat left_color_roi = left_img(left_rc);
    cv::Point2d left_offset(roi.x, roi.y);
    result.left_color = left_color;

    // 根据输入类别选择 BC/TT 的 3D 模板尺寸 并标记用于策略链选择
    // is_class1=true：仅检测到 class1（近距离回退），使用 class1 尺寸 + class1 阈值
    bool use_c1 = (left_group != nullptr && left_group->is_class1);
    {
        binary_extractor_->setUseClass1(use_c1);
        tiny_extractor_->setUseClass1(use_c1);
    }

    // 策略链选择（class1 时使用 class1 专用面积阈值）
    configureStrategyChain(roi_area, use_c1);

    // 时序 seed（机制A: 位姿链）—— 上帧成功位姿作为本帧 PnP 初值候选
    PoseSeed seed;
    const bool use_seed = seedActive();
    if (use_seed) { seed.R = state_.R_prev; seed.t = state_.t_prev; }

    bool is_first = (state_.frame_count == 0);

    // 尝试主策略 + 退化链：提取成功且单目 PnP 成功才算胜出，
    // PnP 失败（如 BC 近似角点数与模板不一致被 MonoPnP 拒解）同样触发退化
    std::vector<FeatureExtractor*> chain;
    chain.push_back(extractor_);
    for (auto* fb : fallback_extractors_)
        chain.push_back(fb);

    FeatureExtractor* winning_ext = nullptr;
    PoseEstimate pose;
    double extract_ms = 0.0, pnp_ms = 0.0;   // 阶段耗时: 特征提取 / 位姿解算 (跨退化链累加)

    for (auto* ext : chain) {
        if (!ext) continue;

        if (verbose_console_) std::cout << "[Mono] Trying extractor: " << ext->name() << "\n";

        // 单目提取（仅左图，2 参数）
        auto t_extract = std::chrono::steady_clock::now();
        PipelineResult local;
        if (ext == binary_extractor_.get() && has_prepared_bc_) {
          local = std::move(prepared_bc_);
          has_prepared_bc_ = false;
        } else if (ext == tiny_extractor_.get() && has_prepared_tt_) {
          local = std::move(prepared_tt_);
          has_prepared_tt_ = false;
        } else
          local = ext->extractMono(left_gray_roi, left_color_roi);
        if(visualize && !output_dir_.empty() && ext->strategyType()==StrategyType::BinaryCorner)
            saveMonoIdentityPanel(*binary_extractor_,local,output_dir_+"/mono_identity_f"+std::to_string(current_frame_)+".png");
        extract_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_extract).count();

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
        for (auto &p : local.bc_rejected_points) {
          p.x += left_offset.x;
          p.y += left_offset.y;
        }

        if (!(local.success && local.n_kp_left >= 3)) {
            if(result.failure_reason.empty())result.failure_reason=local.failure_reason.empty()?"feature_extraction_failed":local.failure_reason;
            if (verbose_console_) std::cout << "[Mono] Extractor " << ext->name() << " failed, degrading..." << "\n";
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
        // （纯重投影择优，成功即采纳），冷重解也失败则本策略判失败继续退化链
        PoseEstimate p;
        GateStatus gate_st = GateStatus::NotApplicable;
        auto t_pnp = std::chrono::steady_clock::now();
        if (use_seed) {
          p = mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K, &seed,
                              config_.temporal.tie_epsilon_px,
                              local.pose_min_inliers);
          if (p.success) {
            if (motionGatePass(p, seed, ext, false)) {
              gate_st = GateStatus::Pass;
            } else {
              PoseEstimate pc =
                  mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K, nullptr, 0.0,
                                  local.pose_min_inliers);
              if (pc.success) {
                // 冷重解 = temporal-OFF 的可信路径，直接采纳（不再被陈旧 seed
                // 锚定的门控二次审查）
                p = pc;
                gate_st = GateStatus::Recovered;
                if (verbose_console_)
                  std::cout
                      << "[Gate] warm pose rejected, cold re-solve accepted"
                      << "\n";
              } else {
                p = PoseEstimate{};
                gate_st = GateStatus::Rejected;
                if (verbose_console_)
                  std::cout << "[Gate] pose rejected by motion gate ("
                            << ext->name() << "), degrading..."
                            << "\n";
              }
            }
            }
        } else {
          p = mono_pnp_.solve(pnp_2d, pnp_3d, camera_.K, nullptr, 0,
                              local.pose_min_inliers);
        }
        pnp_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pnp).count();

        if(!p.success)local.failure_reason=gate_st==GateStatus::Rejected?"motion_gate_rejected":mono_pnp_.lastRejectionReason();
        result = std::move(local);
        result.left_color = left_color;
        result.left_roi_offset_x = static_cast<int>(left_offset.x);
        result.left_roi_offset_y = static_cast<int>(left_offset.y);
        result.warm_start_used = use_seed;
        result.gate_status = gate_st;
        fillTemporalMeta(result);
        winning_ext = ext;

        if (p.success) {
            pose = p;
            if (verbose_console_)
                std::cout << "[Mono] Extractor " << ext->name()
                          << " succeeded, n_kp=" << result.n_kp_left << "\n";
            break;
        }

        if (verbose_console_)
            std::cout << "[Mono] Extractor " << ext->name()
                      << " PnP failed, degrading..." << "\n";
    }

    if (!winning_ext) {
        std::cerr << "[Mono] All extractors failed" << "\n";
        result.is_class1 = use_c1;
        result.extract_ms = extract_ms;
        result.pnp_ms = pnp_ms;
        addLogEntry(result, is_first, true);
        return result;
    }

    result.pose_model_points = winning_ext->templateData().pts_3d;
    finalizePose(result, pose);

    result.strategy_name = winning_ext->name();
    result.success = pose.success;
    result.is_class1 = use_c1;
    result.extract_ms = extract_ms;
    result.pnp_ms = pnp_ms;
    updateStickinessFromWinner(winning_ext->name());
    fillTemporalMeta(result);   // 胜者反馈后重取，switch_reason 反映本帧 realign
    addLogEntry(result, is_first, winning_ext != chain.front());

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
            } else {
                cv::putText(vis, "PnP FAIL", cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
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
            // Draw matched points on axes panel (BC/TT 用小半径避免遮挡角点)
            const int pt_radius = (is_bc || is_tiny) ? 1 : 4;
            for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                cv::Point2f pv = toView_L(result.pts_left_match[i]);
                cv::circle(
                    p_axes,
                    cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                    pt_radius,
                    CORNER_COLORS[std::max(0, physicalCornerId(result, i)) %
                                  10],
                    -1);
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
                    cv::circle(
                        bl_bgr, p, 4,
                        CORNER_COLORS[std::max(0, physicalCornerId(result, i)) %
                                      10],
                        -1);
                }
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_binary" + prefix + ".png", bl_bgr);
            }

            // -- Panel: Upright (rotated-back) binary --
            if (bce && !bce->lastUprightBinary().empty()) {
                cv::Mat up;
                cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_upright" + prefix + ".png", up);
            }

            // -- Panel: Largest region (第1步连通域筛选结果) --
            if (bce && !bce->lastLargestRegion().empty()) {
                cv::Mat lr;
                cv::cvtColor(bce->lastLargestRegion(), lr, cv::COLOR_GRAY2BGR);
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_largest_region" + prefix + ".png", lr);
            }

            // -- Panel: Contour input (第5步角点提取的输入：二值图 + 最大轮廓) --
            if (bce && !bce->lastContourBinary().empty()) {
                cv::Mat ci;
                cv::cvtColor(bce->lastContourBinary(), ci, cv::COLOR_GRAY2BGR);
                const auto& ctr = bce->lastContour();
                if (!ctr.empty()) {
                    std::vector<std::vector<cv::Point>> polys = {ctr};
                    cv::polylines(ci, polys, true, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }
                utils::AsyncImageSaver::write(output_dir_ + "/mono_bc_contour_input" + prefix + ".png", ci);
            }

            // -- Panel: Template correspondence (left-view | matched-template) --
            {
                cv::Mat p_tmpl_l = view_L.clone();
                for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                    cv::Point2f pv = toView_L(result.pts_left_match[i]);
                    cv::circle(
                        p_tmpl_l,
                        cv::Point(static_cast<int>(pv.x),
                                  static_cast<int>(pv.y)),
                        4,
                        CORNER_COLORS[std::max(0, physicalCornerId(result, i)) %
                                      10],
                        -1);
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
                    for (size_t i = 0; i < matched_tmpl->corners.size(); ++i) {
                        const auto& c = matched_tmpl->corners[i];
                        cv::circle(p_tmpl_r,
                                   cv::Point(static_cast<int>(c.x * dsx),
                                             static_cast<int>(c.y * dsy)),
                                   4,
                                   CORNER_COLORS[std::max(0, physicalCornerId(
                                                                 result, i)) %
                                                 10],
                                   1);
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
                          const int model_id = physicalCornerId(result, i);
                          if (model_id < 0 ||
                              model_id >= (int)pnp_pts_3d.size())
                            continue;
                          Eigen::Vector3d P_cam =
                              result.R * pnp_pts_3d[model_id] + result.t;
                          if (std::abs(P_cam.z()) >= 1e-6) {
                            Eigen::Vector2d uv =
                                projectToImage(P_cam, camera_.K);
                            cv::Point pd(static_cast<int>(uv.x() - expand_L.x),
                                         static_cast<int>(uv.y() - expand_L.y));
                            cv::circle(p_reproj, pd, 3, cv::Scalar(0, 255, 0),
                                       -1);
                            cv::circle(
                                p_reproj, po, 5,
                                CORNER_COLORS[std::max(0, physicalCornerId(
                                                              result, i)) %
                                              10],
                                1);
                            cv::line(p_reproj, pd, po, cv::Scalar(0, 255, 255),
                                     1, cv::LINE_AA);
                            }
                        } else {
                          cv::circle(p_reproj, po, 4,
                                     CORNER_COLORS[std::max(0, physicalCornerId(
                                                                   result, i)) %
                                                   10],
                                     -1);
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

            // -- Panel: ROI 与二值化图 (灰度 ROI | 清理前 Otsu | 清理后二值) --
            {
                auto* tte = tiny_extractor_.get();
                const auto& dbg = tte->lastLeftDebug();
                if (!dbg.roi_gray.empty()) {
                    float lx_roi = static_cast<float>(left_offset.x);
                    float ly_roi = static_cast<float>(left_offset.y);
                    auto drawPts = [&](cv::Mat& img, float k) {
                        for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                            cv::Point p(
                                static_cast<int>((result.pts_left_match[i].x - lx_roi) * k),
                                static_cast<int>((result.pts_left_match[i].y - ly_roi) * k));
                            cv::circle(
                                img, p, 2,
                                CORNER_COLORS[std::max(0, physicalCornerId(
                                                              result, i)) %
                                              10],
                                -1);
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
                    cv::Mat panel = tiles[0];
                    for (size_t i = 1; i < tiles.size(); ++i)
                        cv::hconcat(panel, tiles[i], panel);
                    utils::AsyncImageSaver::write(
                        output_dir_ + "/mono_tt_roi_binary" + prefix + ".png", panel);
                }
            }

            // -- Panel: Reprojection error --
            if (!pnp_pts_3d.empty()) {
                cv::Mat p_reproj = view_L.clone();
                size_t n_pts = std::min(result.pts_left_match.size(), pnp_pts_3d.size());
                for (size_t i = 0; i < n_pts; ++i) {
                    cv::Point2f obs_v = toView_L(result.pts_left_match[i]);
                    cv::Point po(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    if (result.success) {
                      const int model_id = physicalCornerId(result, i);
                      if (model_id < 0 || model_id >= (int)pnp_pts_3d.size())
                        continue;
                      Eigen::Vector3d P_cam =
                          result.R * pnp_pts_3d[model_id] + result.t;
                      if (std::abs(P_cam.z()) >= 1e-6) {
                        Eigen::Vector2d uv = projectToImage(P_cam, camera_.K);
                        cv::Point pd(static_cast<int>(uv.x() - expand_L.x),
                                     static_cast<int>(uv.y() - expand_L.y));
                        cv::circle(p_reproj, pd, 3, cv::Scalar(0, 255, 0), -1);
                        cv::circle(p_reproj, po, 5,
                                   CORNER_COLORS[std::max(0, physicalCornerId(
                                                                 result, i)) %
                                                 10],
                                   1);
                        cv::line(p_reproj, pd, po, cv::Scalar(0, 255, 255), 1,
                                 cv::LINE_AA);
                        }
                    } else {
                      cv::circle(p_reproj, po, 4,
                                 CORNER_COLORS[std::max(0, physicalCornerId(
                                                               result, i)) %
                                               10],
                                 -1);
                    }
                }
                utils::AsyncImageSaver::write(output_dir_ + "/mono_tt_reproj" + prefix + ".png", p_reproj);
            }
        }

        } // end visualize_detailed_ per-strategy panels

        if (verbose_console_)
            std::cout << "[Mono] Visualization saved: " << output_dir_
                      << " (strategy=" << strategy << ")" << "\n";
    }

    state_.frame_count++;
    return result;
}

} // namespace gpnp
