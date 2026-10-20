/**
 * @file RoiGenerator.cpp
 * @brief YOLO 检测 → RoiRect 转换的实现。
 */

#include "detection/RoiGenerator.hpp"
#include "common/LogConfig.hpp"
#include "common/TinyTargetPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace gpnp {

RoiSelectionConfig readRoiSelectionConfig(const cv::FileNode &node) {
  RoiSelectionConfig c;
  if (node.empty())
    return c;
  if (!node.isMap())
    throw std::invalid_argument("roi_selection must be an object");
  c.tt_verification = readTinyTargetVerificationConfig(node["tt_verification"]);
  for (auto item : {std::pair<const char *, bool *>{"enabled", &c.enabled},
                    {"verify_mono", &c.verify_mono}}) {
    auto v = node[item.first];
    if (v.empty())
      continue;
    if (!v.isInt() || ((int)v != 0 && (int)v != 1))
      throw std::invalid_argument("roi_selection boolean expected");
    *item.second = (int)v != 0;
  }
  for (auto item :
       {std::pair<const char *, int *>{"max_candidates", &c.max_candidates},
        {"history_max_age", &c.history_max_age}}) {
    auto v = node[item.first];
    if (v.empty())
      continue;
    if (!v.isInt())
      throw std::invalid_argument("roi_selection integer expected");
    *item.second = (int)v;
  }
  for (auto item : {std::pair<const char *, double *>{"containment_min",
                                                      &c.containment_min},
                    {"containment_margin_px", &c.containment_margin_px},
                    {"min_component_ratio", &c.min_component_ratio},
                    {"min_observed_span_px", &c.min_observed_span_px}}) {
    auto v = node[item.first];
    if (v.empty())
      continue;
    if (!v.isInt() && !v.isReal())
      throw std::invalid_argument("roi_selection numeric expected");
    *item.second = (double)v;
    if (!std::isfinite(*item.second))
      throw std::invalid_argument("roi_selection finite threshold expected");
  }
  if (c.max_candidates < 1 || c.max_candidates > 8 || c.history_max_age < 0 ||
      c.history_max_age > 1000 || c.containment_min <= 0 ||
      c.containment_min > 1 || c.containment_margin_px < 0 ||
      c.containment_margin_px > 50 || c.min_component_ratio <= 0 ||
      c.min_component_ratio > 1 || c.min_observed_span_px < 2 ||
      c.min_observed_span_px > 32)
    throw std::invalid_argument("roi_selection threshold out of range");
  return c;
}

std::vector<RoiCandidate>
RoiGenerator::candidates(const std::vector<Detection> &detections,
                         const cv::Size &image_size) const {
  if (!config_.selection.enabled) {
    auto group = generateGroup(detections, image_size);
    return group.valid() ? std::vector<RoiCandidate>{{group, -1, -1, 0, 0}}
                         : std::vector<RoiCandidate>{};
  }
  const cv::Rect2f image(0, 0, image_size.width, image_size.height);
  std::vector<int> classes[2];
  std::vector<cv::Rect2f> boxes(detections.size());
  for (int i = 0; i < (int)detections.size(); ++i) {
    const auto &d = detections[i];
    const auto &b = d.bbox;
    if (d.class_id < 0 || d.class_id > 1 || !std::isfinite(d.confidence) ||
        d.confidence < 0 || d.confidence > 1 || !std::isfinite(b.x) ||
        !std::isfinite(b.y) || !std::isfinite(b.width) ||
        !std::isfinite(b.height) || b.width <= 0 || b.height <= 0)
      continue;
    boxes[i] = b & image;
    if (boxes[i].area() > 0)
      classes[d.class_id].push_back(i);
  }
  for (auto &list : classes) {
    std::stable_sort(list.begin(), list.end(), [&](int a, int b) {
      return detections[a].confidence > detections[b].confidence;
    });
    std::vector<int> unique;
    for (int id : list) {
      bool duplicate = false;
      for (int kept : unique) {
        double intersection = (boxes[id] & boxes[kept]).area();
        if (intersection /
                (boxes[id].area() + boxes[kept].area() - intersection) >
            .7) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate)
        unique.push_back(id);
      if ((int)unique.size() >= config_.selection.max_candidates)
        break;
    }
    list = std::move(unique);
  }
  std::vector<RoiCandidate> out;
  for (int i : classes[0]) {
    int inner = -1;
    double best = -1, coverage = 0;
    auto outer = boxes[i];
    const float margin = (float)config_.selection.containment_margin_px;
    outer.x -= margin;
    outer.y -= margin;
    outer.width += 2 * margin;
    outer.height += 2 * margin;
    for (int j : classes[1]) {
      double fraction = (outer & boxes[j]).area() / boxes[j].area();
      if (boxes[j].area() >= boxes[i].area() ||
          fraction < config_.selection.containment_min)
        continue;
      double score = detections[j].confidence * fraction;
      if (score > best) {
        best = score;
        inner = j;
        coverage = fraction;
      }
    }
    RoiGroup group{detectionToRoi(detections[i], image_size,
                                  config_.roi_expand_ratio,
                                  config_.roi_min_size),
                   {},
                   false};
    if (inner >= 0 && group.primary.width * group.primary.height >
                          config_.dual_trigger_area) {
      group.secondary =
          detectionToRoi(detections[inner], image_size,
                         config_.roi_expand_ratio, config_.roi_min_size);
      group.is_dual = group.secondary.valid();
    }
    if (group.valid())
      out.push_back({group, i, inner,
                     detections[i].confidence + (inner >= 0 ? .15 * best : 0),
                     coverage});
  }
  if (close_range_cfg_.enabled)
    for (int j : classes[1]) {
      auto group = tryCloseRange({detections[j]}, image_size);
      if (group.valid())
        out.push_back({group, -1, j, detections[j].confidence - .25, 0});
    }
  std::stable_sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    return a.score > b.score;
  });
  if ((int)out.size() > config_.selection.max_candidates)
    out.resize(config_.selection.max_candidates);
  return out;
}

// ============================================================================
// 构造
// ============================================================================

RoiGenerator::RoiGenerator()
    : config_{}
{}

RoiGenerator::RoiGenerator(const Config& cfg)
    : config_(cfg)
{}

// ============================================================================
// 单一 ROI 生成
// ============================================================================

RoiRect RoiGenerator::generate(const std::vector<Detection>& detections,
                                const cv::Size& image_size,
                                int class_id) const {
    if (detections.empty()) {
        return RoiRect{};
    }

    // 解析有效的类别 ID：显式参数优先于配置
    int effective_class_id = (class_id >= 0) ? class_id : config_.target_class_id;

    // 查找匹配目标类别且置信度最高的检测结果
    const Detection* best = nullptr;
    float best_conf = -1.0f;

    for (const auto& det : detections) {
        if (det.class_id == effective_class_id || effective_class_id < 0) {
            if (det.confidence > best_conf) {
                best_conf = det.confidence;
                best = &det;
            }
        }
    }

    if (best == nullptr) {
        return RoiRect{};
    }

    return detectionToRoi(*best, image_size,
                          config_.roi_expand_ratio, config_.roi_min_size);
}

// ============================================================================
// 双类别 ROI 生成
// ============================================================================

RoiGroup RoiGenerator::generateGroup(const std::vector<Detection>& detections,
                                     const cv::Size& image_size) const {
  if (config_.selection.enabled) {
    auto choices = candidates(detections, image_size);
    return choices.empty() ? RoiGroup{} : choices.front().group;
  }
    // 1. 始终先尝试 class 0
    RoiRect class0_roi = generate(detections, image_size, 0);
    if (!class0_roi.valid()) {
        // 近距离回退：class0 未检测到，尝试 class1
        RoiGroup fallback = tryCloseRange(detections, image_size);
        if (fallback.valid()) return fallback;
        return RoiGroup{};
    }

    // 2. 检测 class 0 面积是否超过阈值 → 同时提取 class 1
    int area = class0_roi.width * class0_roi.height;
    if (area > config_.dual_trigger_area) {
        RoiRect class1_roi = generate(detections, image_size, 1);
        if (class1_roi.valid()) {
            std::cout << "[RoiGenerator] Dual-ROI mode triggered"
                      << " (class0 area=" << area << " > " << config_.dual_trigger_area << ")"
                      << "  primary=" << class0_roi.width << "x" << class0_roi.height
                      << "  secondary=" << class1_roi.width << "x" << class1_roi.height
                      << std::endl;
            return RoiGroup{class0_roi, class1_roi, true};
        }
    }

    return RoiGroup{class0_roi, {}, false};
}

// ============================================================================
// 近距离回退：class0 丢失，class1 有效 → 外扩 class1 ROI
// ============================================================================

RoiGroup RoiGenerator::tryCloseRange(const std::vector<Detection>& detections,
                                      const cv::Size& image_size) const {
    if (!close_range_cfg_.enabled) return RoiGroup{};

    RoiRect class1_roi = generate(detections, image_size, 1);
    if (!class1_roi.valid()) return RoiGroup{};

    int area = class1_roi.width * class1_roi.height;
    if (area < close_range_cfg_.class1_min_area) return RoiGroup{};

    // 外扩 class1 ROI 以模拟丢失的 class0 全目标范围
    int expand_w = std::max(static_cast<int>(class1_roi.width  * (close_range_cfg_.roi_expand_ratio - 1.0f) / 2),
                            close_range_cfg_.min_expand_pixels / 2);
    int expand_h = std::max(static_cast<int>(class1_roi.height * (close_range_cfg_.roi_expand_ratio - 1.0f) / 2),
                            close_range_cfg_.min_expand_pixels / 2);

    RoiRect expanded;
    expanded.x      = std::max(0, class1_roi.x - expand_w);
    expanded.y      = std::max(0, class1_roi.y - expand_h);
    expanded.width  = std::min(image_size.width  - expanded.x, class1_roi.width  + 2 * expand_w);
    expanded.height = std::min(image_size.height - expanded.y, class1_roi.height + 2 * expand_h);

    if (g_verbose_console) {
        int expanded_area = expanded.width * expanded.height;
        int tt_th = close_range_cfg_.tiny_max_area  > 0 ? close_range_cfg_.tiny_max_area  : 800;
        std::cout << "[RoiGenerator] Close-range fallback: class0 not detected,"
                  << " class1=" << class1_roi.width << "x" << class1_roi.height
                  << " expanded=" << expanded.width << "x" << expanded.height
                  << " (area=" << expanded_area;
        if (expanded_area > tt_th)          std::cout << " → BC chain)";
        else                                std::cout << " → TT)";
        std::cout << std::endl;
    }

    return RoiGroup{expanded, {}, false, true};
}

// ============================================================================
// 立体对归一化（共享辅助函数）
// ============================================================================

void RoiGenerator::normalizeStereoPair(RoiRect& left, RoiRect& right,
                                       const Detection* right_det,
                                       const cv::Size& left_img_size,
                                       const cv::Size& right_img_size) {
    // 1. 如果尺寸不同，以右图检测中心为锚点将右ROI尺寸调整为与左ROI一致
    if (left.width != right.width || left.height != right.height) {
        if (right_det != nullptr) {
            float rx_center = right_det->bbox.x + right_det->bbox.width  / 2.0f;
            float ry_center = right_det->bbox.y + right_det->bbox.height / 2.0f;
            right.x      = static_cast<int>(std::round(rx_center - left.width  / 2.0f));
            right.y      = static_cast<int>(std::round(ry_center - left.height / 2.0f));
            right.width  = left.width;
            right.height = left.height;
        }
    }

    // 2. 限制到右图边界内
    right.x = std::max(0, right.x);
    right.y = std::max(0, right.y);
    if (right.x + right.width > right_img_size.width) {
        right.width = right_img_size.width - right.x;
    }
    if (right.y + right.height > right_img_size.height) {
        right.height = right_img_size.height - right.y;
    }

    // 3. 同步尺寸：光流追踪要求相同尺寸
    int w = std::min(left.width, right.width);
    int h = std::min(left.height, right.height);
    left.width  = w;
    left.height = h;
    right.width = w;
    right.height = h;
}

// ============================================================================
// 立体 ROI 对生成
// ============================================================================

std::pair<RoiRect, RoiRect> RoiGenerator::generateStereo(
    const std::vector<Detection>& detections_left,
    const std::vector<Detection>& detections_right,
    const cv::Size& left_img_size,
    const cv::Size& right_img_size) const {

    // 1. 从左图检测结果生成左 ROI
    RoiRect left_roi = generate(detections_left, left_img_size, config_.target_class_id);
    if (!left_roi.valid()) {
        return {};
    }

    // 2. 从右图检测结果生成右 ROI
    RoiRect right_roi = generate(detections_right, right_img_size, config_.target_class_id);
    if (!right_roi.valid()) {
        // 右图无检测 → 复制左 ROI 作为回退（立体对要求左右 ROI 同时存在）
        right_roi = left_roi;
    }

    // 3. 查找右图检测结果用于中心锚定（normalizeStereoPair 需要）
    const Detection* best_right = nullptr;
    {
        int effective_class = config_.target_class_id;
        float best_conf = -1.0f;
        for (const auto& det : detections_right) {
            if (det.class_id == effective_class || effective_class < 0) {
                if (det.confidence > best_conf) {
                    best_conf = det.confidence;
                    best_right = &det;
                }
            }
        }
    }

    // 4. 归一化立体对
    normalizeStereoPair(left_roi, right_roi, best_right, left_img_size, right_img_size);

    return {left_roi, right_roi};
}

// ============================================================================
// 立体 RoiGroup 对生成（支持双类别）
// ============================================================================

std::pair<RoiGroup, RoiGroup> RoiGenerator::generateStereoGroup(
    const std::vector<Detection>& detections_left,
    const std::vector<Detection>& detections_right,
    const cv::Size& left_img_size,
    const cv::Size& right_img_size) const {

    // 1. 为左右两侧分别独立生成 RoiGroup
    RoiGroup left_group  = generateGroup(detections_left,  left_img_size);
    RoiGroup right_group = generateGroup(detections_right, right_img_size);

    if (!left_group.valid() || !right_group.valid()) {
        return {};
    }
    // A class0 silhouette and the inner class1 pattern use different models;
    // never normalize them into an apparently valid stereo observation pair.
    if(config_.selection.enabled&&left_group.is_class1!=right_group.is_class1)
        return {};

    // 2. 归一化主 ROI 立体对
    if (left_group.primary.valid() && right_group.primary.valid()) {
        // 查找右图的 class-0 检测结果用于中心锚定
        const Detection* best_right0 = nullptr;
        {
            float best_conf = -1.0f;
            for (const auto& det : detections_right) {
                if (det.class_id == 0) {
                    if (det.confidence > best_conf) {
                        best_conf = det.confidence;
                        best_right0 = &det;
                    }
                }
            }
        }
        if (config_.selection.enabled) {
          auto selected = candidates(detections_right, right_img_size);
          int id = selected.empty()
                       ? -1
                       : (right_group.is_class1 ? selected[0].class1_index
                                                : selected[0].class0_index);
          best_right0 = id < 0 ? nullptr : &detections_right[id];
        }
        normalizeStereoPair(left_group.primary, right_group.primary,
                           best_right0, left_img_size, right_img_size);
    }

    // 3. 归一化副 ROI 立体对（仅当两侧均为双 ROI 模式时）
    if (left_group.is_dual && right_group.is_dual) {
        if (left_group.secondary.valid() && right_group.secondary.valid()) {
            const Detection* best_right1 = nullptr;
            {
                float best_conf = -1.0f;
                for (const auto& det : detections_right) {
                    if (det.class_id == 1) {
                        if (det.confidence > best_conf) {
                            best_conf = det.confidence;
                            best_right1 = &det;
                        }
                    }
                }
            }
            if (config_.selection.enabled) {
              auto selected = candidates(detections_right, right_img_size);
              int id = selected.empty() ? -1 : selected[0].class1_index;
              best_right1 = id < 0 ? nullptr : &detections_right[id];
            }
            normalizeStereoPair(left_group.secondary, right_group.secondary,
                               best_right1, left_img_size, right_img_size);
        }
    } else {
        // If only one side is dual, clear secondary on both sides for consistency
        left_group.is_dual = false;
        left_group.secondary = RoiRect{};
        right_group.is_dual = false;
        right_group.secondary = RoiRect{};
    }

    return {left_group, right_group};
}

// ============================================================================
// Detection → RoiRect Conversion
// ============================================================================

RoiRect RoiGenerator::detectionToRoi(const Detection& det,
                                      const cv::Size& img_size,
                                      float expand_ratio, int min_size) {
    const float bw = det.bbox.width;
    const float bh = det.bbox.height;

    // Expand bounding box
    const float dx = bw * expand_ratio;
    const float dy = bh * expand_ratio;

    float x = det.bbox.x - dx;
    float y = det.bbox.y - dy;
    float w = bw + 2.0f * dx;
    float h = bh + 2.0f * dy;

    // Clamp to image boundaries
    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    if (x + w > static_cast<float>(img_size.width)) {
        w = static_cast<float>(img_size.width) - x;
    }
    if (y + h > static_cast<float>(img_size.height)) {
        h = static_cast<float>(img_size.height) - y;
    }

    // Enforce minimum size
    w = std::max(w, static_cast<float>(min_size));
    h = std::max(h, static_cast<float>(min_size));

    // Re-clamp after min-size enforcement
    if (x + w > static_cast<float>(img_size.width)) {
        x = std::max(0.0f, static_cast<float>(img_size.width) - w);
    }
    if (y + h > static_cast<float>(img_size.height)) {
        y = std::max(0.0f, static_cast<float>(img_size.height) - h);
    }

    RoiRect roi;
    roi.x = static_cast<int>(std::round(x));
    roi.y = static_cast<int>(std::round(y));
    roi.width = static_cast<int>(std::round(w));
    roi.height = static_cast<int>(std::round(h));

    return roi;
}

} // namespace gpnp
