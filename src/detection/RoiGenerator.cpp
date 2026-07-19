/**
 * @file RoiGenerator.cpp
 * @brief YOLO 检测 → RoiRect 转换的实现。
 */

#include "detection/RoiGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace gpnp {

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
    // 1. 始终先尝试 class 0
    RoiRect class0_roi = generate(detections, image_size, 0);
    if (!class0_roi.valid()) {
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
        return {};
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
