#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpnp {

// Opt-in monocular TT changes. All distances are in original ROI pixels.
struct TinyTargetRobustnessConfig {
    bool enabled = false;
    bool diagnostics = false;
    bool direction_match = true;
    bool guarded_refinement = true;
    bool pixel_center_mapping = false;
    double min_direction_iou = 0.55;
    double min_direction_gap = 0.04;
    double min_span_px = 4.0;
    double max_refine_shift_px = 1.0;
    double min_gradient_ratio = 0.02;
};

inline void validateTinyTargetRobustnessConfig(const TinyTargetRobustnessConfig& c) {
    auto between = [](double v, double lo, double hi) {
        return std::isfinite(v) && v >= lo && v <= hi;
    };
    if (!between(c.min_direction_iou, 0.0, 1.0) ||
        !between(c.min_direction_gap, 0.0, 1.0) ||
        !between(c.min_span_px, 1.0, 256.0) ||
        !between(c.max_refine_shift_px, 0.0, 4.0) ||
        !between(c.min_gradient_ratio, 0.0, 1.0))
        throw std::invalid_argument("Invalid tiny_target.robustness settings");
}

inline TinyTargetRobustnessConfig readTinyTargetRobustnessConfig(const cv::FileNode& node) {
    TinyTargetRobustnessConfig c;
    if (node.empty()) return c;
    if (!node.isMap()) throw std::invalid_argument("tiny_target.robustness must be an object");
    auto flag = [&](const char* key, bool& value) {
        const auto field = node[key];
        if (field.empty()) return;
        if (!field.isInt() || (static_cast<int>(field) != 0 && static_cast<int>(field) != 1))
            throw std::invalid_argument(std::string("tiny_target.robustness.") + key + " must be boolean");
        value = static_cast<int>(field) != 0;
    };
    auto real = [&](const char* key, double& value) {
        const auto field = node[key];
        if (field.empty()) return;
        if (!field.isInt() && !field.isReal())
            throw std::invalid_argument(std::string("tiny_target.robustness.") + key + " must be numeric");
        value = static_cast<double>(field);
    };
    flag("enabled", c.enabled); flag("diagnostics", c.diagnostics);
    flag("direction_match", c.direction_match);
    flag("guarded_refinement", c.guarded_refinement);
    flag("pixel_center_mapping", c.pixel_center_mapping);
    real("min_direction_iou", c.min_direction_iou);
    real("min_direction_gap", c.min_direction_gap);
    real("min_span_px", c.min_span_px);
    real("max_refine_shift_px", c.max_refine_shift_px);
    real("min_gradient_ratio", c.min_gradient_ratio);
    validateTinyTargetRobustnessConfig(c);
    return c;
}

struct TinyTargetQuality {
    std::string status = "not_run";
    double span_px = 0.0, span_x_px = 0.0, span_y_px = 0.0, area_px = 0.0;
    double legacy_iou = 0.0, direction_iou = 0.0, direction_gap = 0.0;
    // Cyclic source start in unique image TL,TR,BR,BL winding. Physical IDs
    // start at this index. This is not an Euler angle or a template filename.
    int quadrant = -1, legacy_quadrant = -1;
    int refined_count = 0;
    double max_refine_shift_px = 0.0;
    double max_proposed_refine_shift_px = 0.0;
    bool geometry_valid = false, direction_reliable = false;
    std::array<double, 4> direction_scores{{0.0, 0.0, 0.0, 0.0}};
    std::vector<unsigned char> accepted_refinement;
    std::vector<cv::Point2f> initial_corners, final_corners;
    cv::Mat normalized, reference;
    std::map<std::string, double> stage_ms;
};

} // namespace gpnp
