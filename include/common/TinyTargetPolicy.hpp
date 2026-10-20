#pragma once

#include "common/Types.hpp"
#include <cmath>
#include <stdexcept>

namespace gpnp {

inline TinyTargetVerificationConfig readTinyTargetVerificationConfig(const cv::FileNode& node) {
  TinyTargetVerificationConfig c;
  if (node.empty()) return c;
  if (!node.isMap()) throw std::invalid_argument("roi_selection.tt_verification must be an object");
  if (!node["enabled"].empty()) {
    const auto flag = node["enabled"];
    if (!flag.isInt() || (int(flag) != 0 && int(flag) != 1))
      throw std::invalid_argument("tt_verification.enabled must be boolean");
    c.enabled = int(flag) != 0;
  }
  if (!node["allow_small_target_path"].empty()) {
    const auto flag = node["allow_small_target_path"];
    if (!flag.isInt() || (int(flag) != 0 && int(flag) != 1))
      throw std::invalid_argument("tt_verification.allow_small_target_path must be boolean");
    c.allow_small_target_path = int(flag) != 0;
  }
  auto number = [&](const char* key, double& value) {
    if (node[key].empty()) return;
    if (!node[key].isReal() && !node[key].isInt())
      throw std::invalid_argument(std::string("tt_verification.") + key + " must be numeric");
    value = double(node[key]);
  };
  number("min_shape_class0", c.min_shape_class0);
  number("min_shape_class1", c.min_shape_class1);
  number("small_min_shape_class0", c.small_min_shape_class0);
  number("small_min_shape_class1", c.small_min_shape_class1);
  number("small_span_class0", c.small_span_class0);
  number("small_span_class1", c.small_span_class1);
  number("min_direction_iou", c.min_direction_iou);
  number("min_direction_gap", c.min_direction_gap);
  for (double value : {c.min_shape_class0, c.min_shape_class1, c.small_min_shape_class0,
                       c.small_min_shape_class1, c.min_direction_iou, c.min_direction_gap})
    if (!std::isfinite(value) || value < 0 || value > 1)
      throw std::invalid_argument("tt_verification scores must be finite in [0,1]");
  if (!std::isfinite(c.small_span_class0) || !std::isfinite(c.small_span_class1) ||
      c.small_span_class0 <= 0 || c.small_span_class1 <= 0 ||
      c.small_min_shape_class0 > c.min_shape_class0 || c.small_min_shape_class1 > c.min_shape_class1)
    throw std::invalid_argument("Invalid tt_verification size or score ordering");
  return c;
}

struct TinyTargetRoiEvidence {
  bool extracted{false}, finite{false}, geometry_valid{false}, emitted_direction_consistent{false};
  int points{0};
  double legacy_iou{0}, direction_iou{0}, direction_gap{0};
  double span_x{0}, span_y{0};
};

struct TinyTargetRoiDecision {
  bool accepted{false}, small_target_path{false};
  std::string reason;
};

inline TinyTargetRoiDecision evaluateTinyTargetRoi(const TinyTargetRoiEvidence& e,
                                                  const RoiSelectionConfig& config,
                                                  bool class1, int roi_width, int roi_height) {
  auto fail = [](const char* reason) { return TinyTargetRoiDecision{false, false, reason}; };
  if (!e.extracted || e.points != 4) return fail("tt_extraction_failed");
  if (!e.finite || !std::isfinite(e.legacy_iou) || !std::isfinite(e.span_x) ||
      !std::isfinite(e.span_y) || e.span_x < 0 || e.span_y < 0) return fail("tt_nonfinite_observation");
  if (std::min(e.span_x, e.span_y) < config.min_observed_span_px) return fail("tt_observed_span");
  if (e.span_x < roi_width * config.min_component_ratio ||
      e.span_y < roi_height * config.min_component_ratio) return fail("tt_roi_coverage");
  const auto& c = config.tt_verification;
  if (!c.enabled)
    return e.legacy_iou >= .75 ? TinyTargetRoiDecision{true, false, "tt_legacy_accepted"}
                              : fail("tt_legacy_shape");
  if (!e.geometry_valid) return fail("tt_invalid_quadrilateral");
  const double strict = class1 ? c.min_shape_class1 : c.min_shape_class0;
  if (e.legacy_iou >= strict) return {true, false, "tt_strict_accepted"};
  if (!c.allow_small_target_path) return fail("tt_shape_below_strict");
  const double floor = class1 ? c.small_min_shape_class1 : c.small_min_shape_class0;
  const double limit = class1 ? c.small_span_class1 : c.small_span_class0;
  if (e.legacy_iou < floor) return fail("tt_shape_below_floor");
  if (std::max(e.span_x, e.span_y) > limit) return fail("tt_outside_small_target_range");
  if (!std::isfinite(e.direction_iou) || !std::isfinite(e.direction_gap) ||
      e.direction_iou < c.min_direction_iou) return fail("tt_direction_shape");
  if (e.direction_gap < c.min_direction_gap) return fail("tt_direction_ambiguous");
  if (!e.emitted_direction_consistent) return fail("tt_direction_inconsistent");
  return {true, true, "tt_small_target_verified"};
}

} // namespace gpnp
