#pragma once

#include "Types.hpp"
#include <stdexcept>

namespace gpnp {

inline void validateTemplateGeometry(const TemplateGeometryConfig& c) {
    if (!c.class1_center_in_class0_mm.allFinite())
        throw std::invalid_argument("template_geometry offset must contain two finite mm values");
}

// Apply to newly constructed points, never to an already placed model.
// Preserve physical scale, Z, axes, ordering and IDs.
inline void placeTemplateInClass0(std::vector<Eigen::Vector3d>& points,
                                  bool is_class1, const TemplateGeometryConfig& c) {
    validateTemplateGeometry(c);
    if (!c.enabled || points.empty()) return;
    Eigen::Vector2d lo = points.front().head<2>(), hi = lo;
    for (const auto& p : points) {
        if (!p.allFinite()) throw std::invalid_argument("Non-finite template model point");
        lo = lo.cwiseMin(p.head<2>());
        hi = hi.cwiseMax(p.head<2>());
    }
    const Eigen::Vector2d center = lo * 0.5 + hi * 0.5;
    const Eigen::Vector2d target = is_class1 ? c.class1_center_in_class0_mm : Eigen::Vector2d::Zero();
    for (auto& p : points) p.head<2>() += target - center;
}

inline TemplateGeometryConfig readTemplateGeometryConfig(const cv::FileNode& node) {
    TemplateGeometryConfig c;
    if (node.empty()) return c;
    if (!node.isMap()) throw std::invalid_argument("template_geometry must be an object");
    const auto enabled = node["enabled"];
    if (!enabled.empty()) {
        if (!enabled.isInt() || (static_cast<int>(enabled) != 0 && static_cast<int>(enabled) != 1))
            throw std::invalid_argument("template_geometry.enabled must be boolean");
        c.enabled = static_cast<int>(enabled) != 0;
    }
    const auto offset = node["class1_center_in_class0_mm"];
    if (!offset.empty()) {
        if (!offset.isSeq() || offset.size() != 2)
            throw std::invalid_argument("class1_center_in_class0_mm must be [x,y] in mm");
        for (int i = 0; i < 2; ++i) {
            if (!offset[i].isInt() && !offset[i].isReal())
                throw std::invalid_argument("class1_center_in_class0_mm requires numeric values");
            c.class1_center_in_class0_mm(i) = static_cast<double>(offset[i]);
        }
    } else if (c.enabled) {
        throw std::invalid_argument("enabled template_geometry requires class1_center_in_class0_mm");
    }
    validateTemplateGeometry(c);
    return c;
}

} // namespace gpnp
