#pragma once

#include "common/Types.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpnp {

inline void validateMonoPosePrior(const MonoPosePriorConfig& c) {
    if (!std::isfinite(c.max_tilt_deg) || c.max_tilt_deg <= 0.0 || c.max_tilt_deg >= 90.0)
        throw std::invalid_argument("mono_pose_prior.max_tilt_deg must be finite and in (0,90)");
}

inline MonoPosePriorConfig readMonoPosePriorConfig(const cv::FileNode& node) {
    MonoPosePriorConfig c;
    if (node.empty()) return c;
    if (!node.isMap()) throw std::invalid_argument("mono_pose_prior must be an object");
    const auto enabled = node["enabled"];
    if (!enabled.empty()) {
        if (!enabled.isInt() || (static_cast<int>(enabled) != 0 && static_cast<int>(enabled) != 1))
            throw std::invalid_argument("mono_pose_prior.enabled must be boolean");
        c.enabled = static_cast<int>(enabled) != 0;
    }
    const auto tilt = node["max_tilt_deg"];
    if (!tilt.empty()) {
        if (!tilt.isInt() && !tilt.isReal())
            throw std::invalid_argument("mono_pose_prior.max_tilt_deg must be numeric");
        c.max_tilt_deg = static_cast<double>(tilt);
    }
    validateMonoPosePrior(c);
    return c;
}

/// Empty reason means accepted. Models must lie on template Z=0.
/// A normal sign check alone cannot disambiguate two front-facing IPPE poses.
inline std::string planarPosePriorRejection(const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t, const std::vector<Eigen::Vector3d>& points,
        const MonoPosePriorConfig& c) {
    if (!c.enabled) return {};
    if (!R.allFinite() || !t.allFinite()) return "non_finite_pose";
    if ((R.transpose()*R-Eigen::Matrix3d::Identity()).norm() > 1e-6 ||
        std::abs(R.determinant()-1.0) > 1e-6) return "invalid_rotation";
    if (points.empty()) return "empty_model";
    for (const auto& p : points) {
        if (!p.allFinite()) return "non_finite_model";
        if (std::abs(p.z()) > 1e-6) return "model_not_on_z0";
        if ((R*p+t).z() <= 1e-6) return "point_behind_camera";
    }
    // C_board=-R^T*t. The camera must be on the printed/front side Z<0.
    if (R.col(2).dot(t) <= 1e-6) return "camera_on_back_side";
    constexpr double pi = 3.14159265358979323846;
    if (R(2,2) + 1e-10 < std::cos(c.max_tilt_deg*pi/180.0)) return "tilt_out_of_range";
    return {};
}

} // namespace gpnp
