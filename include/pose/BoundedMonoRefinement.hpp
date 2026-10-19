#pragma once
#include "common/Types.hpp"
namespace gpnp {
// Refit all correspondences with normal constrained to the near-nadir cone.
// This changes R AND t through image residual minimization, never just an axis.
bool refineMonoWithinTilt(const std::vector<cv::Point2f>& image,
        const std::vector<Eigen::Vector3d>& model, const Eigen::Matrix3d& K,
        double max_tilt_deg, Eigen::Matrix3d& R, Eigen::Vector3d& t);
}
