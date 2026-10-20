#pragma once

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace gpnp::utils {

// Display the board's upward normal (-model Z), without changing its pose or
// right-handed coordinate frame. Limit length to keep the tip before the camera.
inline double upNormalZ(double preferred_length, double camera_depth) {
    return -std::min(preferred_length, .5 * camera_depth);
}

inline void drawUpNormal(cv::Mat& image, cv::Point2d origin, cv::Point2d tip,
                         int thickness = 2) {
    if (image.empty() || !std::isfinite(origin.x) || !std::isfinite(origin.y) ||
        !std::isfinite(tip.x) || !std::isfinite(tip.y)) return;
    // Avoid overflowing OpenCV's integer drawing coordinates for a near-plane
    // projection; such an arrow cannot be shown meaningfully in the viewport.
    if (std::max({std::abs(origin.x), std::abs(origin.y),
                  std::abs(tip.x), std::abs(tip.y)}) > 1e7) return;
    cv::Point a(cvRound(origin.x), cvRound(origin.y));
    cv::Point b(cvRound(tip.x), cvRound(tip.y));
    if (!cv::clipLine(image.size(), a, b)) return;
    cv::arrowedLine(image, a, b, cv::Scalar(255,0,0), thickness, cv::LINE_AA,
                    0, .15);
    // Hershey fonts cannot render Chinese; make the sign explicit in all panels.
    const cv::Point label(std::clamp(b.x + 6, 0, std::max(0, image.cols - 85)),
                          std::clamp(b.y - 6, std::min(14, image.rows - 1),
                                     std::max(14, image.rows - 4)));
    cv::putText(image, "-Z (up)", label, cv::FONT_HERSHEY_SIMPLEX, .45,
                cv::Scalar(0,0,0), 3, cv::LINE_AA);
    cv::putText(image, "-Z (up)", label, cv::FONT_HERSHEY_SIMPLEX, .45,
                cv::Scalar(255,150,50), 1, cv::LINE_AA);
}

} // namespace gpnp::utils
