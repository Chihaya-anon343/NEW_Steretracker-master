#pragma once

/**
 * @file PoseUtils.hpp
 * @brief Shared utility functions for template loading, IoU calculation,
 *        ROI normalization, corner file parsing, and point ordering.
 *
 * Used by BinaryCornerExtractor and TinyTargetExtractor strategies.
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace gpnp {

// ============================================================================
// 模板加载
// ============================================================================

/// Load template PNG+TXT pairs from a directory.
///
/// For each *_degrees.txt file found, reads the corresponding *_degrees.png,
/// parses corner coordinates, precomputes image_bool = (image > 127).
/// Returns a vector sorted by ascending angle.
///
/// @param template_dir   Directory containing PNG+TXT template pairs
/// @param reshape_to_1x2 If true, reshape each image to 1×N (legacy compat)
/// @return Sorted vector of TemplateData (angle-ascending)
std::vector<TemplateData> loadTemplates(
    const std::string& template_dir,
    bool reshape_to_1x2 = false);

// ============================================================================
// 重叠度 / IoU
// ============================================================================

/// Compute IoU (Intersection over Union) of two binary images.
///
/// Both images must be the same size and CV_8UC1 (0/255).
///
/// @param a  First binary image
/// @param b  Second binary image
/// @return   IoU ∈ [0, 1]; returns 0.0 if union is empty
double calculateOverlap(const cv::Mat& a, const cv::Mat& b);

// ============================================================================
// ROI 规范化
// ============================================================================

/// Extract the white region from a binary image, crop to its bounding square,
/// and resize to target_size.
///
/// @param binary_img   Binary input image (CV_8UC1, 0/255)
/// @param target_size  Target square side length in pixels
/// @return             (normalized_image, success_flag)
std::pair<cv::Mat, bool> extractAndNormalizeRoi(
    const cv::Mat& binary_img, int target_size);

// ============================================================================
// 角点文件解析
// ============================================================================

/// Parse corner coordinates from a _degrees.txt file.
///
/// File format:
///   # Comment lines (skipped)
///   Corner_0: 16.00, -0.17
///   Corner_1: 16.00, 35.69
///   ...
///
/// @param txt_path  Path to the _degrees.txt file
/// @return          Vector of parsed corner coordinates
std::vector<cv::Point2f> readCorners(const std::string& txt_path);

// ============================================================================
// 点排序
// ============================================================================

/// Order 4 corner points as TL → TR → BR → BL (clockwise from top-left).
///
/// Algorithm: sum x+y for each point — argmin = TL, argmax = BR.
/// Of the remaining two, the one with smaller x is BL (left side), larger x is TR.
///
/// @param pts  Input points (must have exactly 4 elements)
/// @return     Ordered points: [TL, TR, BR, BL]
std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts);

} // namespace gpnp
