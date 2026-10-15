#pragma once

/**
 * @file YoloPreprocess.hpp
 * @brief YOLO 检测共享预处理（纯函数，无推理后端依赖）。
 *
 * 从 YoloDetector 的私有静态函数抽取，供 ONNX / RKNN 两个后端共用，
 * 保证 letterbox 的 ratio/dw/dh 在不同后端间逐位一致——检测坐标的
 * 反 letterbox 变换依赖这三个值，任何后端间差异都会造成框偏移。
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gpnp {
namespace yolo {

/// 确保输入为 3 通道 BGR（灰度/BGRA 自动转换）
inline cv::Mat ensureBGR(const cv::Mat& img) {
    if (img.empty()) {
        throw std::runtime_error("Input image is empty.");
    }
    if (img.channels() == 3) {
        return img;
    }
    cv::Mat bgr;
    if (img.channels() == 1) {
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (img.channels() == 4) {
        cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    throw std::runtime_error("Unsupported image channels: " +
                             std::to_string(img.channels()));
}

/// Letterbox：等比缩放 + 两侧 padding（与 Ultralytics 约定一致）
inline void letterbox(const cv::Mat& img, cv::Mat& out,
                      const cv::Size& new_shape,
                      float& ratio, float& dw, float& dh,
                      const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
    const int src_h = img.rows;
    const int src_w = img.cols;

    ratio = std::min(
        static_cast<float>(new_shape.height) / static_cast<float>(src_h),
        static_cast<float>(new_shape.width) / static_cast<float>(src_w));

    int new_unpad_w = static_cast<int>(std::round(src_w * ratio));
    int new_unpad_h = static_cast<int>(std::round(src_h * ratio));
    new_unpad_w = std::max(1, new_unpad_w);
    new_unpad_h = std::max(1, new_unpad_h);

    dw = (new_shape.width - new_unpad_w) / 2.0f;
    dh = (new_shape.height - new_unpad_h) / 2.0f;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h));

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    top = std::max(0, top);
    bottom = std::max(0, bottom);
    left = std::max(0, left);
    right = std::max(0, right);

    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, color);

    if (out.rows != new_shape.height || out.cols != new_shape.width) {
        cv::resize(out, out, new_shape);
    }
}

} // namespace yolo
} // namespace gpnp
