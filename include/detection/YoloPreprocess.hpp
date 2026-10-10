#pragma once

/**
 * @file YoloPreprocess.hpp
 * @brief YOLO 推理前预处理共享函数（无推理后端依赖）。
 *
 * ONNX Runtime 与 RKNN 两个检测器后端共用同一 letterbox 几何与 BGR 规整，
 * 保证 decodeYoloOutput 的反变换 (x-dw)/ratio 与预处理严格互逆。
 * 代码体迁自 YoloDetector.hpp（行为逐位一致）。
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gpnp {

/// 输入统一转为 3 通道 BGR。空图或不支持的通道数时抛异常。
inline cv::Mat ensureBgrImage(const cv::Mat& img) {
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

/// Letterbox：等比缩放 + 居中填充。ratio/dw/dh 供解码阶段反变换回原图坐标。
inline void letterboxImage(const cv::Mat& img, cv::Mat& out,
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

} // namespace gpnp
