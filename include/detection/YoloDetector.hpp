#pragma once

/**
 * @file YoloDetector.hpp
 * @brief ONNX Runtime YOLO 目标检测器（纯头文件）。
 *
 * 验证过的推理流水线：letterbox → BGR→RGB → 归一化 → ONNX → 解码。
 * 使用 common/Types.hpp 中的 Detection / Status / DeviceType / YoloConfig 类型。
 *
 * 构造函数执行所有 ONNX 初始化（失败时抛出异常）。
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "common/Types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace gpnp {

// ============================================================================
// YOLO 目标检测器（纯头文件）
// ============================================================================

class YoloDetector {
public:
    explicit YoloDetector(const YoloConfig& cfg);
    ~YoloDetector();

    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    bool isInitialized() const { return session_ != nullptr; }

    /// 对图像运行检测。成功时将填充 `detections`。
    Status detect(const cv::Mat& image, std::vector<Detection>& detections);

private:
    // --- 预处理辅助函数 ---

    static cv::Mat ensureBGR(const cv::Mat& img);
    static std::pair<int, int> resolveInputShape(const std::vector<int64_t>& shape);
    static void letterbox(const cv::Mat& img, cv::Mat& out,
                          const cv::Size& new_shape,
                          float& ratio, float& dw, float& dh,
                          const cv::Scalar& color = cv::Scalar(114, 114, 114));
    static std::vector<float> blobFromImage(const cv::Mat& img,
                                            std::array<int64_t, 4>& input_shape);
    void preprocess(const cv::Mat& img, cv::Mat& out,
                    float& ratio, float& dw, float& dh) const;

    // --- 后处理 ---

    Status postprocess(const std::vector<Ort::Value>& preds,
                       const cv::Mat& img0,
                       float ratio, float dw, float dh,
                       std::vector<Detection>& detections) const;
    static bool isValidNumber(double v);

    // --- 成员变量 ---

    YoloConfig config_;
    std::pair<int, int> input_hw_{640, 640};

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> output_names_;
};

// ============================================================================
// 实现
// ============================================================================

inline YoloDetector::YoloDetector(const YoloConfig& cfg)
    : config_(cfg)
    , env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector")
{
    // ---- 提前验证：在创建 ONNX 会话之前检查文件是否存在 ----
    {
        std::ifstream f(config_.model_path, std::ios::binary | std::ios::ate);
        if (!f.good()) {
            throw std::runtime_error(
                "ONNX model file not found or unreadable: " + config_.model_path);
        }
        auto size = f.tellg();
        if (size <= 0) {
            throw std::runtime_error(
                "ONNX model file is empty: " + config_.model_path);
        }
        // 快速完整性检查：读取前 4 字节（ONNX protobuf 以 0x08 开头）
        f.seekg(0);
        char header[4] = {};
        f.read(header, 4);
        std::cerr << "[YoloDetector] Model file: " << config_.model_path
                  << " (" << size << " bytes, header: "
                  << std::hex << (int)(unsigned char)header[0] << " "
                  << (int)(unsigned char)header[1] << " "
                  << (int)(unsigned char)header[2] << " "
                  << (int)(unsigned char)header[3] << std::dec << ")"
                  << std::endl;
    }

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(config_.intra_op_threads);

    bool cuda_ok = false;
    if (config_.device == DeviceType::CUDA || config_.device == DeviceType::Auto) {
        std::cerr << "[YoloDetector] Trying CUDA..." << std::flush;
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_opts);
            cuda_ok = true;
            std::cerr << " enabled" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << " unavailable (" << e.what()
                      << "), falling back to CPU" << std::endl;
        } catch (...) {
            std::cerr << " unavailable (unknown error), falling back to CPU" << std::endl;
        }
    }

    if (!cuda_ok && config_.device != DeviceType::CPU) {
        std::cerr << "[YoloDetector] Trying CPU..." << std::flush;
    }

    std::cerr << "[YoloDetector] Creating ONNX session..." << std::endl;
    session_ = std::make_unique<Ort::Session>(env_, config_.model_path.c_str(),
                                               session_options);
    std::cerr << "[YoloDetector] ONNX session created successfully" << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    {
        auto in_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = in_name.get();
    }

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_hw_ = resolveInputShape(tensor_info.GetShape());

    const size_t output_count = session_->GetOutputCount();
    output_names_storage_.reserve(output_count);
    output_names_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        auto out_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_storage_.emplace_back(out_name.get());
    }
    for (const auto& s : output_names_storage_) {
        output_names_.push_back(s.c_str());
    }
}

inline YoloDetector::~YoloDetector() = default;

// --- 主检测入口点 ---

inline Status YoloDetector::detect(const cv::Mat& image,
                                    std::vector<Detection>& detections) {
    detections.clear();

    if (!session_) {
        return Status::ModelLoadFailed;
    }
    if (image.empty()) {
        return Status::EmptyInput;
    }

    try {
        cv::Mat bgr = ensureBGR(image);

        float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
        cv::Mat input_tensor_img;
        preprocess(bgr, input_tensor_img, ratio, dw, dh);

        std::array<int64_t, 4> tensor_shape{};
        auto input_tensor_values = blobFromImage(input_tensor_img, tensor_shape);

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            tensor_shape.data(), tensor_shape.size());

        const char* input_names[] = {input_name_.c_str()};

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_names, &input_tensor, 1,
                                     output_names_.data(), output_names_.size());

        return postprocess(outputs, bgr, ratio, dw, dh, detections);

    } catch (const Ort::Exception& e) {
        return Status::InferenceFailed;
    } catch (const std::exception&) {
        return Status::UnknownError;
    }
}

// --- 确保 BGR 格式 ---

inline cv::Mat YoloDetector::ensureBGR(const cv::Mat& img) {
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

// --- 从 ONNX 模型获取输入尺寸 ---

inline std::pair<int, int> YoloDetector::resolveInputShape(
    const std::vector<int64_t>& shape) {
    if (shape.size() >= 4) {
        int64_t h = shape[2];
        int64_t w = shape[3];
        if (h > 0 && w > 0) {
            return {static_cast<int>(h), static_cast<int>(w)};
        }
    }
    return {640, 640};
}

// --- Letterbox：缩放 + 填充 ---

inline void YoloDetector::letterbox(const cv::Mat& img, cv::Mat& out,
                                     const cv::Size& new_shape,
                                     float& ratio, float& dw, float& dh,
                                     const cv::Scalar& color) {
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

// --- 预处理：letterbox → BGR→RGB → 归一化 ---

inline void YoloDetector::preprocess(const cv::Mat& img, cv::Mat& out,
                                      float& ratio, float& dw, float& dh) const {
    cv::Mat letterboxed;
    letterbox(img, letterboxed,
              cv::Size(input_hw_.second, input_hw_.first), ratio, dw, dh);

    cv::cvtColor(letterboxed, out, cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32FC3, 1.0 / 255.0);
}

// --- 图像转 NCHW blob ---

inline std::vector<float> YoloDetector::blobFromImage(
    const cv::Mat& img, std::array<int64_t, 4>& input_shape) {
    const int h = img.rows;
    const int w = img.cols;
    const size_t plane_size = static_cast<size_t>(h) * static_cast<size_t>(w);
    std::vector<float> input_tensor_values(1 * 3 * plane_size);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(
            input_tensor_values.data() + static_cast<size_t>(c) * plane_size,
            channels[c].data, plane_size * sizeof(float));
    }

    input_shape = {1, 3, h, w};
    return input_tensor_values;
}

// --- 后处理：解码边界框 ---

inline Status YoloDetector::postprocess(const std::vector<Ort::Value>& preds,
                                          const cv::Mat& img0,
                                          float ratio, float dw, float dh,
                                          std::vector<Detection>& detections) const {
    if (preds.empty()) {
        return Status::InferenceFailed;
    }
    if (img0.empty()) {
        return Status::EmptyInput;
    }

    const auto& output = preds[0];
    auto tensor_info = output.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    if (shape.size() != 2 && shape.size() != 3) {
        return Status::InferenceFailed;
    }

    size_t rows = 0, cols = 0;
    if (shape.size() == 3) {
        rows = static_cast<size_t>(shape[1]);
        cols = static_cast<size_t>(shape[2]);
    } else {
        rows = static_cast<size_t>(shape[0]);
        cols = static_cast<size_t>(shape[1]);
    }

    if (cols < 6 || rows == 0) {
        return Status::InferenceFailed;
    }

    const float* data = output.GetTensorData<float>();
    const size_t total = tensor_info.GetElementCount();
    if (!data || total < rows * cols) {
        return Status::InferenceFailed;
    }

    const int img_h = img0.rows;
    const int img_w = img0.cols;

    for (size_t i = 0; i < rows; ++i) {
        const float* p = data + i * cols;

        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
            !std::isfinite(p[2]) || !std::isfinite(p[3]) ||
            !std::isfinite(p[4]) || !std::isfinite(p[5])) {
            continue;
        }

        float x1 = p[0];
        float y1 = p[1];
        float x2 = p[2];
        float y2 = p[3];
        float conf = p[4];
        int cls = static_cast<int>(std::round(p[5]));

        if (!std::isfinite(conf) || conf < config_.conf_threshold) {
            continue;
        }

        // 反向 letterbox 变换
        x1 = (x1 - dw) / ratio;
        x2 = (x2 - dw) / ratio;
        y1 = (y1 - dh) / ratio;
        y2 = (y2 - dh) / ratio;

        if (!isValidNumber(x1) || !isValidNumber(y1) ||
            !isValidNumber(x2) || !isValidNumber(y2)) {
            continue;
        }
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        x1 = std::clamp(x1, 0.0f, static_cast<float>(img_w));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(img_w));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img_h));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(img_h));

        Detection det;
        det.class_id = cls;
        det.confidence = conf;
        det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

        detections.push_back(std::move(det));
    }

    // 空结果并非错误
    return Status::Success;
}

// --- 数值有效性检查 ---

inline bool YoloDetector::isValidNumber(double v) {
    return std::isfinite(v);
}

} // namespace gpnp
