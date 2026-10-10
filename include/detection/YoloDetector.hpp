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
#include "detection/IYoloDetector.hpp"
#include "detection/YoloDecode.hpp"
#include "detection/YoloObbDecode.hpp"
#include "detection/YoloPreprocess.hpp"

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

class YoloDetector : public IYoloDetector {
public:
    explicit YoloDetector(const YoloConfig& cfg);
    ~YoloDetector() override;

    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    bool isInitialized() const { return session_ != nullptr; }

    /// 对图像运行检测。成功时将填充 `detections`。
    Status detect(const cv::Mat& image, std::vector<Detection>& detections) override;

    /// 对图像运行 OBB 检测（需 config_.is_obb == true）。成功时将填充 `detections`。
    Status detectObb(const cv::Mat& image, std::vector<ObbDetection>& detections) override;

private:
    // --- 预处理辅助函数 ---

    static std::pair<int, int> resolveInputShape(const std::vector<int64_t>& shape);
    static std::vector<float> blobFromImage(const cv::Mat& img,
                                            std::array<int64_t, 4>& input_shape);
    void preprocess(const cv::Mat& img, cv::Mat& out,
                    float& ratio, float& dw, float& dh) const;

    // --- 后处理 ---

    Status postprocess(const std::vector<Ort::Value>& preds,
                       const cv::Mat& img0,
                       float ratio, float dw, float dh,
                       std::vector<Detection>& detections) const;

    /// OBB 后处理：输出 [1, 4+nc+1, N]，末通道为旋转角（弧度）。
    Status postprocessObb(const std::vector<Ort::Value>& preds,
                          const cv::Mat& img0,
                          float ratio, float dw, float dh,
                          std::vector<ObbDetection>& detections) const;

    /// 共享推理（letterbox → blob → ONNX Run），detect/detectObb 复用。
    Status runInference(const cv::Mat& image,
                        std::vector<Ort::Value>& outputs,
                        float& ratio, float& dw, float& dh);

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

inline Status YoloDetector::runInference(const cv::Mat& image,
                                         std::vector<Ort::Value>& outputs,
                                         float& ratio, float& dw, float& dh) {
    if (!session_) {
        return Status::ModelLoadFailed;
    }
    if (image.empty()) {
        return Status::EmptyInput;
    }

    cv::Mat bgr = ensureBgrImage(image);

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

    outputs = session_->Run(Ort::RunOptions{nullptr},
                            input_names, &input_tensor, 1,
                            output_names_.data(), output_names_.size());
    return Status::Success;
}

inline Status YoloDetector::detect(const cv::Mat& image,
                                    std::vector<Detection>& detections) {
    detections.clear();

    try {
        float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
        std::vector<Ort::Value> outputs;
        Status st = runInference(image, outputs, ratio, dw, dh);
        if (st != Status::Success) {
            return st;
        }

        cv::Mat bgr = ensureBgrImage(image);
        return postprocess(outputs, bgr, ratio, dw, dh, detections);

    } catch (const Ort::Exception& e) {
        return Status::InferenceFailed;
    } catch (const std::exception&) {
        return Status::UnknownError;
    }
}

inline Status YoloDetector::detectObb(const cv::Mat& image,
                                       std::vector<ObbDetection>& detections) {
    detections.clear();

    if (!config_.is_obb) {
        return Status::UnknownError;   // OBB 解码需 is_obb 配置
    }

    try {
        float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
        std::vector<Ort::Value> outputs;
        Status st = runInference(image, outputs, ratio, dw, dh);
        if (st != Status::Success) {
            return st;
        }

        cv::Mat bgr = ensureBgrImage(image);
        return postprocessObb(outputs, bgr, ratio, dw, dh, detections);

    } catch (const Ort::Exception& e) {
        return Status::InferenceFailed;
    } catch (const std::exception&) {
        return Status::UnknownError;
    }
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

// --- 预处理：letterbox → BGR→RGB → 归一化 ---
// letterbox 几何来自 YoloPreprocess.hpp（与 RKNN 后端共享，保证反变换互逆）

inline void YoloDetector::preprocess(const cv::Mat& img, cv::Mat& out,
                                      float& ratio, float& dw, float& dh) const {
    cv::Mat letterboxed;
    letterboxImage(img, letterboxed,
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

// --- 后处理：解码原始 YOLOv8 输出 + NMS ---

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

    // 原始 YOLOv8 输出：[1, 4+nc, N] (BCN) 或 [1, N, 4+nc] (BNC)，
    // 也可能是去掉 batch 维的 rank-2。区分"属性维"(4+nc) 与"预测框维"(N)。
    size_t num_preds = 0, num_attrs = 0;
    bool shape_is_bcn = false;
    if (!resolveYoloOutputLayout(shape.data(), shape.size(),
                                 num_preds, num_attrs, shape_is_bcn)) {
        return Status::InferenceFailed;
    }

    if (num_attrs < 5 || num_preds == 0) {
        return Status::InferenceFailed;
    }
    const size_t num_classes = num_attrs - 4;

    const float* data = output.GetTensorData<float>();
    const size_t total = tensor_info.GetElementCount();
    if (!data || total < num_attrs * num_preds) {
        return Status::InferenceFailed;
    }

    const int img_h = img0.rows;
    const int img_w = img0.cols;

    detections = decodeYoloOutput(data, shape_is_bcn, num_preds, num_classes,
                                  ratio, dw, dh, img_w, img_h,
                                  config_.conf_threshold, config_.iou_threshold);

    // 空结果并非错误
    return Status::Success;
}

// --- 后处理（OBB）：解码 [1, 4+nc+1, N] + 旋转 NMS ---

inline Status YoloDetector::postprocessObb(const std::vector<Ort::Value>& preds,
                                           const cv::Mat& img0,
                                           float ratio, float dw, float dh,
                                           std::vector<ObbDetection>& detections) const {
    if (preds.empty()) {
        return Status::InferenceFailed;
    }
    if (img0.empty()) {
        return Status::EmptyInput;
    }

    const auto& output = preds[0];
    auto tensor_info = output.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    // OBB 输出：[1, 4+nc+1, N] (BCN) 或 [1, N, 4+nc+1] (BNC)
    size_t num_preds = 0, num_attrs = 0;
    bool shape_is_bcn = false;
    if (!resolveYoloOutputLayout(shape.data(), shape.size(),
                                 num_preds, num_attrs, shape_is_bcn)) {
        return Status::InferenceFailed;
    }

    if (num_attrs < 6 || num_preds == 0) {
        return Status::InferenceFailed;
    }
    const size_t num_classes = num_attrs - 5;               // 多出的 1 通道为角度

    const float* data = output.GetTensorData<float>();
    const size_t total = tensor_info.GetElementCount();
    if (!data || total < num_attrs * num_preds) {
        return Status::InferenceFailed;
    }

    const int img_h = img0.rows;
    const int img_w = img0.cols;

    detections = decodeYoloObbOutput(data, shape_is_bcn, num_preds, num_classes,
                                     ratio, dw, dh, img_w, img_h,
                                     config_.conf_threshold, config_.iou_threshold);

    return Status::Success;
}

} // namespace gpnp
