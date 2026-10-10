/**
 * @file RknnYoloDetector.cpp
 * @brief RKNN NPU YOLO 检测器实现（Rockchip RK3588 等板卡）。
 *
 * 仅在 CMake 探测到 librknnrt + rknn_api.h 时编译（GPNP_WITH_RKNN）；
 * 文件级宏守护是防止被无条件加入源列表的保险。
 */

#ifdef GPNP_WITH_RKNN

#include "detection/RknnYoloDetector.hpp"
#include "detection/YoloDecode.hpp"
#include "detection/YoloObbDecode.hpp"
#include "detection/YoloPreprocess.hpp"

#include <opencv2/opencv.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpnp {

RknnYoloDetector::RknnYoloDetector(const YoloConfig& cfg)
    : config_(cfg)
{
    if (config_.device != DeviceType::CPU) {
        std::cerr << "[RknnYoloDetector] Note: device field is ignored by the RKNN backend (always NPU)\n";
    }

    // ---- 模型文件读取与校验（镜像 ONNX 后端构造期检查） ----
    std::ifstream f(config_.model_path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        throw std::runtime_error("RKNN model file not found or unreadable: " +
                                 config_.model_path);
    }
    const auto size = f.tellg();
    if (size <= 0) {
        throw std::runtime_error("RKNN model file is empty: " + config_.model_path);
    }
    std::vector<unsigned char> model_buf(static_cast<size_t>(size));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(model_buf.data()),
           static_cast<std::streamsize>(model_buf.size()));
    if (!f.good()) {
        throw std::runtime_error("RKNN model file read failed: " + config_.model_path);
    }
    std::cerr << "[RknnYoloDetector] Model file: " << config_.model_path
              << " (" << model_buf.size() << " bytes)" << std::endl;

    // ---- rknn_init ----
    int ret = rknn_init(&ctx_, model_buf.data(),
                        static_cast<uint32_t>(model_buf.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        ctx_ = 0;
        throw std::runtime_error(
            "rknn_init failed (ret=" + std::to_string(ret) + "), model: " +
            config_.model_path +
            " (需确认模型 target_platform 与本机芯片一致)");
    }

    rknn_sdk_version ver = {};
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    if (ret == RKNN_SUCC) {
        std::cerr << "[RknnYoloDetector] Driver: " << ver.drv_version
                  << ", API: " << ver.api_version << std::endl;
    }

    // ---- 输入属性：尺寸 + 数据类型 ----
    rknn_tensor_attr in_attr = {};
    in_attr.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_query(RKNN_QUERY_INPUT_ATTR) failed (ret=" +
                                 std::to_string(ret) + ")");
    }
    if (in_attr.fmt == RKNN_TENSOR_NHWC) {
        input_h_ = static_cast<int>(in_attr.dims[1]);
        input_w_ = static_cast<int>(in_attr.dims[2]);
    } else {
        // NCHW（其他布局按 NCHW 位置兜底）
        input_h_ = static_cast<int>(in_attr.dims[2]);
        input_w_ = static_cast<int>(in_attr.dims[3]);
    }
    if (input_h_ <= 0 || input_w_ <= 0) {
        throw std::runtime_error("RKNN model input shape invalid");
    }
    if (in_attr.type == RKNN_TENSOR_UINT8 || in_attr.type == RKNN_TENSOR_INT8) {
        // 量化输入（含 w8a8/i8 输入变体）：直喂原始 uint8，runtime 按模型
        // 内置 scale/zp 完成量化（归一化须已在转换时 baked 进模型）
        input_is_uint8_ = true;
    } else if (in_attr.type == RKNN_TENSOR_FLOAT32 || in_attr.type == RKNN_TENSOR_FLOAT16) {
        // 浮点输入：CPU 上 /255 后喂 float32（fp16 模型由 runtime 转换）
        input_is_uint8_ = false;
    } else {
        throw std::runtime_error(
            "Unsupported RKNN model input dtype (type=" +
            std::to_string(static_cast<int>(in_attr.type)) +
            "; 支持 UINT8/INT8 量化或 FLOAT32/FLOAT16 输入模型)");
    }
    std::cerr << "[RknnYoloDetector] Input: " << input_w_ << "x" << input_h_
              << ", dtype=" << (input_is_uint8_ ? "uint8 feed (quantized input)"
                                                : "float32 feed (float input)")
              << ", model_type=" << static_cast<int>(in_attr.type)
              << ", zp=" << in_attr.zp << ", scale=" << in_attr.scale
              << ", fmt=" << (in_attr.fmt == RKNN_TENSOR_NHWC ? "NHWC" : "NCHW")
              << std::endl;

    // ---- 输出属性：维度缓存（解码布局推导用） ----
    rknn_tensor_attr out_attr = {};
    out_attr.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_query(RKNN_QUERY_OUTPUT_ATTR) failed (ret=" +
                                 std::to_string(ret) + ")");
    }
    out_n_dims_ = out_attr.n_dims;
    if (out_n_dims_ > RKNN_MAX_DIMS) {
        out_n_dims_ = RKNN_MAX_DIMS;
    }
    out_elems_ = 1;
    std::cerr << "[RknnYoloDetector] Output dims: [";
    for (uint32_t i = 0; i < out_n_dims_; ++i) {
        out_dims_[i] = out_attr.dims[i];
        out_elems_ *= out_attr.dims[i];
        std::cerr << (i ? ", " : "") << out_dims_[i];
    }
    std::cerr << "]" << std::endl;
    if (out_elems_ == 0) {
        throw std::runtime_error("RKNN model output shape invalid");
    }
}

RknnYoloDetector::~RknnYoloDetector() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

Status RknnYoloDetector::runInference(const cv::Mat& image,
                                      std::vector<float>& out_data,
                                      float& ratio, float& dw, float& dh) {
    if (ctx_ == 0) {
        return Status::ModelLoadFailed;
    }
    if (image.empty()) {
        return Status::EmptyInput;
    }

    cv::Mat bgr;
    try {
        bgr = ensureBgrImage(image);
    } catch (const std::exception&) {
        return Status::EmptyInput;
    }

    cv::Mat letterboxed;
    letterboxImage(bgr, letterboxed, cv::Size(input_w_, input_h_), ratio, dw, dh);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb_float;
    if (!input_is_uint8_) {
        rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);
    }

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].pass_through = 0;  // runtime 负责 NHWC → 模型内部布局的转换
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    if (input_is_uint8_) {
        // 量化模型：假定转换时 baked mean/std（mean_values=0, std_values=255）
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].buf = rgb.data;
        inputs[0].size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    } else {
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].buf = rgb_float.data;
        inputs[0].size = static_cast<uint32_t>(rgb_float.total() * rgb_float.elemSize());
    }

    // 单 ctx 串行化：std::async 双目并发调用时保证线程安全
    std::lock_guard<std::mutex> lock(run_mtx_);

    if (rknn_inputs_set(ctx_, 1, inputs) != RKNN_SUCC) {
        return Status::InferenceFailed;
    }
    if (rknn_run(ctx_, nullptr) != RKNN_SUCC) {
        return Status::InferenceFailed;
    }

    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;  // 统一浮点输出（量化模型自动反量化）
    if (rknn_outputs_get(ctx_, 1, outputs, nullptr) != RKNN_SUCC) {
        return Status::InferenceFailed;
    }

    const float* out_buf = static_cast<const float*>(outputs[0].buf);
    if (out_buf == nullptr) {
        rknn_outputs_release(ctx_, 1, outputs);
        return Status::InferenceFailed;
    }
    out_data.assign(out_buf, out_buf + out_elems_);
    rknn_outputs_release(ctx_, 1, outputs);
    return Status::Success;
}

Status RknnYoloDetector::detect(const cv::Mat& image,
                                std::vector<Detection>& detections) {
    detections.clear();

    float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
    std::vector<float> out_data;
    Status st = runInference(image, out_data, ratio, dw, dh);
    if (st != Status::Success) {
        return st;
    }

    size_t num_preds = 0, num_attrs = 0;
    bool shape_is_bcn = false;
    if (!resolveYoloOutputLayout(out_dims_, out_n_dims_,
                                 num_preds, num_attrs, shape_is_bcn)) {
        std::cerr << "[RknnYoloDetector] Unexpected output layout" << std::endl;
        return Status::InferenceFailed;
    }
    if (num_attrs < 5 || out_data.size() < num_attrs * num_preds) {
        return Status::InferenceFailed;
    }
    const size_t num_classes = num_attrs - 4;

    detections = decodeYoloOutput(out_data.data(), shape_is_bcn, num_preds, num_classes,
                                  ratio, dw, dh, image.cols, image.rows,
                                  config_.conf_threshold, config_.iou_threshold);
    return Status::Success;
}

Status RknnYoloDetector::detectObb(const cv::Mat& image,
                                   std::vector<ObbDetection>& detections) {
    detections.clear();

    if (!config_.is_obb) {
        return Status::UnknownError;  // OBB 解码需 is_obb 配置
    }

    float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
    std::vector<float> out_data;
    Status st = runInference(image, out_data, ratio, dw, dh);
    if (st != Status::Success) {
        return st;
    }

    size_t num_preds = 0, num_attrs = 0;
    bool shape_is_bcn = false;
    if (!resolveYoloOutputLayout(out_dims_, out_n_dims_,
                                 num_preds, num_attrs, shape_is_bcn)) {
        std::cerr << "[RknnYoloDetector] Unexpected output layout" << std::endl;
        return Status::InferenceFailed;
    }
    if (num_attrs < 6 || out_data.size() < num_attrs * num_preds) {
        return Status::InferenceFailed;
    }
    const size_t num_classes = num_attrs - 5;  // 末通道为角度

    detections = decodeYoloObbOutput(out_data.data(), shape_is_bcn, num_preds, num_classes,
                                     ratio, dw, dh, image.cols, image.rows,
                                     config_.conf_threshold, config_.iou_threshold);
    return Status::Success;
}

} // namespace gpnp

#endif // GPNP_WITH_RKNN
