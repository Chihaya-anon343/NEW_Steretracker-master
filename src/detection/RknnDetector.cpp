/**
 * @file RknnDetector.cpp
 * @brief RKNN NPU YOLO 检测器实现。
 *
 * 输出契约 (yolov8n_p2.rknn, 上板探针实测):
 *   12 输出 = 4 尺度 (160/80/40/20, stride 4/8/16/32) × 3 张量:
 *     [1,64,H,W] DFL raw (box i 占通道 [i*16, i*16+16))
 *     [1,nc,H,W] 类别分数 (已 sigmoid)
 *     [1, 1,H,W] max 概览 (解码不需要)
 *   每三输出一组, 组内按通道数识别 (64→DFL, 1→max, 其余→cls)。
 *
 * 输入: 模型量化输入 INT8 (zp=-128, scale=1/255), 以 UINT8 NHWC 喂入
 * (pass_through=0, runtime 自动量化), 与探针验证路径一致。
 *
 * GPNP_RKNN_ENABLED 未定义时编译为存根, 仅在构造时抛异常。
 */

#include "detection/RknnDetector.hpp"
#include "detection/YoloDecode.hpp"
#include "detection/YoloPreprocess.hpp"

#ifdef GPNP_RKNN_ENABLED
#include "rknn_api.h"
#endif

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpnp {

#ifdef GPNP_RKNN_ENABLED

namespace {

constexpr int kMaxScales = 8;

} // namespace

struct RknnDetector::RknnState {
    rknn_context ctx = 0;
    std::vector<uint8_t> model_data;   // rknn_init 后保持存活至析构

    int input_h = 640;
    int input_w = 640;
    uint32_t n_output = 0;
    int num_classes = 0;
    int num_scales = 0;

    int dfl_idx[kMaxScales] = {0};
    int cls_idx[kMaxScales] = {0};
    int map_h[kMaxScales] = {0};
    int map_w[kMaxScales] = {0};
    int strides[kMaxScales] = {0};

    std::mutex mtx;                    // rknn_run 非线程安全, 串行化 detect
};

RknnDetector::RknnDetector(const YoloConfig& cfg)
    : st_(std::make_unique<RknnState>())
    , config_(cfg)
{
    // ---- 模型文件检查 (与 YoloDetector 同风格) ----
    {
        std::ifstream f(config_.model_path, std::ios::binary | std::ios::ate);
        if (!f.good()) {
            throw std::runtime_error(
                "RKNN model file not found or unreadable: " + config_.model_path);
        }
        auto size = f.tellg();
        if (size <= 0) {
            throw std::runtime_error(
                "RKNN model file is empty: " + config_.model_path);
        }
        f.seekg(0);
        st_->model_data.resize(static_cast<size_t>(size));
        if (!f.read(reinterpret_cast<char*>(st_->model_data.data()), size)) {
            throw std::runtime_error("RKNN model read failed: " + config_.model_path);
        }
        std::cerr << "[RknnDetector] Model file: " << config_.model_path
                  << " (" << size << " bytes)" << std::endl;
    }

    if (rknn_init(&st_->ctx, st_->model_data.data(),
                  static_cast<uint32_t>(st_->model_data.size()), 0, nullptr) != RKNN_SUCC) {
        throw std::runtime_error("rknn_init failed: " + config_.model_path);
    }

    rknn_sdk_version ver;
    std::memset(&ver, 0, sizeof(ver));
    if (rknn_query(st_->ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC) {
        std::cerr << "[RknnDetector] SDK api=" << ver.api_version
                  << " drv=" << ver.drv_version << std::endl;
    }

    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    if (rknn_query(st_->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC) {
        rknn_destroy(st_->ctx);
        st_->ctx = 0;
        throw std::runtime_error("rknn_query IN_OUT_NUM failed");
    }
    st_->n_output = io.n_output;
    std::cerr << "[RknnDetector] n_input=" << io.n_input
              << " n_output=" << io.n_output << std::endl;

    // ---- 输入尺寸: dims 按 fmt 解释 (NHWC [1,H,W,C] / NCHW [1,C,H,W]) ----
    {
        rknn_tensor_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.index = 0;
        if (rknn_query(st_->ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC ||
            attr.n_dims < 4) {
            rknn_destroy(st_->ctx);
            st_->ctx = 0;
            throw std::runtime_error("rknn_query INPUT_ATTR failed");
        }
        if (attr.fmt == RKNN_TENSOR_NHWC) {
            st_->input_h = static_cast<int>(attr.dims[1]);
            st_->input_w = static_cast<int>(attr.dims[2]);
        } else {
            st_->input_h = static_cast<int>(attr.dims[2]);
            st_->input_w = static_cast<int>(attr.dims[3]);
        }
        std::cerr << "[RknnDetector] Input " << st_->input_w << "x" << st_->input_h
                  << " type=" << get_type_string(attr.type)
                  << " zp=" << attr.zp << " scale=" << attr.scale << std::endl;
    }

    // ---- 输出布局验证: 每 3 输出一组 (64ch DFL + nc cls + 1ch max) ----
    if (io.n_output == 0 || io.n_output % 3 != 0 ||
        static_cast<int>(io.n_output) / 3 > kMaxScales) {
        rknn_destroy(st_->ctx);
        st_->ctx = 0;
        throw std::runtime_error(
            "RKNN output layout mismatch: n_output=" + std::to_string(io.n_output) +
            " (expected 3 per scale: DFL/cls/max)");
    }

    const int num_groups = static_cast<int>(io.n_output) / 3;
    for (int g = 0; g < num_groups; ++g) {
        int dfl_out = -1, cls_out = -1, num_cls = -1;
        for (int k = 0; k < 3; ++k) {
            rknn_tensor_attr attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.index = static_cast<uint32_t>(3 * g + k);
            if (rknn_query(st_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC ||
                attr.n_dims < 4) {
                rknn_destroy(st_->ctx);
                st_->ctx = 0;
                throw std::runtime_error("rknn_query OUTPUT_ATTR failed (index " +
                                         std::to_string(attr.index) + ")");
            }
            // 组内按通道数识别 (dims 为 [1,C,H,W]):
            // 64 → DFL; 1 → max 概览 (解码不用); 其余 → cls
            const int C = static_cast<int>(attr.dims[1]);
            if (C == 64) {
                dfl_out = 3 * g + k;
                st_->map_h[st_->num_scales] = static_cast<int>(attr.dims[2]);
                st_->map_w[st_->num_scales] = static_cast<int>(attr.dims[3]);
                st_->strides[st_->num_scales] =
                    st_->input_w / std::max(1, static_cast<int>(attr.dims[3]));
            } else if (C == 1) {
                // max 概览图
            } else {
                cls_out = 3 * g + k;
                num_cls = C;
            }
        }
        if (dfl_out < 0 || cls_out < 0) {
            rknn_destroy(st_->ctx);
            st_->ctx = 0;
            throw std::runtime_error(
                "RKNN output group " + std::to_string(g) +
                " layout mismatch (expected 64ch DFL + nc cls + 1ch max)");
        }
        st_->dfl_idx[st_->num_scales] = dfl_out;
        st_->cls_idx[st_->num_scales] = cls_out;
        st_->num_classes = num_cls;
        ++st_->num_scales;
    }

    std::cerr << "[RknnDetector] Split-head: " << st_->num_scales << " scales, "
              << st_->num_classes << " classes";
    for (int s = 0; s < st_->num_scales; ++s) {
        std::cerr << " [" << st_->map_w[s] << "x" << st_->map_h[s]
                  << "/s" << st_->strides[s] << "]";
    }
    std::cerr << std::endl;

    if (st_->num_classes < 1) {
        rknn_destroy(st_->ctx);
        st_->ctx = 0;
        throw std::runtime_error("RKNN cls channels not found");
    }
}

RknnDetector::~RknnDetector() {
    if (st_ && st_->ctx) {
        rknn_destroy(st_->ctx);
        st_->ctx = 0;
    }
}

bool RknnDetector::isInitialized() const {
    return st_ && st_->ctx != 0;
}

Status RknnDetector::detect(const cv::Mat& image,
                            std::vector<Detection>& detections) {
    detections.clear();

    if (!isInitialized()) {
        return Status::ModelLoadFailed;
    }
    if (image.empty()) {
        return Status::EmptyInput;
    }

    std::lock_guard<std::mutex> lock(st_->mtx);

    try {
        cv::Mat bgr = yolo::ensureBGR(image);

        float ratio = 1.0f, dw = 0.0f, dh = 0.0f;
        cv::Mat letterboxed;
        yolo::letterbox(bgr, letterboxed,
                        cv::Size(st_->input_w, st_->input_h), ratio, dw, dh);

        cv::Mat rgb;
        cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

        rknn_input inp;
        std::memset(&inp, 0, sizeof(inp));
        inp.index = 0;
        inp.buf = rgb.data;
        inp.size = rgb.total() * rgb.elemSize();
        inp.pass_through = 0;
        inp.type = RKNN_TENSOR_UINT8;
        inp.fmt = RKNN_TENSOR_NHWC;
        if (rknn_inputs_set(st_->ctx, 1, &inp) != RKNN_SUCC) {
            return Status::InferenceFailed;
        }
        if (rknn_run(st_->ctx, nullptr) != RKNN_SUCC) {
            return Status::InferenceFailed;
        }

        // 全量取回 (runtime 要求请求所有输出, 否则报 index out of range)
        const int n = static_cast<int>(st_->n_output);
        std::vector<rknn_output> outs(n);
        for (int i = 0; i < n; ++i) {
            std::memset(&outs[i], 0, sizeof(outs[i]));
            outs[i].want_float = 1;
            outs[i].index = static_cast<uint32_t>(i);
        }
        if (rknn_outputs_get(st_->ctx, st_->n_output, outs.data(), nullptr) != RKNN_SUCC) {
            return Status::InferenceFailed;
        }

        const float* dfl_maps[kMaxScales] = {nullptr};
        const float* cls_maps[kMaxScales] = {nullptr};
        for (int s = 0; s < st_->num_scales; ++s) {
            dfl_maps[s] = static_cast<const float*>(outs[st_->dfl_idx[s]].buf);
            cls_maps[s] = static_cast<const float*>(outs[st_->cls_idx[s]].buf);
        }

        detections = decodeYoloRknnSplit(
            dfl_maps, cls_maps, st_->map_h, st_->map_w, st_->strides,
            st_->num_scales, st_->num_classes,
            ratio, dw, dh, bgr.cols, bgr.rows,
            config_.conf_threshold, config_.iou_threshold);

        rknn_outputs_release(st_->ctx, st_->n_output, outs.data());
        return Status::Success;

    } catch (const std::exception&) {
        return Status::UnknownError;
    }
}

#else  // !GPNP_RKNN_ENABLED

struct RknnDetector::RknnState {};

RknnDetector::RknnDetector(const YoloConfig& cfg)
    : st_(nullptr)
    , config_(cfg)
{
    throw std::runtime_error(
        "RknnDetector: RKNN backend not compiled (GPNP_RKNN_ENABLED undefined, "
        "librknnrt not found at configure time): " + cfg.model_path);
}

RknnDetector::~RknnDetector() = default;

bool RknnDetector::isInitialized() const {
    return false;
}

Status RknnDetector::detect(const cv::Mat&, std::vector<Detection>&) {
    return Status::ModelLoadFailed;
}

#endif // GPNP_RKNN_ENABLED

} // namespace gpnp
