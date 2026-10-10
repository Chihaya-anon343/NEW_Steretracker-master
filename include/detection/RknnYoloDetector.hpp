#pragma once

/**
 * @file RknnYoloDetector.hpp
 * @brief RKNN NPU YOLO 检测器（Rockchip RK3588 等板卡，librknnrt ≥ 2.x）。
 *
 * 仅可被 YoloDetectorFactory.cpp 在 GPNP_WITH_RKNN 宏开启时包含——
 * 本头直接包含 rknn_api.h，未安装该头的环境不能引入。
 *
 * 输入自适应（按模型输入张量属性）：
 *   - UINT8（INT8 量化模型）→ letterbox+RGB 后直喂原始字节，
 *     归一化/量化由 runtime 处理（前提：转换时 baked mean/std）
 *   - FLOAT32 → CPU 上 /255 归一化后喂 float
 * 后处理复用 YoloDecode/YoloObbDecode 纯函数，与 ONNX 后端逐位一致。
 */

#include <rknn_api.h>

#include "detection/IYoloDetector.hpp"

#include <cstdint>
#include <mutex>

namespace gpnp {

class RknnYoloDetector : public IYoloDetector {
public:
    explicit RknnYoloDetector(const YoloConfig& cfg);  // 失败抛 std::runtime_error
    ~RknnYoloDetector() override;

    RknnYoloDetector(const RknnYoloDetector&) = delete;
    RknnYoloDetector& operator=(const RknnYoloDetector&) = delete;

    Status detect(const cv::Mat& image,
                  std::vector<Detection>& detections) override;
    Status detectObb(const cv::Mat& image,
                     std::vector<ObbDetection>& detections) override;

private:
    /// letterbox → 喂数 → rknn_run → 浮点输出（want_float=1）。内部加锁串行。
    Status runInference(const cv::Mat& image,
                        std::vector<float>& out_data,
                        float& ratio, float& dw, float& dh);

    YoloConfig config_;
    rknn_context ctx_{0};
    int input_h_ = 640;
    int input_w_ = 640;
    bool input_is_uint8_ = false;

    uint32_t out_dims_[RKNN_MAX_DIMS] = {};
    uint32_t out_n_dims_ = 0;
    size_t out_elems_ = 0;

    std::mutex run_mtx_;  // 单 rknn_context 串行化：YoloRoiProvider::detect 的 std::async 双目并发下保证安全
};

} // namespace gpnp
