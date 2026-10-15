#pragma once

/**
 * @file RknnDetector.hpp
 * @brief RKNN NPU YOLO 检测器（RK3588, yolov8n_p2.rknn split-head 导出）。
 *
 * 头文件不含任何 rknn_api 类型（pimpl），无 RKNN SDK 的环境可正常包含。
 * 实现见 src/detection/RknnDetector.cpp：
 *   - GPNP_RKNN_ENABLED 已定义 → 完整实现（链接 librknnrt）
 *   - 未定义 → 存根（构造抛异常，detect 返回 ModelLoadFailed）
 *
 * 线程安全：detect() 内部持互斥锁串行化（rknn_run 非线程安全）。
 * YoloRoiProvider 的双目 std::async 并行推理在 RKNN 后端退化为串行。
 */

#include "common/Types.hpp"
#include "detection/IYoloBackend.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace gpnp {

class RknnDetector : public IYoloBackend {
public:
    /// 模型文件缺失 / rknn_init 失败 / 输出布局不符合 split-head 约定时抛异常
    explicit RknnDetector(const YoloConfig& cfg);
    ~RknnDetector() override;

    RknnDetector(const RknnDetector&) = delete;
    RknnDetector& operator=(const RknnDetector&) = delete;

    bool isInitialized() const override;

    Status detect(const cv::Mat& image, std::vector<Detection>& detections) override;

private:
    struct RknnState;
    std::unique_ptr<RknnState> st_;
    YoloConfig config_;
};

} // namespace gpnp
