#pragma once

/**
 * @file IYoloBackend.hpp
 * @brief YOLO 推理后端抽象接口（ONNX Runtime / RKNN NPU 共用）。
 *
 * YoloRoiProvider 持有 IYoloBackend，按模型文件后缀路由到具体后端：
 *   - *.onnx → YoloDetector (ONNX Runtime)
 *   - *.rknn → RknnDetector (RK3588 NPU, 需 GPNP_RKNN_ENABLED)
 *
 * 两个后端共享同一套 letterbox 预处理 (YoloPreprocess.hpp)，保证
 * ratio/dw/dh 与检测坐标反变换在不同后端间逐位一致。
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

class IYoloBackend {
public:
    virtual ~IYoloBackend() = default;

    virtual bool isInitialized() const = 0;

    /// 对图像运行检测。成功时将填充 `detections`。
    virtual Status detect(const cv::Mat& image, std::vector<Detection>& detections) = 0;
};

} // namespace gpnp
