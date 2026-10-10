#pragma once

/**
 * @file IYoloDetector.hpp
 * @brief YOLO 检测器抽象接口 + 后端工厂。
 *
 * 后端实现与路由见 src/detection/YoloDetectorFactory.cpp：
 *   - ONNX Runtime（include/detection/YoloDetector.hpp）
 *   - RKNN NPU（include/detection/RknnYoloDetector.hpp，需 GPNP_WITH_RKNN 宏）
 * 工厂按 YoloConfig::model_path 后缀 ".rknn"（大小写不敏感）路由后端，
 * 其余一律走 ONNX Runtime。两后端构造失败均抛 std::runtime_error。
 */

#include "common/Types.hpp"
#include "detection/YoloObbDecode.hpp"

#include <memory>
#include <vector>

namespace gpnp {

class IYoloDetector {
public:
    virtual ~IYoloDetector() = default;

    /// 对图像运行检测。成功时填充 detections。
    virtual Status detect(const cv::Mat& image,
                          std::vector<Detection>& detections) = 0;

    /// 对图像运行 OBB 检测（需 config.is_obb == true）。成功时填充 detections。
    virtual Status detectObb(const cv::Mat& image,
                             std::vector<ObbDetection>& detections) = 0;
};

/// 按配置创建检测器实例（后端路由规则见文件头注释）。
std::unique_ptr<IYoloDetector> createYoloDetector(const YoloConfig& cfg);

} // namespace gpnp
