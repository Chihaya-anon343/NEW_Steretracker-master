#pragma once

/**
 * @file YoloRoiProvider.hpp
 * @brief 外观类：YOLO 检测 → 双目 ROI 对。
 *
 * 将 YoloDetector + RoiGenerator 封装在单一接口之后。
 * main.cpp 仅依赖此头文件——无需了解 ONNX 推理或 ROI 生成的内部细节。
 */

#include "common/Types.hpp"
#include "detection/RoiGenerator.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <utility>

namespace gpnp {

class YoloDetector;

class YoloRoiProvider {
public:
    YoloRoiProvider();
    ~YoloRoiProvider();

    YoloRoiProvider(const YoloRoiProvider&) = delete;
    YoloRoiProvider& operator=(const YoloRoiProvider&) = delete;

    /// 初始化检测器和 ROI 生成器。失败时返回 false。
    bool initialize(const YoloConfig& yolo_cfg, const RoiGenerator::Config& roi_cfg);

    /// 若 YOLO 已准备好推理则返回 true。
    bool isReady() const;

    /// 对两张图像运行 YOLO，返回双目 RoiGroup 对。
    /// 每个 RoiGroup 最多可包含 2 个 ROI（主=class 0，次=class 1）。
    /// 若任一侧检测失败则返回 {invalid, invalid}。
    std::pair<RoiGroup, RoiGroup> detect(const cv::Mat& left_img,
                                          const cv::Mat& right_img);

    /// 单目模式：仅对左图运行 YOLO，返回单个 RoiGroup。
    /// 右图不做任何处理。
    RoiGroup detectMono(const cv::Mat& left_img);

private:
    std::unique_ptr<YoloDetector> detector_;
    std::unique_ptr<RoiGenerator> roi_gen_;
};

} // namespace gpnp