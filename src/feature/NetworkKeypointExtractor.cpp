/**
 * @file NetworkKeypointExtractor.cpp
 * @brief 对比方法占位实现 —— 关键点网络接入点。
 *
 * TODO(接入真实网络时替换):
 *   1. initialize(): Ort::Session 加载 cfg_.model_path (参考 YoloDetector 构造)
 *   2. detect():     预处理 ROI → 推理 → 解码关键点 (归一化坐标 × ROI 尺寸)
 *                    按 cfg_.num_keypoints / template_real_*_mm 生成 1:1 的 pts_3d
 */

#include "feature/NetworkKeypointExtractor.hpp"

#include <iostream>

namespace gpnp {

NetworkKeypointExtractor::NetworkKeypointExtractor(const Config& cfg)
    : cfg_(cfg) {}

bool NetworkKeypointExtractor::initialize() {
    if (cfg_.model_path.empty()) {
        std::cerr << "[NetworkKeypoint] 占位模式: 未配置 model_path, "
                     "detect() 将恒返回失败" << std::endl;
        return false;
    }
    // TODO: 加载 ONNX 关键点模型
    std::cerr << "[NetworkKeypoint] 占位模式: 模型加载未实现 ("
              << cfg_.model_path << ")" << std::endl;
    return false;
}

NetworkKeypointExtractor::Result NetworkKeypointExtractor::detect(const cv::Mat& roi_bgr) {
    Result res;
    if (roi_bgr.empty()) {
        res.message = "empty roi";
        return res;
    }
    // TODO: 网络推理 + 关键点解码
    res.message = "network inference not implemented (placeholder)";
    return res;
}

} // namespace gpnp
