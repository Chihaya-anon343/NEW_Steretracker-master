/**
 * @file YoloRoiProvider.cpp
 * @brief 外观模式实现：YOLO 检测 → 立体 ROI 对。
 */

#include "detection/YoloRoiProvider.hpp"
#include "detection/YoloDetector.hpp"

#include <iostream>

namespace gpnp {

YoloRoiProvider::YoloRoiProvider() = default;
YoloRoiProvider::~YoloRoiProvider() = default;

bool YoloRoiProvider::initialize(const YoloConfig& yolo_cfg,
                                  const RoiGenerator::Config& roi_cfg) {
    try {
        detector_ = std::make_unique<YoloDetector>(yolo_cfg);
        roi_gen_  = std::make_unique<RoiGenerator>(roi_cfg);
        std::cout << "[YoloRoiProvider] Ready. Model: " << yolo_cfg.model_path
                  << ", conf=" << yolo_cfg.conf_threshold
                  << ", class=" << yolo_cfg.target_class_id
                  << ", expand=" << roi_cfg.roi_expand_ratio
                  << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[YoloRoiProvider] Init failed: " << e.what() << std::endl;
        return false;
    }
}

bool YoloRoiProvider::isReady() const {
    return detector_ != nullptr && roi_gen_ != nullptr;
}

std::pair<RoiGroup, RoiGroup> YoloRoiProvider::detect(const cv::Mat& left_img,
                                                       const cv::Mat& right_img) {
    if (!isReady()) return {};

    std::vector<Detection> det_left, det_right;
    Status sl = detector_->detect(left_img,  det_left);
    Status sr = detector_->detect(right_img, det_right);

    if (sl != Status::Success || sr != Status::Success ||
        det_left.empty() || det_right.empty()) {
        std::cerr << "[YoloRoiProvider] Detection failed"
                  << " (L status=" << static_cast<int>(sl)
                  << ", R status=" << static_cast<int>(sr) << ")"
                  << std::endl;
        return {};
    }

    auto [lg, rg] = roi_gen_->generateStereoGroup(
        det_left, det_right, left_img.size(), right_img.size());

    if (lg.valid() && rg.valid()) {
        std::cout << "[YoloRoiProvider] ROI=(" << lg.primary.x << "," << lg.primary.y << ","
                  << lg.primary.width << "," << lg.primary.height << ")";
        if (lg.is_dual) {
            std::cout << " + secondary=(" << lg.secondary.x << "," << lg.secondary.y << ","
                      << lg.secondary.width << "," << lg.secondary.height << ")";
        }
        std::cout << std::endl;
    }

    return {lg, rg};
}

RoiGroup YoloRoiProvider::detectMono(const cv::Mat& left_img) {
    if (!isReady()) return {};

    std::vector<Detection> det_left;
    Status sl = detector_->detect(left_img, det_left);

    if (sl != Status::Success || det_left.empty()) {
        std::cerr << "[YoloRoiProvider] Mono detection failed"
                  << " (status=" << static_cast<int>(sl) << ")"
                  << std::endl;
        return {};
    }

    // 单目模式：仅对左图生成 RoiGroup，右图为空
    RoiGroup lg = roi_gen_->generateGroup(det_left, left_img.size());

    if (lg.valid()) {
        std::cout << "[YoloRoiProvider] Mono ROI=("
                  << lg.primary.x << "," << lg.primary.y << ","
                  << lg.primary.width << "," << lg.primary.height << ")";
        if (lg.is_dual) {
            std::cout << " + secondary=(" << lg.secondary.x << "," << lg.secondary.y << ","
                      << lg.secondary.width << "," << lg.secondary.height << ")";
        }
        std::cout << std::endl;
    }

    return lg;
}

} // namespace gpnp