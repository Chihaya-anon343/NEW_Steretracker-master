/**
 * @file YoloRoiProvider.cpp
 * @brief 外观模式实现：YOLO 检测 → 立体 ROI 对。
 */

#include "detection/YoloRoiProvider.hpp"
#include "detection/YoloDetector.hpp"
#include "common/LogConfig.hpp"

#include <iostream>

namespace gpnp {

YoloRoiProvider::YoloRoiProvider() = default;
YoloRoiProvider::~YoloRoiProvider() = default;

bool YoloRoiProvider::initialize(const YoloConfig& yolo_cfg,
                                  const RoiGenerator::Config& roi_cfg) {
    try {
        detector_ = std::make_unique<YoloDetector>(yolo_cfg);
        roi_gen_  = std::make_unique<RoiGenerator>(roi_cfg);
        if (g_verbose_console)
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

void YoloRoiProvider::setCloseRangeConfig(const RoiGenerator::CloseRangeConfig& cfg) {
    if (roi_gen_) roi_gen_->setCloseRangeConfig(cfg);
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

    // 双侧都有检测 → 生成立体配对；仅一侧检测 → 各自独立生成，另一侧返回 invalid
    RoiGroup lg, rg;
    bool left_ok  = (sl == Status::Success && !det_left.empty());
    bool right_ok = (sr == Status::Success && !det_right.empty());

    if (left_ok && right_ok) {
        auto [lg_pair, rg_pair] = roi_gen_->generateStereoGroup(
            det_left, det_right, left_img.size(), right_img.size());
        lg = lg_pair;
        rg = rg_pair;
    } else if (left_ok) {
        lg = roi_gen_->generateGroup(det_left, left_img.size());
    } else if (right_ok) {
        rg = roi_gen_->generateGroup(det_right, right_img.size());
    }

    if (g_verbose_console) {
        if (lg.valid() && rg.valid()) {
            std::cout << "[YoloRoiProvider] ROI=(" << lg.primary.x << "," << lg.primary.y << ","
                      << lg.primary.width << "," << lg.primary.height << ")";
            if (lg.is_dual) {
                std::cout << " + secondary=(" << lg.secondary.x << "," << lg.secondary.y << ","
                          << lg.secondary.width << "," << lg.secondary.height << ")";
            }
            std::cout << std::endl;
        } else if (lg.valid()) {
            std::cout << "[YoloRoiProvider] Left-only ROI=(" << lg.primary.x << ","
                      << lg.primary.y << "," << lg.primary.width << "," << lg.primary.height << ")"
                      << std::endl;
        } else if (rg.valid()) {
            std::cout << "[YoloRoiProvider] Right-only ROI=(" << rg.primary.x << ","
                      << rg.primary.y << "," << rg.primary.width << "," << rg.primary.height << ")"
                      << std::endl;
        }
    }

    return {lg, rg};
}

RoiGroup YoloRoiProvider::detectMono(const cv::Mat& left_img) {
    if (!isReady()) return {};

    std::vector<Detection> det_left;
    Status sl = detector_->detect(left_img, det_left);

    if (sl != Status::Success || det_left.empty()) {
        return {};
    }

    // 单目模式：仅对左图生成 RoiGroup，右图为空
    RoiGroup lg = roi_gen_->generateGroup(det_left, left_img.size());

    if (lg.valid()) {
        if (g_verbose_console) {
            std::cout << "[YoloRoiProvider] Mono ROI=("
                      << lg.primary.x << "," << lg.primary.y << ","
                      << lg.primary.width << "," << lg.primary.height << ")";
            if (lg.is_dual) {
                std::cout << " + secondary=(" << lg.secondary.x << "," << lg.secondary.y << ","
                          << lg.secondary.width << "," << lg.secondary.height << ")";
            }
            std::cout << std::endl;
        }
    }

    return lg;
}

} // namespace gpnp