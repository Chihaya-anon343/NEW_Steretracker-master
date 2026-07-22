#include "input/InputProvider.hpp"

#include "input/FileStereoSource.hpp"
#include "input/DirectoryStereoSource.hpp"
#include "input/SequenceSource.hpp"
#include "input/IStereoImageSource.hpp"

#include <iostream>

namespace gpnp {
namespace input {

// ============================================================================
// 构造 / 析构
// ============================================================================

InputProvider::InputProvider() = default;
InputProvider::~InputProvider() {
    // Phase 2: delete static_cast<IImuSource*>(imu_source_) etc.
}

// ============================================================================
// initialize()
// ============================================================================

bool InputProvider::initialize(const InputSystemConfig& config) {
    config_ = config;
    current_frame_ = 0;

    if (!createImageSource()) {
        std::cerr << "[InputProvider] 图像源创建失败" << std::endl;
        return false;
    }

    // Phase 2: createImuSource() / createAltimeterSource() + TimeSyncUnit
    // 当前 Phase 1 仅图像源

    std::cout << "[InputProvider] 初始化完成, "
              << "总帧数: " << totalFrames() << std::endl;
    return true;
}

// ============================================================================
// createImageSource()
// ============================================================================

bool InputProvider::createImageSource() {
    const auto& img_cfg = config_.image;

    switch (img_cfg.type) {
    case ImageSourceType::File: {
        auto src = std::make_unique<FileStereoSource>();
        if (!src->open(img_cfg.left_path, img_cfg.right_path)) {
            return false;
        }
        image_source_ = std::move(src);
        return true;
    }

    case ImageSourceType::Directory: {
        auto src = std::make_unique<DirectoryStereoSource>();
        if (!src->open(img_cfg.directory_path,
                       img_cfg.left_pattern,
                       img_cfg.right_pattern)) {
            return false;
        }
        image_source_ = std::move(src);
        return true;
    }

    case ImageSourceType::Sequence: {
        auto src = std::make_unique<SequenceSource>();
        if (!src->open(img_cfg.directory_path, img_cfg.sequence_pattern)) {
            return false;
        }
        image_source_ = std::move(src);
        return true;
    }

    case ImageSourceType::Camera:
        std::cerr << "[InputProvider] Camera 源尚未实现 (Phase 3)" << std::endl;
        return false;
    }

    std::cerr << "[InputProvider] 未知的图像源类型" << std::endl;
    return false;
}

// ============================================================================
// createImuSource() — Phase 2 占位
// ============================================================================

bool InputProvider::createImuSource() {
    if (!config_.imu.enabled) return true; // 未启用，不是错误
    std::cerr << "[InputProvider] IMU 源尚未实现 (Phase 2)" << std::endl;
    return false;
}

// ============================================================================
// createAltimeterSource() — Phase 2 占位
// ============================================================================

bool InputProvider::createAltimeterSource() {
    if (!config_.altimeter.enabled) return true; // 未启用，不是错误
    std::cerr << "[InputProvider] 高度计源尚未实现 (Phase 2)" << std::endl;
    return false;
}

// ============================================================================
// getNextPacket()
// ============================================================================

bool InputProvider::getNextPacket(SensorPacket& packet) {
    if (!image_source_ || !image_source_->isOpen()) {
        return false;
    }

    // 1. 从图像源获取下一帧
    cv::Mat left, right;
    int64_t ts_us = 0;
    if (!image_source_->nextFrame(left, right, ts_us)) {
        return false;
    }

    // 2. 组装数据包
    packet.timestamp_us = ts_us;
    packet.left_image = left;
    packet.right_image = right;

    // Phase 2: 查询 TimeSyncUnit 进行 IMU 插值 + 高度计融合
    packet.imu.reset();
    packet.height.reset();

    packet.valid = true;
    ++current_frame_;
    return true;
}

// ============================================================================
// 状态查询
// ============================================================================

bool InputProvider::isOpen() const {
    return image_source_ && image_source_->isOpen();
}

int InputProvider::totalFrames() const {
    return image_source_ ? image_source_->totalFrames() : -1;
}

int InputProvider::currentFrame() const {
    return image_source_ ? image_source_->currentFrame() : -1;
}

bool InputProvider::reset() {
    current_frame_ = 0;
    return image_source_ ? image_source_->reset() : false;
}

} // namespace input
} // namespace gpnp
