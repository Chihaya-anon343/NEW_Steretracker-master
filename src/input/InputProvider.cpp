#include "input/InputProvider.hpp"

#include "input/FileStereoSource.hpp"
#include "input/DirectoryStereoSource.hpp"
#include "input/SequenceSource.hpp"
#include "input/CameraSource.hpp"
#include "input/IStereoImageSource.hpp"

#include <iostream>

namespace gpnp {
namespace input {

// ============================================================================
// 构造 / 析构
// ============================================================================

InputProvider::InputProvider() = default;
InputProvider::~InputProvider() {
    shutdown();  // 停止采集线程 (若已启动), 避免线程访问已析构成员
}

// ============================================================================
// initialize()
// ============================================================================

bool InputProvider::initialize(const InputSystemConfig& config) {
    config_ = config;
    current_frame_ = 0;

    // 防御: 若此前已初始化并启动了采集线程, 先停止 (重复 initialize 安全)
    if (capture_thread_.joinable()) shutdown();

    if (!createImageSource()) {
        std::cerr << "[InputProvider] 图像源创建失败" << std::endl;
        return false;
    }

    // Phase 2: createImuSource() / createAltimeterSource() + TimeSyncUnit
    // 当前 Phase 1 仅图像源

    // ---- Phase 3.1: 线程化采集 ----
    // Camera 类型自动启用 (实时流必须解耦采集与处理); 其他源可由配置显式启用
    threaded_capture_ = (config_.image.type == ImageSourceType::Camera)
                        || config_.use_threaded_capture;
    if (threaded_capture_) {
        int cap = config_.ring_capacity > 0 ? config_.ring_capacity : 4;
        frame_ring_ = std::make_unique<RingBuffer<SensorPacket>>(
            static_cast<size_t>(cap));
        captured_frames_ = 0;
        consumed_frames_ = 0;
        capture_failed_ = false;
        capture_running_ = true;
        capture_thread_ = std::thread(&InputProvider::captureLoop, this);
        std::cout << "[InputProvider] 线程化采集已启动 (缓冲 " << cap << " 帧, "
                  << "take-latest 策略)" << std::endl;
    }

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

    case ImageSourceType::Camera: {
        auto src = std::make_unique<CameraSource>();
        // 多设备预留: "0;1" 取第一个; 后续可扩展为 USB 双目
        std::string dev = img_cfg.camera_devices;
        if (dev.empty()) dev = "0";
        auto sep = dev.find(';');
        if (sep != std::string::npos) dev = dev.substr(0, sep);
        if (!src->open(dev, img_cfg.target_fps)) {
            return false;
        }
        image_source_ = std::move(src);
        return true;
    }
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
// 线程化采集 (Phase 3.1)
// ============================================================================

void InputProvider::captureLoop() {
    while (capture_running_.load()) {
        // 1. 从图像源获取下一帧 (Camera: 阻塞在 cap_->read/grab 上, 按相机帧率节拍)
        cv::Mat left, right;
        int64_t ts_us = 0;
        if (!image_source_->nextFrame(left, right, ts_us)) {
            capture_failed_ = true;   // 摄像头断开/读取失败 → 通知消费者退出
            break;
        }

        // 2. 组装数据包并推入缓冲 (满时丢弃最旧帧并计数)
        SensorPacket pkt;
        pkt.timestamp_us = ts_us;
        pkt.left_image = left;
        pkt.right_image = right;
        pkt.imu.reset();
        pkt.height.reset();
        pkt.valid = true;

        frame_ring_->push(std::move(pkt));
        ++captured_frames_;
        queue_cv_.notify_one();
    }
    queue_cv_.notify_all();   // 唤醒可能阻塞在 wait 中的消费者
}

void InputProvider::shutdown() {
    capture_running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    queue_cv_.notify_all();
}

InputProvider::InputStats InputProvider::stats() const {
    InputStats s;
    s.captured = captured_frames_.load();
    s.consumed = consumed_frames_.load();
    s.dropped = s.captured - s.consumed
                - (frame_ring_ ? static_cast<int64_t>(frame_ring_->size()) : 0);
    if (s.dropped < 0) s.dropped = 0;
    return s;
}

// ============================================================================
// getNextPacket()
// ============================================================================

bool InputProvider::getNextPacket(SensorPacket& packet, int timeout_ms) {
    if (!image_source_ || !image_source_->isOpen()) {
        return false;
    }

    if (threaded_capture_) {
        // ---- 线程化: 从缓冲取最新帧 (take-latest) ----
        // 处理快 → 缓冲基本为空, 阻塞等下一帧 (等效同步模式);
        // 处理慢 → 缓冲积压, 取最新帧并丢弃旧帧 (延迟有界, 丢帧可统计)
        // 有限超时: 让主循环能周期性醒来检查外部停止标志 (如 Ctrl+C)
        std::unique_lock lock(queue_cv_mtx_);
        auto ready = [&] {
            return !frame_ring_->empty()
                || !capture_running_.load()
                || capture_failed_.load();
        };
        if (timeout_ms >= 0) {
            queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
        } else {
            queue_cv_.wait(lock, ready);
        }
        if (frame_ring_->empty()) return false;   // 采集已停止/失败/超时

        size_t discarded = 0;
        if (!frame_ring_->takeLatest(packet, discarded)) return false;
        lock.unlock();

        ++consumed_frames_;
        ++current_frame_;
        return true;
    }

    // ---- 同步模式 (File/Directory/Sequence): 逐帧直读 ----
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

bool InputProvider::isCaptureStopped() const {
    // 同步模式: 无采集线程, getNextPacket 返回 false 即源结束
    if (!threaded_capture_) return true;
    return !capture_running_.load() || capture_failed_.load();
}

int InputProvider::totalFrames() const {
    return image_source_ ? image_source_->totalFrames() : -1;
}

int InputProvider::currentFrame() const {
    return image_source_ ? image_source_->currentFrame() : -1;
}

bool InputProvider::reset() {
    current_frame_ = 0;
    if (frame_ring_) frame_ring_->clear();
    return image_source_ ? image_source_->reset() : false;
}

} // namespace input
} // namespace gpnp
