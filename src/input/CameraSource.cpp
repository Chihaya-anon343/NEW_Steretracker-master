#include "input/CameraSource.hpp"

#include <chrono>
#include <iostream>
#include <thread>

namespace gpnp {
namespace input {

namespace {

int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // anonymous namespace

// ============================================================================
// open()
// ============================================================================

bool CameraSource::open(const std::string& uri) {
    return open(uri, 0.0);
}

bool CameraSource::open(const std::string& device, double target_fps) {
    close();
    target_fps_ = target_fps;

    // "0" → 摄像头索引; "/dev/video0" 等 → 设备路径
    int index = -1;
    try {
        index = std::stoi(device);
    } catch (...) {
        index = -1;
    }

    cap_ = std::make_unique<cv::VideoCapture>();
    bool ok = (index >= 0) ? cap_->open(index) : cap_->open(device);
    if (!ok) {
        std::cerr << "[CameraSource] 无法打开摄像头: " << device << std::endl;
        cap_.reset();
        return false;
    }

    if (target_fps_ > 0) {
        cap_->set(cv::CAP_PROP_FPS, target_fps_);
    }

    // 等几帧让摄像头完成自动曝光/白平衡 (与 scripts/camera_capture.py 一致)
    for (int i = 0; i < 5; ++i) {
        cap_->grab();
    }

    current_frame_ = 0;
    std::cout << "[CameraSource] 摄像头已打开: " << device
              << "  " << cap_->get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap_->get(cv::CAP_PROP_FRAME_HEIGHT)
              << "  " << cap_->get(cv::CAP_PROP_FPS) << " fps" << std::endl;
    return true;
}

// ============================================================================
// nextFrame()
// ============================================================================

bool CameraSource::nextFrame(cv::Mat& left, cv::Mat& right,
                             int64_t& timestamp_us) {
    if (!isOpen()) {
        return false;
    }

    // 帧率限速: 摄像头帧率高于 target_fps 时 sleep 到目标间隔
    if (target_fps_ > 0) {
        const int64_t interval_us = static_cast<int64_t>(1e6 / target_fps_);
        if (last_frame_us_ > 0) {
            int64_t elapsed = nowUs() - last_frame_us_;
            if (elapsed < interval_us) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(interval_us - elapsed));
            }
        }
    }

    cv::Mat frame;
    if (!cap_->read(frame)) {   // 阻塞直到下一帧到达 (天然按摄像头帧率节流)
        std::cerr << "[CameraSource] 读取帧失败" << std::endl;
        return false;
    }

    left = frame;
    right = frame.clone();      // 单目源: 右图 = 左图副本
    timestamp_us = nowUs();
    last_frame_us_ = timestamp_us;
    ++current_frame_;
    return true;
}

// ============================================================================
// close()
// ============================================================================

void CameraSource::close() {
    if (cap_) {
        cap_->release();
        cap_.reset();
    }
    current_frame_ = -1;
}

} // namespace input
} // namespace gpnp
