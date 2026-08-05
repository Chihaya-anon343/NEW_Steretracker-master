#pragma once

/**
 * @file CameraSource.hpp
 * @brief 实时摄像头源 (Phase 3)。
 *
 * 通过 OpenCV VideoCapture 打开本机摄像头 (USB / 内置 webcam)，
 * 逐帧返回单目图像；右图 = 左图副本 (与 SequenceSource 单目语义一致)。
 *
 * nextFrame() 内部阻塞在 VideoCapture::read() 上，天然按摄像头帧率
 * 节流；若摄像头帧率高于 target_fps，则额外 sleep 限速。
 */

#include "input/IStereoImageSource.hpp"

#include <opencv2/videoio.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace gpnp {
namespace input {

class CameraSource : public IStereoImageSource {
public:
    CameraSource() = default;
    ~CameraSource() override { close(); }

    /// @param uri 设备描述: 数字索引 "0" 或平台路径 "/dev/video0"。
    bool open(const std::string& uri) override;

    /// @param device      设备索引或路径。
    /// @param target_fps  目标帧率, 0 表示不限制。
    bool open(const std::string& device, double target_fps);

    bool nextFrame(cv::Mat& left, cv::Mat& right,
                   int64_t& timestamp_us) override;

    void close() override;
    bool isOpen() const override { return cap_ && cap_->isOpened(); }

    /// 实时流总帧数未知 (接口契约: -1)。
    int totalFrames() const override { return -1; }
    int currentFrame() const override { return current_frame_; }

private:
    std::unique_ptr<cv::VideoCapture> cap_;
    double target_fps_ = 0.0;
    int64_t last_frame_us_ = 0;
    int current_frame_ = -1;
};

} // namespace input
} // namespace gpnp
