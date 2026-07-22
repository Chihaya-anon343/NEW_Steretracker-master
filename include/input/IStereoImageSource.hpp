#pragma once

/**
 * @file IStereoImageSource.hpp
 * @brief 双目图像源的抽象接口。
 *
 * 所有图像输入源（文件、目录序列、摄像头、视频）均实现此接口。
 * InputProvider 通过此接口解耦图像来源与下游处理管线。
 */

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace gpnp {
namespace input {

class IStereoImageSource {
public:
    virtual ~IStereoImageSource() = default;

    /// 打开图像源。
    /// @param uri  含义由子类决定（文件路径、目录路径、设备路径等）。
    /// @return 成功返回 true。
    virtual bool open(const std::string& uri) = 0;

    /// 获取下一帧双目图像对。
    /// @param[out] left          左图（CV_8UC3 BGR）。
    /// @param[out] right         右图（CV_8UC3 BGR）。
    /// @param[out] timestamp_us  帧时间戳（微秒），若源不提供则为 0。
    /// @return 成功返回 true；到达末尾或无更多帧返回 false。
    virtual bool nextFrame(cv::Mat& left, cv::Mat& right,
                           int64_t& timestamp_us) = 0;

    /// 关闭图像源，释放资源。
    virtual void close() = 0;

    /// 图像源是否已打开。
    virtual bool isOpen() const = 0;

    /// 总帧数。
    /// @return 帧数；-1 表示未知（如摄像头流）。
    virtual int totalFrames() const = 0;

    /// 当前帧序号（从 0 开始）。
    virtual int currentFrame() const = 0;

    // ---- 可选：重置到第一帧 ----
    virtual bool reset() { return false; }
};

} // namespace input
} // namespace gpnp
