#pragma once

/**
 * @file SequenceSource.hpp
 * @brief 单目图像序列源。
 *
 * 扫描目录中按文件名前缀匹配的图像文件（如 frame_0000.jpg），
 * 排序后逐帧返回。每帧作为 left 图像，right = left.clone()。
 */

#include "input/IStereoImageSource.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gpnp {
namespace input {

class SequenceSource : public IStereoImageSource {
public:
    SequenceSource() = default;
    ~SequenceSource() override { close(); }

    /// @param uri  目录路径。
    bool open(const std::string& uri) override;

    /// @param dir      目录路径。
    /// @param pattern  文件名前缀，如 "frame"。
    bool open(const std::string& dir, const std::string& pattern);

    bool nextFrame(cv::Mat& left, cv::Mat& right,
                   int64_t& timestamp_us) override;

    void close() override;
    bool isOpen() const override { return !files_.empty(); }

    int totalFrames() const override { return static_cast<int>(files_.size()); }
    int currentFrame() const override { return current_index_; }

    bool reset() override;

private:
    bool scanDirectory(const std::string& dir, const std::string& pattern);

    std::vector<std::string> files_;
    int current_index_ = -1;
};

} // namespace input
} // namespace gpnp
