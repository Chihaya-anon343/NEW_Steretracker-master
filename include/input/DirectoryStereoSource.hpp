#pragma once

/**
 * @file DirectoryStereoSource.hpp
 * @brief 目录图像序列双目源。
 *
 * 扫描目录中的编号图像对（如 left_0001.png / right_0001.png），
 * 按文件名排序后逐帧返回。
 */

#include "input/IStereoImageSource.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gpnp {
namespace input {

class DirectoryStereoSource : public IStereoImageSource {
public:
    DirectoryStereoSource() = default;
    ~DirectoryStereoSource() override { close(); }

    /// @param uri  目录路径。
    bool open(const std::string& uri) override;

    /// 指定左右文件名模式。
    /// @param dir            目录路径。
    /// @param left_pattern   左图文件名前缀，如 "left"。
    /// @param right_pattern  右图文件名前缀，如 "right"。
    bool open(const std::string& dir,
              const std::string& left_pattern,
              const std::string& right_pattern);

    bool nextFrame(cv::Mat& left, cv::Mat& right,
                   int64_t& timestamp_us) override;

    void close() override;
    bool isOpen() const override { return !left_files_.empty(); }

    int totalFrames() const override {
        return static_cast<int>(left_files_.size());
    }
    int currentFrame() const override { return current_index_; }

    bool reset() override;

private:
    /// 扫描目录，收集匹配的图像文件对。
    bool scanDirectory(const std::string& dir,
                       const std::string& left_pattern,
                       const std::string& right_pattern);

    std::vector<std::string> left_files_;
    std::vector<std::string> right_files_;
    int current_index_ = -1;
};

} // namespace input
} // namespace gpnp
