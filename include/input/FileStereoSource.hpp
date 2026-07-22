#pragma once

/**
 * @file FileStereoSource.hpp
 * @brief 静态文件双目图像源 —— 当前默认行为。
 *
 * 从配置中指定的左右图像路径加载一对图像。
 * nextFrame() 每次返回同一帧（兼容 warm-start 多次调用模式）。
 */

#include "input/IStereoImageSource.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace gpnp {
namespace input {

class FileStereoSource : public IStereoImageSource {
public:
    FileStereoSource() = default;
    ~FileStereoSource() override { close(); }

    /// @param uri  格式: "left_path;right_path"（分号分隔）。
    bool open(const std::string& uri) override;

    bool nextFrame(cv::Mat& left, cv::Mat& right,
                   int64_t& timestamp_us) override;

    void close() override;
    bool isOpen() const override { return loaded_; }

    int totalFrames() const override { return loaded_ ? 1 : 0; }
    int currentFrame() const override { return frame_returned_ ? 0 : -1; }

    bool reset() override {
        frame_returned_ = false;
        return loaded_;
    }

    /// 便捷接口：分别指定左右路径。
    bool open(const std::string& left_path, const std::string& right_path);

private:
    cv::Mat left_img_;
    cv::Mat right_img_;
    bool loaded_ = false;
    bool frame_returned_ = false;
};

} // namespace input
} // namespace gpnp
