#include "input/SequenceSource.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace gpnp {
namespace input {

namespace fs = std::filesystem;

namespace {

int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// 支持的图像文件扩展名
bool isImageFile(const std::string& ext) {
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".tiff" || ext == ".tif";
}

} // anonymous namespace

// ============================================================================
// open(uri) — 默认模式 "frame"
// ============================================================================

bool SequenceSource::open(const std::string& uri) {
    return open(uri, "frame");
}

// ============================================================================
// open(dir, pattern)
// ============================================================================

bool SequenceSource::open(const std::string& dir, const std::string& pattern) {
    close();

    if (!scanDirectory(dir, pattern)) {
        return false;
    }

    if (files_.empty()) {
        std::cerr << "[SequenceSource] 目录中未找到匹配文件: "
                  << dir << " (" << pattern << "_*)" << std::endl;
        return false;
    }

    current_index_ = 0;
    std::cout << "[SequenceSource] 找到 " << files_.size()
              << " 帧, 目录: " << dir << " (模式: " << pattern << "_*)" << std::endl;
    return true;
}

// ============================================================================
// scanDirectory()
// ============================================================================

bool SequenceSource::scanDirectory(const std::string& dir,
                                    const std::string& pattern) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "[SequenceSource] 目录不存在: " << dir << std::endl;
        return false;
    }

    files_.clear();

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.find(pattern) == 0 && isImageFile(entry.path().extension().string())) {
            files_.push_back(entry.path().string());
        }
    }

    // 按文件名排序（自然序：frame_0000 < frame_0001 < ...）
    std::sort(files_.begin(), files_.end());

    return true;
}

// ============================================================================
// nextFrame()
// ============================================================================

bool SequenceSource::nextFrame(cv::Mat& left, cv::Mat& right,
                                int64_t& timestamp_us) {
    if (current_index_ < 0 ||
        current_index_ >= static_cast<int>(files_.size())) {
        return false;
    }

    left = cv::imread(files_[current_index_], cv::IMREAD_COLOR);
    if (left.empty()) {
        std::cerr << "[SequenceSource] 读取帧 " << current_index_
                  << " 失败: " << files_[current_index_] << std::endl;
        return false;
    }

    // 单目序列：右图 = 左图副本
    right = left.clone();
    timestamp_us = nowUs();
    ++current_index_;
    return true;
}

// ============================================================================
// close() / reset()
// ============================================================================

void SequenceSource::close() {
    files_.clear();
    current_index_ = -1;
}

bool SequenceSource::reset() {
    if (files_.empty()) return false;
    current_index_ = 0;
    return true;
}

} // namespace input
} // namespace gpnp
