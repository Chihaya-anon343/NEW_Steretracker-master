#include "input/DirectoryStereoSource.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <regex>
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

} // anonymous namespace

// ============================================================================
// open(uri)
// ============================================================================

bool DirectoryStereoSource::open(const std::string& uri) {
    return open(uri, "left", "right");
}

// ============================================================================
// open(dir, left_pattern, right_pattern)
// ============================================================================

bool DirectoryStereoSource::open(const std::string& dir,
                                  const std::string& left_pattern,
                                  const std::string& right_pattern) {
    close();

    if (!scanDirectory(dir, left_pattern, right_pattern)) {
        return false;
    }

    if (left_files_.empty()) {
        std::cerr << "[DirectoryStereoSource] 目录中未找到匹配的图像对: "
                  << dir << " (" << left_pattern << "*/" << right_pattern << "*)" << std::endl;
        return false;
    }

    current_index_ = 0;
    std::cout << "[DirectoryStereoSource] 找到 " << left_files_.size()
              << " 对图像，目录: " << dir << std::endl;
    return true;
}

// ============================================================================
// scanDirectory()
// ============================================================================

bool DirectoryStereoSource::scanDirectory(const std::string& dir,
                                           const std::string& left_pattern,
                                           const std::string& right_pattern) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "[DirectoryStereoSource] 目录不存在: " << dir << std::endl;
        return false;
    }

    // 收集所有左图文件
    std::vector<fs::path> left_paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        // 匹配: left_pattern + 任意字符 + 图像扩展名
        if (fname.find(left_pattern) == 0) {
            std::string ext = entry.path().extension().string();
            // 仅支持常见图像格式
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                left_paths.push_back(entry.path());
            }
        }
    }

    if (left_paths.empty()) return true; // 空结果不是错误

    // 按文件名排序
    std::sort(left_paths.begin(), left_paths.end());

    // 为每个左图查找对应的右图
    left_files_.clear();
    right_files_.clear();
    left_files_.reserve(left_paths.size());
    right_files_.reserve(left_paths.size());

    for (const auto& lpath : left_paths) {
        std::string lfname = lpath.filename().string();
        std::string rfname = lfname;
        // 将文件名中的 left_pattern 替换为 right_pattern
        size_t pos = rfname.find(left_pattern);
        if (pos == 0) {
            rfname.replace(0, left_pattern.length(), right_pattern);
        }

        fs::path rpath = lpath.parent_path() / rfname;
        if (fs::exists(rpath) && fs::is_regular_file(rpath)) {
            left_files_.push_back(lpath.string());
            right_files_.push_back(rpath.string());
        } else {
            std::cerr << "[DirectoryStereoSource] 警告: 右图缺失，跳过 "
                      << lfname << " → " << rfname << std::endl;
        }
    }

    return true;
}

// ============================================================================
// nextFrame()
// ============================================================================

bool DirectoryStereoSource::nextFrame(cv::Mat& left, cv::Mat& right,
                                       int64_t& timestamp_us) {
    if (current_index_ < 0 ||
        current_index_ >= static_cast<int>(left_files_.size())) {
        return false;
    }

    left = cv::imread(left_files_[current_index_], cv::IMREAD_COLOR);
    right = cv::imread(right_files_[current_index_], cv::IMREAD_COLOR);

    if (left.empty() || right.empty()) {
        std::cerr << "[DirectoryStereoSource] 读取帧 " << current_index_
                  << " 失败" << std::endl;
        return false;
    }

    timestamp_us = nowUs();
    ++current_index_;
    return true;
}

// ============================================================================
// close() / reset()
// ============================================================================

void DirectoryStereoSource::close() {
    left_files_.clear();
    right_files_.clear();
    current_index_ = -1;
}

bool DirectoryStereoSource::reset() {
    if (left_files_.empty()) return false;
    current_index_ = 0;
    return true;
}

} // namespace input
} // namespace gpnp
