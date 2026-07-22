#include "input/FileStereoSource.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

namespace gpnp {
namespace input {

// ============================================================================
// 辅助：按分隔符切分字符串
// ============================================================================
namespace {

std::vector<std::string> splitPath(const std::string& uri, char delim = ';') {
    std::vector<std::string> parts;
    std::istringstream iss(uri);
    std::string token;
    while (std::getline(iss, token, delim)) {
        if (!token.empty()) {
            parts.push_back(token);
        }
    }
    return parts;
}

int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // anonymous namespace

// ============================================================================
// open(uri) — 分号分隔的路径对
// ============================================================================

bool FileStereoSource::open(const std::string& uri) {
    close();

    auto parts = splitPath(uri, ';');
    if (parts.size() < 2) {
        std::cerr << "[FileStereoSource] URI 格式错误，期望 'left;right'，收到: "
                  << uri << std::endl;
        return false;
    }
    return open(parts[0], parts[1]);
}

// ============================================================================
// open(left, right) — 便捷接口
// ============================================================================

bool FileStereoSource::open(const std::string& left_path,
                             const std::string& right_path) {
    close();

    left_img_ = cv::imread(left_path, cv::IMREAD_COLOR);
    if (left_img_.empty()) {
        std::cerr << "[FileStereoSource] 无法读取左图: " << left_path << std::endl;
        return false;
    }

    right_img_ = cv::imread(right_path, cv::IMREAD_COLOR);
    if (right_img_.empty()) {
        std::cerr << "[FileStereoSource] 无法读取右图: " << right_path << std::endl;
        left_img_.release();
        return false;
    }

    loaded_ = true;
    frame_returned_ = false;
    std::cout << "[FileStereoSource] 已加载: " << left_path << " / "
              << right_path << std::endl;
    return true;
}

// ============================================================================
// nextFrame() — 每次返回同一帧（兼容 warm-start 多次调用）
// ============================================================================

bool FileStereoSource::nextFrame(cv::Mat& left, cv::Mat& right,
                                  int64_t& timestamp_us) {
    if (!loaded_) return false;

    // FileStereoSource 始终返回同一帧 ——
    // 这使得现有 warm-start 循环（同一帧处理两次）无需修改。
    left_img_.copyTo(left);
    right_img_.copyTo(right);
    timestamp_us = nowUs();
    frame_returned_ = true;
    return true;
}

// ============================================================================
// close()
// ============================================================================

void FileStereoSource::close() {
    left_img_.release();
    right_img_.release();
    loaded_ = false;
    frame_returned_ = false;
}

} // namespace input
} // namespace gpnp
