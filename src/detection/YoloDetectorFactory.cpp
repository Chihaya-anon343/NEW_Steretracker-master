/**
 * @file YoloDetectorFactory.cpp
 * @brief createYoloDetector —— 按模型后缀路由推理后端的唯一 #ifdef 分支点。
 */

#include "detection/IYoloDetector.hpp"
#include "detection/YoloDetector.hpp"

#ifdef GPNP_WITH_RKNN
#include "detection/RknnYoloDetector.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

namespace gpnp {

namespace {

bool hasRknnExtension(const std::string& path) {
    if (path.size() < 5) {
        return false;
    }
    std::string ext = path.substr(path.size() - 5);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".rknn";
}

} // namespace

std::unique_ptr<IYoloDetector> createYoloDetector(const YoloConfig& cfg) {
    if (hasRknnExtension(cfg.model_path)) {
#ifdef GPNP_WITH_RKNN
        return std::make_unique<RknnYoloDetector>(cfg);
#else
        throw std::runtime_error(
                "model '" + cfg.model_path + "' requires the RKNN backend, but this "
                "build was configured without RKNN support (librknnrt not found)");
#endif
    }
    return std::make_unique<YoloDetector>(cfg);
}

} // namespace gpnp
