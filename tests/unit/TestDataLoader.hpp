#pragma once

/**
 * @file TestDataLoader.hpp
 * @brief 测试数据 (fixtures) 加载工具 — 供 test_extractors / test_integration 共用。
 *
 * 数据来源: tests/data/fixtures/
 *   - 各场景目录: synthetic_tiny / synthetic_bc / synthetic_akaze /
 *                 synthetic_bc_class1 / synthetic_akaze_class1 / synthetic_dual /
 *                 mono_tiny / mono_bc / mono_akaze / mono_bc_class1 /
 *                 mono_akaze_class1 / mono_dual
 *     (双目场景含 left_XXX.png + right_XXX.png; 单目场景仅 left_XXX.png)
 *   - rois.json: 每场景每帧每侧 (left/right) 的 class0 / class1 ROI 表
 *
 * 目录解析优先级:
 *   1) 编译期注入的 GPNP_TEST_FIXTURES_DIR (CMake -DTEST_FIXTURES_DIR=...)
 *   2) 相对路径 "tests/data/fixtures" (从项目根运行)
 * 读取失败时函数返回空值, 由调用方按 "缺失即跳过" 语义处理。
 */

#include <opencv2/core.hpp>   // cv::Rect, cv::FileStorage

#include <cstdio>
#include <string>

namespace gpnp_test {

namespace detail {

inline std::string fixtureDirImpl() {
#ifdef GPNP_TEST_FIXTURES_DIR
    return GPNP_TEST_FIXTURES_DIR;
#else
    return "tests/data/fixtures";
#endif
}

} // namespace detail

/// 返回 fixtures 根目录 (编译期注入或相对项目根)。
inline std::string defaultFixturesDir() {
    return detail::fixtureDirImpl();
}

/// 拼接 rois.json 完整路径。
inline std::string roisJsonPath(const std::string& fixturesDir) {
    return fixturesDir + "/rois.json";
}

/// 拼接场景图片路径: fixturesDir/<scene>/<tag>_<frame:03d>.png
/// @param tag  "left" / "right"
/// @param frame 帧号 (0 起)
inline std::string fixtureImagePath(const std::string& fixturesDir,
                                    const std::string& scene,
                                    const std::string& tag,
                                    int frame) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%03d.png", tag.c_str(), frame);
    return fixturesDir + "/" + scene + "/" + buf;
}

/// 从 rois.json 读取某场景某帧某侧某类别的 ROI。
/// @param jsonPath rois.json 路径
/// @param scene    场景名 (如 "synthetic_bc")
/// @param frame    帧号 (0 起)
/// @param side     "left" / "right"
/// @param cls      "class0" / "class1"
/// @return cv::Rect{x, y, w, h}; 读取失败或条目缺失时返回空 Rect (width<=0 即无效)
inline cv::Rect loadFixtureRoi(const std::string& jsonPath,
                               const std::string& scene, int frame,
                               const std::string& side, const std::string& cls) {
    cv::Rect out{};  // 空 Rect, width<=0 表示无效
    try {
        cv::FileStorage fs(jsonPath, cv::FileStorage::READ);
        if (!fs.isOpened()) return out;

        cv::FileNode node = fs["scenes"][scene]["frames"][frame][side][cls];
        if (node.empty() || node.isNone()) return out;

        int x = static_cast<int>(node["x"]);
        int y = static_cast<int>(node["y"]);
        int w = static_cast<int>(node["width"]);
        int h = static_cast<int>(node["height"]);
        if (w > 0 && h > 0) out = cv::Rect(x, y, w, h);
    } catch (...) {
        // FileStorage 解析失败 (如 JSON 格式不兼容) → 视为缺失
    }
    return out;
}

} // namespace gpnp_test
