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
#include <fstream>
#include <iterator>
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
namespace detail {

/// 简单 JSON 键值提取: 在 text 中查找 "\"key\":" 后的数字 (支持浮点取整)。
/// 返回是否找到; 找不到时 out 保持不变。
inline bool jsonIntValue(const std::string& text, const std::string& key, int& out) {
    std::string pat = "\"" + key + "\":";
    std::size_t pos = text.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < text.size() &&
           (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' ||
            text[pos] == '\r')) {
        ++pos;
    }
    if (pos >= text.size()) return false;
    bool neg = (text[pos] == '-');
    if (neg) ++pos;
    long long v = 0;
    bool any = false;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        v = v * 10 + (text[pos] - '0');
        ++pos;
        any = true;
    }
    if (!any) return false;
    out = static_cast<int>(neg ? -v : v);
    return true;
}

} // namespace detail

/// @return cv::Rect{x, y, w, h}; 读取失败或条目缺失时返回空 Rect (width<=0 即无效)
///
/// 说明: 不使用 cv::FileStorage 解析 JSON —— OpenCV 的 FileStorage 对标准
/// JSON 的 null/嵌套数组支持不完整, 会导致解析失败。rois.json 结构固定简单,
/// 这里用字符串查找直接提取所需字段值。
inline cv::Rect loadFixtureRoi(const std::string& jsonPath,
                               const std::string& scene, int frame,
                               const std::string& side, const std::string& cls) {
    cv::Rect out{};  // 空 Rect, width<=0 表示无效
    try {
        std::ifstream in(jsonPath);
        if (!in) return out;
        std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        if (text.empty()) return out;

        // 1) 定位 "scenes" 中的 scene 段
        std::string sceneKey = "\"" + scene + "\"";
        std::size_t scenePos = text.find(sceneKey);
        if (scenePos == std::string::npos) return out;

        // 2) 在该段内定位目标 frame (只查该 scene 的 frames 数组)
        //    帧记录形如: { "frame": N, ... }
        int target = -1;
        std::size_t searchFrom = scenePos;
        while (true) {
            std::size_t fpos = text.find("\"frame\":", searchFrom);
            if (fpos == std::string::npos) return out;
            int fn = -1;
            if (!detail::jsonIntValue(text.substr(fpos, 40), "frame", fn))
                return out;
            if (fn == frame) { target = static_cast<int>(fpos); break; }
            searchFrom = fpos + 9;  // 跳到下一个 "frame":
        }
        if (target < 0) return out;

        // 3) 在该帧记录段内定位 side/cls
        //    "side": { "class0": {...}, "class1": null }
        std::string sideKey = "\"" + side + "\"";
        std::size_t sidePos = text.find(sideKey, target);
        if (sidePos == std::string::npos) return out;
        std::string clsKey = "\"" + cls + "\"";
        std::size_t clsPos = text.find(clsKey, sidePos);
        if (clsPos == std::string::npos) return out;

        // 3.5) 守卫: 目标 cls 值必须是对象 "{...}", 而非 null/字符串
        //       (如 "class1": null)。否则会误读到下一个对象的字段。
        std::size_t colon = text.find(':', clsPos);
        if (colon == std::string::npos) return out;
        std::size_t v = colon + 1;
        while (v < text.size() &&
               (text[v] == ' ' || text[v] == '\t' || text[v] == '\n' ||
                text[v] == '\r')) {
            ++v;
        }
        if (v >= text.size() || text[v] != '{') return out;

        // 4) 提取 x/y/width/height; 用段起点限制查找范围
        std::string seg = text.substr(clsPos, 512);
        int x = 0, y = 0, w = 0, h = 0;
        if (!detail::jsonIntValue(seg, "x", x))     return out;
        if (!detail::jsonIntValue(seg, "y", y))     return out;
        if (!detail::jsonIntValue(seg, "width", w)) return out;
        if (!detail::jsonIntValue(seg, "height", h)) return out;
        if (w > 0 && h > 0) out = cv::Rect(x, y, w, h);
    } catch (...) {
        // 读取/解析异常 → 视为缺失
    }
    return out;
}

} // namespace gpnp_test
