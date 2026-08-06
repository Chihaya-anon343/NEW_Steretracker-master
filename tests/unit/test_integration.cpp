// =============================================================================
// test_integration.cpp — 端到端冒烟集成测试
//
// 目标:
//   1) MonoTracker::process()   单目主流程
//   2) StereoTracker::process() 双目主流程 (含 GPnP / InitialPnP warm-start)
//   3) Dual-ROI 独立路径 (is_dual=true)
//
// 数据依赖:
//   - AKAZE 模板目录 data/NewMuBan(reordered)/
//   - BinaryCorner / TinyTarget 模板目录 (缺省回退到 AKAZE 目录)
//   - 输入图片: tests/data/fixtures/ (真实背景 + data/big/img_1.png 目标合成)
//     ROI 取自同目录 rois.json
//   路径通过 --template-dir / --binary-template-dir / --tiny-template-dir /
//   --fixtures-dir 传入; 依赖缺失时用例自动 SKIP(不打 FAIL),
//   便于无数据 CI 环境仍可编译执行。
//
// 定位: 冒烟测试 —— 断言管线可运行、无异常、计时合法、输出目录可创建;
//       不对具体位姿精度做强断言。
// =============================================================================

#include "../framework/TestAssert.hpp"
#include "TestDataLoader.hpp"

#include "common/Config.hpp"
#include "tracker/MonoTracker.hpp"
#include "tracker/StereoTracker.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using namespace gpnp;

namespace {

// 默认模板路径: 相对 CLion 工作目录(项目根)解析, 指向 data/ 下。
//   - AKAZE 模板: 单个图像文件 (cv::imread 读取) → data/big/img_1.png
//   - BC/TT 模板: 目录 (含 N_degrees.txt/.png) → data/NewMuBan(reordered)
// 均可通过 --template-dir / --binary-template-dir / --tiny-template-dir 覆盖。
std::string g_template_dir  = "data/big/img_1.png";
std::string g_binary_dir    = "data/NewMuBan(reordered)";
std::string g_tiny_dir      = "data/NewMuBan(reordered)";

// fixtures 目录 (tests/data/fixtures) + rois.json, 可通过 --fixtures-dir 覆盖
std::string g_fixtures_dir = "tests/data/fixtures";

// ---------------------------------------------------------------------------
// 轻量级冒烟测试框架: 与 TestAssert.hpp 不同, 该文件使用可跳过的
// (SKIP) 用例, 因此维护独立的 runTest / CHECK 统计。
//   - CHECK 失败 → 抛出异常 → runTest 捕获并计为 FAIL
//   - 模板缺失 → skipNotice 抛出 SkipTestException → runTest 计为 SKIP
// ---------------------------------------------------------------------------
int g_passed_tests = 0;
int g_failed_tests = 0;

struct SkipTestException : public std::exception {
    const char* what() const noexcept override { return "skipped"; }
};

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            throw std::runtime_error(std::string("CHECK 失败: ") + #cond);   \
        }                                                                    \
    } while (0)

void runTest(const char* name, const std::function<void()>& fn) {
    std::printf("\n[TEST] %s\n", name);
    try {
        fn();
        ++g_passed_tests;
        std::printf("  [PASS]\n");
    } catch (const SkipTestException&) {
        std::printf("  [SKIP]\n");
    } catch (const std::exception& e) {
        ++g_failed_tests;
        std::printf("  [FAIL] %s\n", e.what());
    } catch (...) {
        ++g_failed_tests;
        std::printf("  [FAIL] 未捕获未知异常\n");
    }
}

// ---------------------------------------------------------------------------
// fixtures 加载: tests/data/fixtures/<scene>/<tag>_<frame:03d>.png + rois.json
// ---------------------------------------------------------------------------

// 相机内参, 主点取图像中心 (fx=fy=1000 与真实参数量级一致)
Eigen::Matrix3d makeK(int img_w, int img_h) {
    Eigen::Matrix3d K;
    K << 1000.0, 0.0, img_w / 2.0,
         0.0, 1000.0, img_h / 2.0,
         0.0, 0.0, 1.0;
    return K;
}

// 加载场景图片 (左右); 缺失返回 false 并清空图像
bool loadFixture(const std::string& scene, int frame,
                 cv::Mat& left, cv::Mat& right) {
    left  = cv::imread(gpnp_test::fixtureImagePath(g_fixtures_dir, scene, "left",  frame));
    right = cv::imread(gpnp_test::fixtureImagePath(g_fixtures_dir, scene, "right", frame));
    return !left.empty() && !right.empty();
}

// 从 rois.json 读取 ROI 并转为 gpnp::RoiRect
RoiRect roiFromJson(const std::string& scene, int frame,
                    const std::string& side, const std::string& cls) {
    cv::Rect r = gpnp_test::loadFixtureRoi(
        gpnp_test::roisJsonPath(g_fixtures_dir), scene, frame, side, cls);
    return RoiRect{r.x, r.y, r.width, r.height};
}

TrackerConfig makeTrackerCfg() {
    return makeTrackerConfig(/*scale=*/0.5f, /*gpnp_min_pts=*/3,
                             /*use_initial_pnp=*/true, /*mad_sigma=*/3.0f,
                             /*akaze_min_area=*/40000, /*tiny_max_area=*/800);
}

BinaryCornerExtractor::Config makeBinaryCfg() {
    BinaryCornerExtractor::Config c;
    c.corners = 10;
    c.kernel_size = 3;
    c.scale = 1.0f;
    c.target_size = cv::Size(100, 100);
    c.pixel_to_meter_scale_class0 = 0.5f;
    c.roi_pad_pixels = 0;
    c.otsu_ratio = 1.3f;
    return c;
}

TinyTargetExtractor::Config makeTinyCfg() {
    TinyTargetExtractor::Config c;
    c.target_size = cv::Size(50, 50);
    c.scale_factor = 4.0f;
    c.square_size_m_class0 = 0.05f;
    c.roi_pad_pixels = 0;
    return c;
}

// 检查 AKAZE 模板文件与 BC/TT 模板目录是否齐全。
bool templatesAvailable() {
    if (g_template_dir.empty()) return false;
    return std::filesystem::exists(g_template_dir);
}

void skipNotice(const char* name) {
    std::printf("  [SKIP] %s (缺少模板/fixtures, 指定 --template-dir / --fixtures-dir 后启用)\n", name);
    throw SkipTestException();
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--template-dir" && i + 1 < argc)        g_template_dir = argv[++i];
        else if (a == "--binary-template-dir" && i + 1 < argc) g_binary_dir = argv[++i];
        else if (a == "--tiny-template-dir" && i + 1 < argc)  g_tiny_dir = argv[++i];
        else if (a == "--fixtures-dir" && i + 1 < argc)       g_fixtures_dir = argv[++i];
        else if (a == "--help") {
            std::printf("用法: %s [--template-dir FILE] [--binary-template-dir DIR] "
                        "[--tiny-template-dir DIR] [--fixtures-dir DIR]\n"
                        "默认: AKAZE=%s | BC/TT=%s | fixtures=%s\n",
                        argv[0], g_template_dir.c_str(), g_binary_dir.c_str(),
                        g_fixtures_dir.c_str());
            return 0;
        }
    }

    int exit_code = 0;

    runTest("MonoTracker 冒烟: mono_akaze fixture + rois.json ROI", [&] {
        if (!templatesAvailable()) { skipNotice(__func__); }

        // 输入: mono_akaze 左图 (640×480), ROI = class0 (214,135,213×210)
        cv::Mat left = cv::imread(
            gpnp_test::fixtureImagePath(g_fixtures_dir, "mono_akaze", "left", 0));
        if (left.empty()) { skipNotice(__func__); }
        RoiRect roi = roiFromJson("mono_akaze", 0, "left", "class0");
        if (roi.width <= 0 || roi.height <= 0) { skipNotice(__func__); }
        RoiGroup group{roi, RoiRect{}, false};

        MonoTracker tracker(makeK(left.cols, left.rows), g_template_dir,
                            makeTrackerCfg(),
                            makeBinaryCfg(), g_binary_dir,
                            makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);
        tracker.setOutputDir("output/test_mono");

        PipelineResult res = tracker.process(left, /*visualize=*/false, &group);
        CHECK(res.total_time_ms() >= 0.0);
        CHECK(tracker.frameCount() >= 1);
        CHECK(tracker.getLogs().size() >= 1);
    });

    runTest("StereoTracker 冒烟: synthetic_akaze fixture + warm-start 两帧", [&] {
        if (!templatesAvailable()) { skipNotice(__func__); }

        Eigen::Vector3d t_rl(-120.0, 0.0, 0.0);   // 基线 120mm

        // 输入: synthetic_akaze 双目对 (640×480, 右图水平偏移 16px)
        // 左右 ROI 不同 (右图 x = 左图 x + 视差)
        cv::Mat l0, r0, l1, r1;
        if (!loadFixture("synthetic_akaze", 0, l0, r0) ||
            !loadFixture("synthetic_akaze", 1, l1, r1)) {
            skipNotice(__func__);
        }
        RoiRect lg0 = roiFromJson("synthetic_akaze", 0, "left",  "class0");
        RoiRect rg0 = roiFromJson("synthetic_akaze", 0, "right", "class0");
        RoiRect lg1 = roiFromJson("synthetic_akaze", 1, "left",  "class0");
        RoiRect rg1 = roiFromJson("synthetic_akaze", 1, "right", "class0");
        if (lg0.width <= 0 || rg0.width <= 0 || lg1.width <= 0 || rg1.width <= 0) {
            skipNotice(__func__);
        }
        RoiGroup lg0g{lg0, RoiRect{}, false};
        RoiGroup rg0g{rg0, RoiRect{}, false};
        RoiGroup lg1g{lg1, RoiRect{}, false};
        RoiGroup rg1g{rg1, RoiRect{}, false};

        StereoTracker tracker(makeK(l0.cols, l0.rows), Eigen::Matrix3d::Identity(),
                              t_rl, g_template_dir, makeTrackerCfg(),
                              makeBinaryCfg(), g_binary_dir,
                              makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);
        tracker.setOutputDir("output/test_stereo");

        // 两帧连续处理验证 warm-start 路径无异常
        PipelineResult r1res = tracker.process(l0, r0, false, &lg0g, &rg0g);
        PipelineResult r2res = tracker.process(l1, r1, false, &lg1g, &rg1g);
        CHECK(r1res.total_time_ms() >= 0.0);
        CHECK(r2res.total_time_ms() >= 0.0);
        CHECK(tracker.frameCount() >= 2);
    });

    runTest("StereoTracker Dual-ROI 冒烟: is_dual=true 独立路径", [&] {
        if (!templatesAvailable()) { skipNotice(__func__); }

        Eigen::Vector3d t_rl(-120.0, 0.0, 0.0);

        // 输入: synthetic_dual 双目对 (1280×960)
        //   primary(class0) = 外框 731×720; secondary(class1) = 中心 120×120
        cv::Mat l0, r0, l1, r1;
        if (!loadFixture("synthetic_dual", 0, l0, r0) ||
            !loadFixture("synthetic_dual", 1, l1, r1)) {
            skipNotice(__func__);
        }
        RoiRect pri = roiFromJson("synthetic_dual", 0, "left",  "class0");
        RoiRect sec = roiFromJson("synthetic_dual", 0, "left",  "class1");
        if (pri.width <= 0 || sec.width <= 0) { skipNotice(__func__); }

        RoiGroup lg{pri, sec, /*is_dual=*/true};
        RoiGroup rg{pri, sec, /*is_dual=*/true};

        StereoTracker tracker(makeK(l0.cols, l0.rows), Eigen::Matrix3d::Identity(),
                              t_rl, g_template_dir, makeTrackerCfg(),
                              makeBinaryCfg(), g_binary_dir,
                              makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);

        // Dual-ROI 为独立路径: process 不应抛出, 且 frame_count 递增
        PipelineResult r1res = tracker.process(l0, r0, false, &lg, &rg);
        PipelineResult r2res = tracker.process(l1, r1, false, &lg, &rg);
        CHECK(r1res.total_time_ms() >= 0.0);
        CHECK(r2res.total_time_ms() >= 0.0);
        CHECK(tracker.frameCount() >= 2);
    });

    exit_code = (g_failed_tests == 0) ? 0 : 1;
    std::printf("\n================ 集成测试结果 ================\n");
    std::printf("通过 %d, 失败 %d\n", g_passed_tests, g_failed_tests);
    return exit_code;
}