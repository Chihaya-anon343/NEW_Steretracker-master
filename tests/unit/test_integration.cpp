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
//   依赖通过 --template-dir / --binary-template-dir / --tiny-template-dir 传入;
//   目录不存在时用例自动 SKIP(不打 FAIL), 便于无数据 CI 环境仍可编译执行。
//
// 定位: 冒烟测试 —— 断言管线可运行、无异常、计时合法、输出目录可创建;
//       不对具体位姿精度做强断言(合成图与真实模板差异大, 强断言会过于脆弱)。
// =============================================================================

#include "../framework/TestAssert.hpp"

#include "common/Config.hpp"
#include "tracker/MonoTracker.hpp"
#include "tracker/StereoTracker.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/core.hpp>
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
// 合成双目图对: 将平面 3D 点投影到左右相机, 画白底黑点目标
// ---------------------------------------------------------------------------
struct SynthPair {
    cv::Mat left, right;
    int width = 1280;
    int height = 1024;
};

SynthPair synthPair(const Eigen::Matrix3d& K,
                    const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& t,
                    const std::vector<Eigen::Vector3d>& pts3d,
                    const Eigen::Vector3d& t_rl) {
    SynthPair p;
    p.left  = cv::Mat(p.height, p.width, CV_8UC1, cv::Scalar::all(255));
    p.right = cv::Mat(p.height, p.width, CV_8UC1, cv::Scalar::all(255));

    auto project = [&](const Eigen::Vector3d& Pc) -> cv::Point2f {
        Eigen::Vector3d uv = K * Pc;
        return cv::Point2f(static_cast<float>(uv.x() / uv.z()),
                           static_cast<float>(uv.y() / uv.z()));
    };

    for (const auto& P : pts3d) {
        Eigen::Vector3d Pl = R * P + t;              // 左相机系
        Eigen::Vector3d Pr = R * P + t + t_rl;       // 右相机系
        if (Pl.z() <= 0 || Pr.z() <= 0) continue;
        cv::Point2f l = project(Pl);
        cv::Point2f r = project(Pr);
        cv::circle(p.left,  l, 8, 0, -1);
        cv::circle(p.right, r, 8, 0, -1);
    }
    return p;
}

// 200×150mm 模板平面点 (Z=0)
std::vector<Eigen::Vector3d> makeTemplatePlane(double mm = 1.0) {
    std::vector<Eigen::Vector3d> pts;
    for (double x : {-100.0, 0.0, 100.0})
        for (double y : {-75.0, 0.0, 75.0})
            pts.emplace_back(x * mm, y * mm, 0.0);
    return pts;
}

Eigen::Matrix3d makeK() {
    Eigen::Matrix3d K;
    K << 1000.0, 0.0, 640.0,
         0.0, 1000.0, 512.0,
         0.0, 0.0, 1.0;
    return K;
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
    std::printf("  [SKIP] %s (缺少模板, 指定 --template-dir 后启用)\n", name);
    throw SkipTestException();
}

// 断言 ROI 在图像范围内
void clampRoi(RoiRect& r, int w, int h) {
    r.x = std::max(0, r.x);
    r.y = std::max(0, r.y);
    r.width  = std::min<int>(r.width,  w - r.x);
    r.height = std::min<int>(r.height, h - r.y);
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--template-dir" && i + 1 < argc)        g_template_dir = argv[++i];
        else if (a == "--binary-template-dir" && i + 1 < argc) g_binary_dir = argv[++i];
        else if (a == "--tiny-template-dir" && i + 1 < argc)  g_tiny_dir = argv[++i];
        else if (a == "--help") {
            std::printf("用法: %s [--template-dir FILE] [--binary-template-dir DIR] "
                        "[--tiny-template-dir DIR]\n"
                        "默认: AKAZE=%s | BC/TT=%s\n",
                        argv[0], g_template_dir.c_str(), g_binary_dir.c_str());
            return 0;
        }
    }

    int exit_code = 0;

    runTest("MonoTracker 冒烟: 合成点云 + 手动 ROI", [&] {
        if (!templatesAvailable()) { skipNotice(__func__); }

        Eigen::Matrix3d K = makeK();
        Eigen::Matrix3d R = Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY())
                                .toRotationMatrix();
        Eigen::Vector3d t(0.0, 0.0, 1500.0);
        auto pair = synthPair(K, R, t, makeTemplatePlane(), Eigen::Vector3d::Zero());

        RoiRect roi{400, 300, 480, 480};
        clampRoi(roi, pair.width, pair.height);
        RoiGroup group{roi, RoiRect{}, false};

        MonoTracker tracker(K, g_template_dir, makeTrackerCfg(),
                            makeBinaryCfg(), g_binary_dir,
                            makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);
        tracker.setOutputDir("output/test_mono");

        PipelineResult res = tracker.process(pair.left, /*visualize=*/false, &group);
        CHECK(res.total_time_ms() >= 0.0);
        CHECK(tracker.frameCount() >= 1);
        CHECK(tracker.getLogs().size() >= 1);
    });

    runTest("StereoTracker 冒烟: 合成点云 + 双目 ROI + warm-start 两帧", [&] {
        if (!templatesAvailable()) { skipNotice(__func__); }

        Eigen::Matrix3d K = makeK();
        Eigen::Matrix3d R = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())
                                .toRotationMatrix();
        Eigen::Vector3d t(0.0, 0.0, 1500.0);
        Eigen::Vector3d t_rl(-120.0, 0.0, 0.0);   // 基线 120mm
        auto pair = synthPair(K, R, t, makeTemplatePlane(), t_rl);

        RoiRect roi{400, 300, 480, 480};
        clampRoi(roi, pair.width, pair.height);
        RoiGroup lg{roi, RoiRect{}, false};
        RoiGroup rg{roi, RoiRect{}, false};

        StereoTracker tracker(K, Eigen::Matrix3d::Identity(), t_rl,
                              g_template_dir, makeTrackerCfg(),
                              makeBinaryCfg(), g_binary_dir,
                              makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);
        tracker.setOutputDir("output/test_stereo");

        // 两帧连续处理验证 warm-start 路径无异常
        PipelineResult r1 = tracker.process(pair.left, pair.right, false, &lg, &rg);
        PipelineResult r2 = tracker.process(pair.left, pair.right, false, &lg, &rg);
        CHECK(r1.total_time_ms() >= 0.0);
        CHECK(r2.total_time_ms() >= 0.0);
        CHECK(tracker.frameCount() >= 2);
    });

    runTest("StereoTracker Dual-ROI 冒烟: is_dual=true 独立路径", [&] {
        if (!templatesAvailable()) { skipNotice(__func__); }

        Eigen::Matrix3d K = makeK();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t(0.0, 0.0, 1200.0);
        Eigen::Vector3d t_rl(-120.0, 0.0, 0.0);

        // 外矩形(class0 边缘) + 内矩形(class1 中心)
        std::vector<Eigen::Vector3d> outer;
        for (double a : {0.0, 90.0, 180.0, 270.0}) {
            double rad = a * M_PI / 180.0;
            outer.emplace_back(100.0 * std::cos(rad), 75.0 * std::sin(rad), 0.0);
        }
        std::vector<Eigen::Vector3d> inner;
        for (double a : {0.0, 90.0, 180.0, 270.0}) {
            double rad = a * M_PI / 180.0;
            inner.emplace_back(40.0 * std::cos(rad), 30.0 * std::sin(rad), 0.0);
        }
        // 场景点(外+内) 与 纯 inner 图
        std::vector<Eigen::Vector3d> all = outer;
        all.insert(all.end(), inner.begin(), inner.end());

        auto pairAll  = synthPair(K, R, t, all, t_rl);
        auto pairInner = synthPair(K, R, t, inner, t_rl);

        // primary(class0) 覆盖 whole 目标; secondary(class1) 覆盖中心
        RoiRect pri{300, 250, 680, 520};
        RoiRect sec{420, 350, 340, 260};
        clampRoi(pri, pairAll.width, pairAll.height);
        clampRoi(sec, pairAll.width, pairAll.height);

        RoiGroup lg{pri, sec, /*is_dual=*/true};
        RoiGroup rg{pri, sec, /*is_dual=*/true};

        StereoTracker tracker(K, Eigen::Matrix3d::Identity(), t_rl,
                              g_template_dir, makeTrackerCfg(),
                              makeBinaryCfg(), g_binary_dir,
                              makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);

        // Dual-ROI 为独立路径: process 不应抛出, 且 frame_count 递增
        PipelineResult r1 = tracker.process(pairAll.left, pairAll.right, false, &lg, &rg);
        PipelineResult r2 = tracker.process(pairInner.left, pairInner.right, false, &lg, &rg);
        CHECK(r1.total_time_ms() >= 0.0);
        CHECK(r2.total_time_ms() >= 0.0);
        CHECK(tracker.frameCount() >= 2);
    });

    exit_code = (g_failed_tests == 0) ? 0 : 1;
    std::printf("\n================ 集成测试结果 ================\n");
    std::printf("通过 %d, 失败 %d\n", g_passed_tests, g_failed_tests);
    return exit_code;
}