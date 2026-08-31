// =============================================================================
// test_synthetic_pipeline.cpp — 合成图全流程测试 (YOLO → 位姿)
//
// 目标:
//   覆盖 fixtures_rich 合成数据集上「YOLO 检测 → ROI → 特征提取 → 匹配 →
//   PnP → 位姿」的完整链路，双目/单目 × 四策略（TinyTarget / BinaryCorner /
//   AKAZE / Dual-ROI）共 8 项，并与 solvePnP 真值比对位姿精度。
//
// 数据依赖 (tests/data/fixtures_rich, 由 scripts/generate_synthetic_dataset.py
// 生成, 被 .gitignore 忽略):
//   - synthetic_{tiny,bc,akaze,dual}/  双目对 (右图=左图右移 DISP px)
//   - mono_{tiny,bc,akaze,dual}/       单目左图副本
//   - 每图 *_class0.txt / *_class1.txt  特征点 (Corner_N: x, y)
//   - scripts/class0_points.txt         模板 10 点 (img_1.png 像素坐标)
//   - 模板: data/big/img_1.png (AKAZE) + data/NewMuBan(reordered) (BC/TT)
//
// 合成内参: f=1000, cx=W/2, cy=H/2; 模板物理 500×500 mm。
//
// 定位: 端到端精度测试 —— 与 solvePnP 真值比对平移/旋转误差;
//       YOLO 检测结果与特征点包围盒(真值)不一致时回退到特征点 ROI (不 FAIL)。
//
// 实测结论 (fixtures_rich, f=1000 内参 + 500mm 物理约定):
//   - Dual-ROI (单目/双目): 平移 <1% / 旋转 <0.5°, 精确。
//   - 单目 tiny/bc/akaze: 平移 2~5% 正确, 旋转有平面歧义误差 (6~34°)。
//   - 双目 bc: 平移 2.7% 正确; 双目 tiny/akaze: 立体光流/匹配退化, 平移/旋转均差
//     (立体路径对小目标的光流 FB 校验 + 左右角点 IoU 匹配失败, 属已知限制)。
// =============================================================================

#include "common/Config.hpp"
#include "tracker/MonoTracker.hpp"
#include "tracker/StereoTracker.hpp"
#include "detection/YoloRoiProvider.hpp"
#include "detection/RoiGenerator.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "utils/PoseUtils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using namespace gpnp;

namespace {

// ---------------------------------------------------------------------------
// 默认路径 (相对项目根, 经 docker-toolchain 在容器内以项目根为 CWD 运行)
// ---------------------------------------------------------------------------
std::string g_fixtures_rich_dir = "tests/data/fixtures_rich";
std::string g_template_dir      = "data/big/img_1.png";
std::string g_binary_dir        = "data/NewMuBan(reordered)";
std::string g_tiny_dir          = "data/NewMuBan(reordered)";
std::string g_points_dir        = "scripts";               // class0/class1_points.txt
std::string g_model_path        = "best.onnx";
bool g_no_yolo                  = false;                   // 强制回退特征点 ROI (诊断用)

// ---------------------------------------------------------------------------
// 轻量级可跳过测试框架 (与 test_integration 一致)
// ---------------------------------------------------------------------------
int g_passed_tests = 0;
int g_failed_tests = 0;
int g_skipped_tests = 0;

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
        ++g_skipped_tests;
        std::printf("  [SKIP]\n");
    } catch (const std::exception& e) {
        ++g_failed_tests;
        std::printf("  [FAIL] %s\n", e.what());
    } catch (...) {
        ++g_failed_tests;
        std::printf("  [FAIL] 未捕获未知异常\n");
    }
}

void skipNotice(const char* why) {
    std::printf("  [SKIP] %s\n", why);
    throw SkipTestException();
}

bool fileExists(const std::string& p) { return std::filesystem::exists(p); }

// ---------------------------------------------------------------------------
// 相机内参 (合成数据: f=1000, 主点=图像中心)
// ---------------------------------------------------------------------------
Eigen::Matrix3d makeK(int img_w, int img_h) {
    Eigen::Matrix3d K;
    K << 1000.0, 0.0, img_w / 2.0,
         0.0, 1000.0, img_h / 2.0,
         0.0, 0.0, 1.0;
    return K;
}

// ---------------------------------------------------------------------------
// 配置构造 (物理尺寸统一取 500×500 mm, 与 AKAZE 模板/真值一致)
// ---------------------------------------------------------------------------
TrackerConfig makeTrackerCfg() {
    // scale=1.0: 合成图 akaze 目标仅 ~201px, 0.5 降采样后特征点过少导致匹配失败
    // dual_roi_akaze_scale=1.0: 与 config/tracker_config_synth_dual.json 一致
    return makeTrackerConfig(/*scale=*/1.0, /*gpnp_min_pts=*/3,
                             /*use_initial_pnp=*/true,
                             /*template_real_width_mm=*/500.0,
                             /*template_real_height_mm=*/500.0,
                             /*akaze_min_area=*/40000, /*tiny_max_area=*/800,
                             /*akaze_min_area_class1=*/0, /*tiny_max_area_class1=*/0,
                             /*dual_roi_secondary_expand=*/50,
                             /*dual_roi_akaze_scale=*/1.0);
}

BinaryCornerExtractor::Config makeBinaryCfg() {
    BinaryCornerExtractor::Config c;
    c.corners = 10;
    c.kernel_size = 3;
    c.scale = 3.0;                      // 对应 synth config 的 corner_scale
    c.target_size = cv::Size(100, 100);
    // 标定: NewMuBan 模板(50 单位) = class0 轮廓(569px) = 500mm 约定下 356mm
    // → 356/49.7 ≈ 7.2 mm/单位, 与 AKAZE 500mm 约定对齐
    c.pixel_to_meter_scale_class0 = 0.0072;
    c.pixel_to_meter_scale_class1 = 0.00072;
    c.roi_pad_pixels = 3;
    c.otsu_ratio = 1.0;
    return c;
}

TinyTargetExtractor::Config makeTinyCfg() {
    TinyTargetExtractor::Config c;
    c.target_size = cv::Size(50, 50);
    c.scale_factor = 4;
    // 标定: class0 轮廓 ≈ 356mm (500mm 约定), 与 AKAZE 对齐
    c.square_size_m_class0 = 0.356;
    c.square_size_m_class1 = 0.04;
    c.roi_pad_pixels = 5;
    return c;
}

// ---------------------------------------------------------------------------
// 场景定义: 4 策略 × 单目/双目
//   disp          右图右移像素 (生成器 DISP)
//   size_nominal  名义目标短边像素 (生成器 STATE_SIZES)
//   t_thresh_mm   平移绝对误差阈值 (mm)
//   r_thresh_deg  旋转误差阈值 (°)
// ---------------------------------------------------------------------------
struct SceneDef {
    const char* scene;         // "tiny" | "bc" | "akaze" | "dual"
    const char* mono_dir;      // "mono_tiny" ...
    const char* stereo_dir;    // "synthetic_tiny" ...
    double disp;
    double size_nominal;
    double t_rel_thresh_pct;   // 平移相对误差阈值 (%) — 深度跨 625~25000mm, 用相对值
    double r_sanity_deg;       // 旋转粗检阈值 (°) — 平面目标有姿态歧义, 仅粗检
};

static const SceneDef kScenes[4] = {
    {"tiny",  "mono_tiny",      "synthetic_tiny",   4,   20,  15.0, 60.0},
    {"bc",    "mono_bc",        "synthetic_bc",     8,   150,  10.0, 45.0},
    {"akaze", "mono_akaze",     "synthetic_akaze",  16,  210,  10.0, 45.0},
    {"dual",  "mono_dual",      "synthetic_dual",   20,  720,  5.0,  15.0},
};

// ---------------------------------------------------------------------------
// 真值位姿: solvePnP(class0 投影点 2D, class0 模板点→3D mm)
//   3D 点 X = px*500/798, Y = py*500/786, Z = 0  (与 AKAZE setTemplateData 一致)
// ---------------------------------------------------------------------------
struct GroundTruth {
    bool valid = false;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

GroundTruth computeGroundTruth(const std::string& class0_txt,
                               const std::string& class0_tmpl_txt,
                               const Eigen::Matrix3d& K) {
    GroundTruth gt;
    auto img_pts = gpnp::readCorners(class0_txt);
    auto tmpl_pts = gpnp::readCorners(class0_tmpl_txt);
    const size_t n = std::min(img_pts.size(), tmpl_pts.size());
    if (n < 4) return gt;

    std::vector<cv::Point3d> obj_pts(n);
    std::vector<cv::Point2d> img_pts_cv(n);
    for (size_t i = 0; i < n; ++i) {
        obj_pts[i] = cv::Point3d(tmpl_pts[i].x * 500.0 / 798.0 - 250.0,
                                 tmpl_pts[i].y * 500.0 / 786.0 - 250.0, 0.0);
        img_pts_cv[i] = cv::Point2d(img_pts[i].x, img_pts[i].y);
    }

    cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
        K(0,0), K(0,1), K(0,2),
        K(1,0), K(1,1), K(1,2),
        K(2,0), K(2,1), K(2,2));

    cv::Mat rvec, tvec;
    if (!cv::solvePnP(obj_pts, img_pts_cv, K_cv, cv::noArray(),
                      rvec, tvec, false, cv::SOLVEPNP_ITERATIVE)) {
        return gt;
    }
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    gt.R << R_cv.at<double>(0,0), R_cv.at<double>(0,1), R_cv.at<double>(0,2),
            R_cv.at<double>(1,0), R_cv.at<double>(1,1), R_cv.at<double>(1,2),
            R_cv.at<double>(2,0), R_cv.at<double>(2,1), R_cv.at<double>(2,2);
    gt.t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    gt.valid = true;
    return gt;
}

// ---------------------------------------------------------------------------
// 回退 ROI: 特征点 txt 包围盒
//   class0 是目标外轮廓 (模板 img_1.png 798×786 中的 569×571 轮廓), 生成器按
//   完整模板短边 786px 计算 "size"。故 class0 轮廓 bbox 需按边距比 (798/569,
//   786/571) 外扩成完整目标 bbox, 使 ROI 面积与生成器 "size" 一致 (否则 akaze
//   场景会被误判为 bc)。class1 是中心区域, 无需外扩。
// ---------------------------------------------------------------------------
RoiRect bboxToRoi(const std::vector<cv::Point2f>& pts, int pad) {
    if (pts.empty()) return RoiRect{};
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const auto& p : pts) {
        x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    }
    int x = std::max(0, static_cast<int>(std::floor(x0)) - pad);
    int y = std::max(0, static_cast<int>(std::floor(y0)) - pad);
    int w = static_cast<int>(std::ceil(x1)) - x + pad;
    int h = static_cast<int>(std::ceil(y1)) - y + pad;
    return RoiRect{x, y, w, h};
}

RoiRect fullTargetRoi(const std::vector<cv::Point2f>& class0_pts, int pad) {
    RoiRect c = bboxToRoi(class0_pts, 0);
    if (!c.valid()) return c;
    const double kCw = 569.0, kCh = 571.0;  // class0 轮廓在模板中的尺寸
    const double kFw = 798.0, kFh = 786.0;  // 模板完整尺寸
    double cx = c.x + c.width / 2.0;
    double cy = c.y + c.height / 2.0;
    double fw = c.width * kFw / kCw;
    double fh = c.height * kFh / kCh;
    int x = std::max(0, static_cast<int>(std::round(cx - fw / 2.0)) - pad);
    int y = std::max(0, static_cast<int>(std::round(cy - fh / 2.0)) - pad);
    int w = static_cast<int>(std::round(fw)) + pad;
    int h = static_cast<int>(std::round(fh)) + pad;
    return RoiRect{x, y, w, h};
}

RoiGroup fallbackGroup(const std::string& class0_txt,
                       const std::string& class1_txt,
                       bool need_class1) {
    auto c0 = gpnp::readCorners(class0_txt);
    RoiRect pri = fullTargetRoi(c0, 3);
    if (need_class1) {
        auto c1 = gpnp::readCorners(class1_txt);
        RoiRect sec = bboxToRoi(c1, 2);
        if (sec.width > 0 && sec.height > 0) {
            return RoiGroup{pri, sec, /*is_dual=*/true};
        }
    }
    return RoiGroup{pri, RoiRect{}, /*is_dual=*/false};
}

// ---------------------------------------------------------------------------
// 位姿误差度量
// ---------------------------------------------------------------------------
double rotErrorDeg(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) {
    Eigen::Matrix3d dR = R2.transpose() * R1;
    double c = (dR.trace() - 1.0) / 2.0;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / M_PI;
}

double transErrorMm(const Eigen::Vector3d& t1, const Eigen::Vector3d& t2) {
    return (t1 - t2).norm();
}

std::string tStr(const Eigen::Vector3d& t) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "(%.1f, %.1f, %.1f)", t(0), t(1), t(2));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// YOLO 优先 + 真值回退 的 ROI 获取
//   YOLO 检测结果需与特征点包围盒 (真值) 一致, 否则回退 (YOLO 对合成小目标不可靠)
// ---------------------------------------------------------------------------
bool roiConsistent(const RoiRect& a, const RoiRect& b) {
    if (!a.valid() || !b.valid()) return false;
    cv::Point2f ca(a.x + a.width / 2.0f, a.y + a.height / 2.0f);
    cv::Point2f cb(b.x + b.width / 2.0f, b.y + b.height / 2.0f);
    double center_dist = cv::norm(ca - cb);
    double max_side = std::max({(double)a.width, (double)a.height,
                                (double)b.width, (double)b.height});
    double size_ratio = std::max((double)a.width / b.width,
                                 (double)b.width / a.width);
    // 中心距 < 0.5×最大边 且 尺寸比 < 1.5 视为一致
    return center_dist < 0.5 * max_side && size_ratio < 1.5;
}

// ---------------------------------------------------------------------------
// 单个用例执行器
// ---------------------------------------------------------------------------
void runCase(bool stereo, const SceneDef& def,
             YoloRoiProvider& yolo, bool yolo_ready) {
    const char* dir = stereo ? def.stereo_dir : def.mono_dir;
    const std::string left_path = g_fixtures_rich_dir + "/" + dir + "/left_000.png";
    const std::string right_path = g_fixtures_rich_dir + "/" + dir + "/right_000.png";

    // 依赖检查 (缺失 SKIP)
    if (!fileExists(left_path)) skipNotice("缺 fixtures_rich 左图");
    if (!fileExists(g_template_dir)) skipNotice("缺 AKAZE 模板");
    if (!fileExists(g_binary_dir + "/0_degrees.png")) skipNotice("缺 BC/TT 模板");
    const std::string tmpl_c0 = g_points_dir + "/class0_points.txt";
    const std::string tmpl_c1 = g_points_dir + "/class1_points.txt";
    if (!fileExists(tmpl_c0) || !fileExists(tmpl_c1)) skipNotice("缺模板特征点 txt");

    cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
    if (left.empty()) skipNotice("imread 左图失败");

    const std::string lc0 = g_fixtures_rich_dir + "/" + dir + "/left_000_class0.txt";
    const std::string lc1 = g_fixtures_rich_dir + "/" + dir + "/left_000_class1.txt";
    if (!fileExists(lc0)) skipNotice("缺 class0 特征点 txt");

    const Eigen::Matrix3d K = makeK(left.cols, left.rows);
    GroundTruth gt = computeGroundTruth(lc0, tmpl_c0, K);
    if (!gt.valid) skipNotice("真值 solvePnP 失败");

    // ROI (YOLO 优先 + 回退)
    bool need_class1 = (def.scene == std::string("dual"));
    bool used_yolo = false;

    PipelineResult res;
    if (stereo) {
        cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
        if (right.empty()) skipNotice("imread 右图失败");
        const std::string rc0 = g_fixtures_rich_dir + "/" + dir + "/right_000_class0.txt";
        const std::string rc1 = g_fixtures_rich_dir + "/" + dir + "/right_000_class1.txt";

        RoiGroup fb_l = fallbackGroup(lc0, lc1, need_class1);
        RoiGroup fb_r = fallbackGroup(rc0, rc1, need_class1);
        RoiGroup lg = fb_l, rg = fb_r;
        if (yolo_ready && !g_no_yolo) {
            auto pair = yolo.detect(left, right);
            bool ok = pair.first.valid() && pair.second.valid() &&
                      roiConsistent(pair.first.primary, fb_l.primary) &&
                      roiConsistent(pair.second.primary, fb_r.primary) &&
                      (!need_class1 || (pair.first.is_dual && pair.second.is_dual));
            if (ok) { lg = pair.first; rg = pair.second; used_yolo = true; }
        }
        if (!lg.valid() || !rg.valid()) skipNotice("ROI 无效");

        double B = def.disp * 500.0 / def.size_nominal;
        Eigen::Vector3d t_rl(-B, 0.0, 0.0);
        StereoTracker tracker(K, Eigen::Matrix3d::Identity(), t_rl,
                              g_template_dir, makeTrackerCfg(),
                              makeBinaryCfg(), g_binary_dir,
                              makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);
        tracker.setOutputDir("output/test_synthetic_pipeline");
        tracker.setFrameNumber(0);
        res = tracker.process(left, right, /*visualize=*/false, &lg, &rg);
    } else {
        RoiGroup fb = fallbackGroup(lc0, lc1, need_class1);
        RoiGroup lg = fb;
        if (yolo_ready && !g_no_yolo) {
            RoiGroup gy = yolo.detectMono(left);
            bool ok = gy.valid() && roiConsistent(gy.primary, fb.primary) &&
                      (!need_class1 || gy.is_dual);
            if (ok) { lg = gy; used_yolo = true; }
        }
        if (!lg.valid()) skipNotice("ROI 无效");

        MonoTracker tracker(K, g_template_dir, makeTrackerCfg(),
                            makeBinaryCfg(), g_binary_dir,
                            makeTinyCfg(), g_tiny_dir);
        tracker.setVerboseConsole(false);
        tracker.setOutputDir("output/test_synthetic_pipeline");
        tracker.setFrameNumber(0);
        res = tracker.process(left, /*visualize=*/false, &lg);
    }

    // 断言: 成功 + 旋转误差 + 平移误差 (相对真值)
    const double r_err = rotErrorDeg(res.R, gt.R);
    const double t_err = transErrorMm(res.t, gt.t);
    const double t_gt_norm = gt.t.norm();
    const double t_err_rel = (t_gt_norm > 1e-9) ? (t_err / t_gt_norm * 100.0) : 0.0;

    std::printf("    ROI=%s | t=%s | t_gt=%s | r_err=%.2f° | t_err=%.1fmm (%.1f%%)\n",
                used_yolo ? "YOLO" : "回退",
                tStr(res.t).c_str(), tStr(gt.t).c_str(),
                r_err, t_err, t_err_rel);

    CHECK(res.success);
    CHECK(res.t(2) > 0.0);
    if (t_err_rel >= def.t_rel_thresh_pct) {
        throw std::runtime_error("平移相对误差超阈值: " + std::to_string(t_err_rel) +
                                 "% >= " + std::to_string(def.t_rel_thresh_pct) + "%");
    }
    if (r_err >= def.r_sanity_deg) {
        throw std::runtime_error("旋转误差粗检超阈值: " + std::to_string(r_err) +
                                 "° >= " + std::to_string(def.r_sanity_deg) + "°");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::printf("=== [诊断] test_synthetic_pipeline 路径解析 ===\n");
    std::printf("  CWD        : %s\n", std::filesystem::current_path().string().c_str());
    std::printf("  fixtures_rich : %s exists=%d\n", g_fixtures_rich_dir.c_str(),
                fileExists(g_fixtures_rich_dir));
    std::printf("  template   : %s exists=%d\n", g_template_dir.c_str(),
                fileExists(g_template_dir));
    std::printf("  binary_dir : %s exists=%d\n", g_binary_dir.c_str(),
                fileExists(g_binary_dir));
    std::printf("  model      : %s exists=%d\n", g_model_path.c_str(),
                fileExists(g_model_path));
    std::printf("=== [诊断] 结束 ===\n");

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--fixtures-rich-dir" && i + 1 < argc) g_fixtures_rich_dir = argv[++i];
        else if (a == "--template-dir" && i + 1 < argc)       g_template_dir = argv[++i];
        else if (a == "--binary-template-dir" && i + 1 < argc) g_binary_dir = argv[++i];
        else if (a == "--tiny-template-dir" && i + 1 < argc)  g_tiny_dir = argv[++i];
        else if (a == "--points-dir" && i + 1 < argc)          g_points_dir = argv[++i];
        else if (a == "--model-path" && i + 1 < argc)          g_model_path = argv[++i];
        else if (a == "--no-yolo")                              g_no_yolo = true;
        else if (a == "--help") {
            std::printf("用法: %s [--fixtures-rich-dir DIR] [--template-dir FILE] "
                        "[--binary-template-dir DIR] [--tiny-template-dir DIR] "
                        "[--points-dir DIR] [--model-path FILE] [--no-yolo]\n", argv[0]);
            return 0;
        }
    }

    // 一次性构造 YOLO provider (模型缺失则回退特征点 ROI, 不 SKIP)
    YoloRoiProvider yolo;
    bool yolo_ready = false;
    if (fileExists(g_model_path)) {
        YoloConfig yolo_cfg = makeYoloConfig(g_model_path, DeviceType::CPU, 0.5f);
        yolo_cfg.target_class_id = 0;
        yolo_cfg.roi_expand_ratio = 0.0f;
        RoiGenerator::Config roi_cfg{/*target_class_id=*/0, /*roi_expand_ratio=*/0.0f,
                                     /*roi_min_size=*/0, /*dual_trigger_area=*/490000};
        yolo_ready = yolo.initialize(yolo_cfg, roi_cfg);
        std::printf("YOLO 初始化: %s\n", yolo_ready ? "成功" : "失败(回退特征点 ROI)");
    } else {
        std::printf("YOLO 模型缺失, 全部走特征点 ROI 回退\n");
    }

    for (const auto& def : kScenes) {
        char mono_name[64], stereo_name[64];
        std::snprintf(mono_name, sizeof(mono_name), "单目 %s", def.scene);
        std::snprintf(stereo_name, sizeof(stereo_name), "双目 %s", def.scene);
        runTest(mono_name,   [&] { runCase(false, def, yolo, yolo_ready); });
        runTest(stereo_name, [&] { runCase(true,  def, yolo, yolo_ready); });
    }

    const int exit_code = (g_failed_tests == 0) ? 0 : 1;
    std::printf("\n================ 合成全流程测试结果 ================\n");
    std::printf("通过 %d, 失败 %d, 跳过 %d\n", g_passed_tests, g_failed_tests, g_skipped_tests);
    return exit_code;
}
