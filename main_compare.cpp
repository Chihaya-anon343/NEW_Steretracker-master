/**
 * @file main_compare.cpp
 * @brief 对比方法主 pipeline —— YOLO 框 → SIFT 特征点匹配 → PnP。
 *
 * 与主 pipeline (main.cpp) 的区别:
 *   - 不走 StereoTracker/MonoTracker: 无面积分级、无策略链、无退化链、无 ROI padding
 *   - 链路: 输入文件夹 → YOLO 检测 (class0=整体框 / class1=中心框) →
 *           ROI 裁剪 → SiftKeypointExtractor (或 network 占位) → MonoPnP
 *
 * 运行: ./pose_compare [config_path]
 * 配置: config/compare_config.json (仅 input.directory_path 必填, 其余占位)
 */

#include "detection/YoloDetector.hpp"
#include "feature/NetworkKeypointExtractor.hpp"
#include "feature/SiftKeypointExtractor.hpp"
#include "pose/MonoPnPSolver.hpp"
#include "utils/AsyncImageSaver.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;
void handleSignal(int) { g_stop_requested = 1; }

// ---- 配置读取辅助: 缺省字段用默认值 (仅 directory_path 必填) ----
std::string readStr(const cv::FileNode& n, const char* key, const std::string& def) {
    cv::FileNode v = n[key];
    return v.empty() ? def : static_cast<std::string>(v);
}
double readDouble(const cv::FileNode& n, const char* key, double def) {
    cv::FileNode v = n[key];
    return v.empty() ? def : static_cast<double>(v);
}
int readInt(const cv::FileNode& n, const char* key, int def) {
    cv::FileNode v = n[key];
    return v.empty() ? def : static_cast<int>(v);
}
bool readBool(const cv::FileNode& n, const char* key, bool def) {
    cv::FileNode v = n[key];
    return v.empty() ? def : static_cast<int>(v) != 0;
}

/// 列出目录下全部图像 (按文件名排序)
std::vector<std::filesystem::path> listImages(const std::string& dir) {
    namespace fsp = std::filesystem;
    std::vector<std::filesystem::path> out;
    std::error_code ec;
    for (auto it = fsp::directory_iterator(dir, fsp::directory_options::skip_permission_denied, ec);
         it != fsp::directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            out.push_back(it->path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// 取指定 class 中置信度最高的检测框
const gpnp::Detection* bestDetection(const std::vector<gpnp::Detection>& dets, int class_id) {
    const gpnp::Detection* best = nullptr;
    for (const auto& d : dets)
        if (d.class_id == class_id && (!best || d.confidence > best->confidence))
            best = &d;
    return best;
}

/// 检测框 → 与图像边界求交的整数 ROI
cv::Rect bboxToRoi(const cv::Rect2f& bbox, const cv::Size& img_sz) {
    int x = std::max(0, static_cast<int>(bbox.x));
    int y = std::max(0, static_cast<int>(bbox.y));
    int w = std::min(static_cast<int>(std::ceil(bbox.width)), img_sz.width - x);
    int h = std::min(static_cast<int>(std::ceil(bbox.height)), img_sz.height - y);
    if (w <= 0 || h <= 0) return cv::Rect();
    return cv::Rect(x, y, w, h);
}

/// 统一的提取结果 (SiftKeypointExtractor::Result 与
/// NetworkKeypointExtractor::Result 字段同形, 归一后供帧循环使用)
struct ExtractResult {
    bool success = false;
    std::string message;
    std::vector<cv::Point2f> pts_2d;       ///< ROI 局部坐标
    std::vector<Eigen::Vector3d> pts_3d;   ///< 模板 3D (mm, z=0), 与 pts_2d 1:1
};
using ExtractFn = std::function<ExtractResult(const cv::Mat& roi_bgr)>;

} // namespace

int main(int argc, char** argv) {
    using namespace gpnp;

    // ========================================================================
    // ① 读取配置 (仅 input.directory_path 必填, 其余为占位默认值)
    // ========================================================================
    std::string config_path = "config/compare_config.json";
    if (argc >= 2) config_path = argv[1];

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开配置文件: " << config_path << std::endl;
        return 1;
    }

    std::string input_dir = readStr(fs["input"], "directory_path", "");
    if (input_dir.empty()) {
        std::cerr << "配置缺少必填项: input.directory_path" << std::endl;
        return 1;
    }

    // 相机内参 (占位: 标定后替换)
    double fx = readDouble(fs["camera"], "fx", 1000.0);
    double fy = readDouble(fs["camera"], "fy", 1000.0);
    double cx = readDouble(fs["camera"], "cx", 640.0);
    double cy = readDouble(fs["camera"], "cy", 512.0);

    // YOLO (占位: 与主 pipeline 共用同一模型)
    YoloConfig yolo_cfg;
    yolo_cfg.model_path     = readStr(fs["yolo"], "model_path", "yolo_onnx/yolov8n.onnx");
    yolo_cfg.conf_threshold = static_cast<float>(readDouble(fs["yolo"], "conf_threshold", 0.5));
    yolo_cfg.iou_threshold  = static_cast<float>(readDouble(fs["yolo"], "iou_threshold", 0.45));
    yolo_cfg.device         = DeviceType::CPU;
    yolo_cfg.intra_op_threads = readInt(fs["yolo"], "intra_op_threads", 4);

    // 特征点提取: method = "sift" (默认) | "orb" | "network" (占位)
    std::string extract_method = readStr(fs["extractor"], "method", "sift");

    SiftKeypointExtractor::Config sift_cfg;
    sift_cfg.method                = extract_method;
    sift_cfg.template_path         = readStr(fs["extractor"], "template_path", "");
    sift_cfg.template_real_width_mm  = readDouble(fs["extractor"], "template_real_width_mm", 0.0);
    sift_cfg.template_real_height_mm = readDouble(fs["extractor"], "template_real_height_mm", 0.0);
    sift_cfg.scale                 = readDouble(fs["extractor"], "scale", 1.0);
    sift_cfg.nfeatures             = readInt(fs["extractor"], "nfeatures", 2000);
    sift_cfg.ratio_threshold       = readDouble(fs["extractor"], "ratio_threshold", 0.75);
    sift_cfg.min_pts               = readInt(fs["extractor"], "min_pts", 4);

    // 关键点网络 (占位: 接入真实模型后生效)
    NetworkKeypointExtractor::Config net_cfg;
    net_cfg.model_path             = readStr(fs["extractor"], "model_path", "");
    net_cfg.num_keypoints          = readInt(fs["extractor"], "num_keypoints", 0);
    net_cfg.template_real_width_mm  = readDouble(fs["extractor"], "template_real_width_mm", 0.0);
    net_cfg.template_real_height_mm = readDouble(fs["extractor"], "template_real_height_mm", 0.0);

    // 输出
    bool visualize = readBool(fs["output"], "visualize", true);
    bool log_file  = readBool(fs["output"], "log_file", false);

    fs.release();

    // ========================================================================
    // ② 输出目录: output/compare_<输入目录名>
    // ========================================================================
    namespace fsp = std::filesystem;
    std::string folder = fsp::path(input_dir).filename().string();
    if (folder.empty()) folder = fsp::path(input_dir).parent_path().filename().string();
    std::string output_dir = "output/compare_" + (folder.empty() ? "sequence" : folder);
    if (visualize || log_file) fsp::create_directories(output_dir);
    std::cout << "输出目录: " << output_dir << std::endl;

    // ========================================================================
    // ③ 初始化组件: 输入序列 / YOLO / 关键点网络 (占位) / PnP
    // ========================================================================
    auto images = listImages(input_dir);
    if (images.empty()) {
        std::cerr << "输入目录无图像: " << input_dir << std::endl;
        return 1;
    }
    std::cout << "输入序列: " << images.size() << " 帧 (" << input_dir << ")" << std::endl;

    YoloDetector yolo(yolo_cfg);   // 构造失败抛异常 → 由外层 catch 报错退出

    // 特征点提取器: 按 method 分发, 统一为 ExtractFn 供帧循环调用
    ExtractFn run_extract;
    if (extract_method == "network") {
        auto extractor = std::make_unique<NetworkKeypointExtractor>(net_cfg);
        extractor->initialize();   // 占位: 打印警告, 不阻断
        run_extract = [p = extractor.release()](const cv::Mat& roi) {
            auto r = p->detect(roi);
            ExtractResult out;
            out.success = r.success;
            out.message = r.message;
            out.pts_2d = std::move(r.pts_2d);
            out.pts_3d = std::move(r.pts_3d);
            return out;
        };
    } else {
        if (extract_method != "sift" && extract_method != "orb") {
            std::cout << "未知 extractor.method '" << extract_method
                      << "', 回退到 sift" << std::endl;
            sift_cfg.method = "sift";
        }
        auto extractor = std::make_unique<SiftKeypointExtractor>(sift_cfg);
        if (!extractor->initialize()) {   // 真实方法: 模板加载失败直接退出
            std::cerr << "SIFT 提取器初始化失败, 退出" << std::endl;
            return 1;
        }
        run_extract = [p = extractor.release()](const cv::Mat& roi) {
            auto r = p->detect(roi);
            ExtractResult out;
            out.success = r.success;
            out.message = r.message;
            out.pts_2d = std::move(r.pts_2d);
            out.pts_3d = std::move(r.pts_3d);
            return out;
        };
    }

    MonoPnPSolver pnp_solver;

    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    // ========================================================================
    // ④ 逐帧处理: YOLO 框 (class0/class1) → 关键点 (占位) → PnP
    // ========================================================================
    int frame = 0, pose_ok = 0;
    double yolo_ms_total = 0.0, extract_ms_total = 0.0, pnp_ms_total = 0.0;
    auto steadyNow = []() { return std::chrono::steady_clock::now(); };
    auto elapsedMs = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    };
    std::ostringstream log_body;

    for (const auto& img_path : images) {
        if (g_stop_requested) break;
        ++frame;

        cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "[Frame " << frame << "] 读取失败: " << img_path.string() << std::endl;
            continue;
        }

        // ---- 1. YOLO 检测, 区分 class0 (整体) / class1 (中心) ----
        std::vector<Detection> dets;
        auto t_yolo = steadyNow();
        Status st = yolo.detect(img, dets);
        yolo_ms_total += elapsedMs(t_yolo);
        if (st != Status::Success) {
            std::cout << "[Frame " << frame << "] YOLO 推理失败" << std::endl;
            continue;
        }

        const Detection* det_c0 = bestDetection(dets, 0);
        const Detection* det_c1 = bestDetection(dets, 1);
        cv::Rect roi_c0 = det_c0 ? bboxToRoi(det_c0->bbox, img.size()) : cv::Rect();
        cv::Rect roi_c1 = det_c1 ? bboxToRoi(det_c1->bbox, img.size()) : cv::Rect();

        if (roi_c0.empty() && roi_c1.empty()) {
            std::cout << "[Frame " << frame << "] YOLO未检测到目标" << std::endl;
            continue;
        }

        // 主 ROI: 优先 class0 (整体), 无则 class1 (中心)
        // TODO: 真实网络若需要双 ROI (class0+class1 分别推理), 在此处扩展
        cv::Rect roi = !roi_c0.empty() ? roi_c0 : roi_c1;

        // ---- 2. 特征点提取 (SIFT / 网络占位) ----
        auto t_extract = steadyNow();
        ExtractResult kps = run_extract(img(roi));
        extract_ms_total += elapsedMs(t_extract);

        if (!kps.success) {
            std::cout << "[Frame " << frame << "] 关键点提取失败: " << kps.message << std::endl;
            continue;
        }

        // ROI 局部 → 全图坐标 (PnP 用全图内参)
        std::vector<cv::Point2f> pts_2d = kps.pts_2d;
        for (auto& p : pts_2d) { p.x += roi.x; p.y += roi.y; }

        // ---- 3. PnP ----
        auto t_pnp = steadyNow();
        PoseEstimate pose = pnp_solver.solve(pts_2d, kps.pts_3d, K);
        pnp_ms_total += elapsedMs(t_pnp);

        if (pose.success) ++pose_ok;
        std::ostringstream os;
        os << "[Frame " << frame << "] " << (pose.success ? "OK" : "PNP_FAIL")
           << "  n_kp=" << pts_2d.size()
           << "  roi=" << (roi == roi_c0 ? "class0" : "class1")
           << "  t(mm)=[" << pose.t.x() << ", " << pose.t.y() << ", " << pose.t.z() << "]";
        std::cout << os.str() << std::endl;
        log_body << os.str() << "\n";

        // ---- 4. 可视化: 检测框 + 关键点 + 三维轴 ----
        if (visualize) {
            cv::Mat vis = img.clone();
            if (!roi_c0.empty())
                cv::rectangle(vis, roi_c0, cv::Scalar(255, 0, 0), 2);   // class0 蓝
            if (!roi_c1.empty())
                cv::rectangle(vis, roi_c1, cv::Scalar(0, 255, 0), 2);   // class1 绿
            for (const auto& p : pts_2d)
                cv::drawMarker(vis, p, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 8, 2);
            if (pose.success) {
                cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
                    K(0, 0), K(0, 1), K(0, 2),
                    K(1, 0), K(1, 1), K(1, 2),
                    K(2, 0), K(2, 1), K(2, 2));
                cv::Mat rvec, R_cv = (cv::Mat_<double>(3, 3) <<
                    pose.R(0, 0), pose.R(0, 1), pose.R(0, 2),
                    pose.R(1, 0), pose.R(1, 1), pose.R(1, 2),
                    pose.R(2, 0), pose.R(2, 1), pose.R(2, 2));
                cv::Rodrigues(R_cv, rvec);
                cv::Mat tvec = (cv::Mat_<double>(3, 1) <<
                    pose.t(0), pose.t(1), pose.t(2));
                std::vector<cv::Point3d> axis = {{0, 0, 0}, {100, 0, 0}, {0, 100, 0}, {0, 0, 100}};
                std::vector<cv::Point2d> axis_px;
                cv::projectPoints(axis, rvec, tvec, K_cv, cv::Mat(), axis_px);
                if (axis_px.size() == 4) {
                    cv::line(vis, axis_px[0], axis_px[1], cv::Scalar(0, 0, 255), 3);
                    cv::line(vis, axis_px[0], axis_px[2], cv::Scalar(0, 255, 0), 3);
                    cv::line(vis, axis_px[0], axis_px[3], cv::Scalar(255, 0, 0), 3);
                }
            }
            utils::AsyncImageSaver::write(
                output_dir + "/compare_f" + std::to_string(frame) + ".png", vis);
        }
    }

    utils::AsyncImageSaver::flush();

    // ========================================================================
    // ⑤ 汇总统计
    // ========================================================================
    std::ostringstream summary;
    summary << "\n===== 运行汇总 (pose_compare) =====\n"
            << "处理帧数: " << frame
            << "  位姿成功: " << pose_ok << "\n";
    if (frame > 0) {
        summary << "YOLO 平均用时: " << yolo_ms_total / frame << " ms/帧\n"
                << "关键点提取平均用时: " << extract_ms_total / frame << " ms/帧\n"
                << "PnP 平均用时: " << pnp_ms_total / frame << " ms/帧\n";
    }
    std::cout << summary.str();

    if (log_file) {
        std::ofstream ofs(output_dir + "/compare_log.txt");
        if (ofs.is_open()) {
            ofs << "===== pose_compare 对比 pipeline =====\n"
                << "配置: " << config_path << "\n"
                << "输入目录: " << input_dir << "\n"
                << "相机: fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << "\n"
                << "====================================\n\n"
                << log_body.str() << summary.str();
        }
    }

    return 0;
}
