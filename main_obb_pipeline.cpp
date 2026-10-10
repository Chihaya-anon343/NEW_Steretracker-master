/**
 * @file main_obb_pipeline.cpp
 * @brief 最小端到端 OBB 流水线（pipeline 分支）：
 *          YOLO-OBB 检测 → OBB 四边形角点提取 → MonoPnP 位姿解算
 *
 * 不含五状态分级 / 退化链 / warm-start / ESKF 等鲁棒性设计；
 * 输入为图像目录，逐帧输出位姿与可视化。
 * 详细输出（启动信息 + 逐帧行 + PnP 候选日志）写入 <out_dir>/obb_log.txt，
 * 终端仅打印末尾汇总。
 *
 * 用法（容器内）:
 *   ./cmake-build-debug/obb_pipeline config/obb_pipeline.json
 */

#include "common/Types.hpp"
#include "detection/IYoloDetector.hpp"
#include "feature/ObbCornerExtractor.hpp"
#include "pose/MonoPnPSolver.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Dense>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/// 角点索引 → 固定 BGR 色（HSV 色环均分, 与 Python corner_palette 一致）。
cv::Scalar cornerColor(int i, int n) {
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(i * 180 / n, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b& v = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(v[0], v[1], v[2]);
}

bool isImageFile(const fs::path& p) {
    static const std::string exts[] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    std::string e = p.extension().string();
    for (auto& x : exts) {
        if (e == x) return true;
    }
    return false;
}

gpnp::DeviceType parseDevice(const std::string& s) {
    if (s == "CPU") return gpnp::DeviceType::CPU;
    if (s == "CUDA") return gpnp::DeviceType::CUDA;
    return gpnp::DeviceType::Auto;
}

/// Eigen 3×3 → cv::Matx33d（无隐式转换, 手工搬运）。
cv::Matx33d toCvMatx(const Eigen::Matrix3d& m) {
    cv::Matx33d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = m(r, c);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cfg_path = (argc > 1) ? argv[1] : "config/obb_pipeline.json";
    cv::FileStorage fs_cfg(cfg_path, cv::FileStorage::READ);
    if (!fs_cfg.isOpened()) {
        std::cerr << "[obb_pipeline] 无法打开配置: " << cfg_path << std::endl;
        return 1;
    }

    // ---- 配置读取 ----
    gpnp::YoloConfig ycfg;
    ycfg.model_path = (std::string)fs_cfg["yolo"]["model_path"];
    ycfg.device = parseDevice((std::string)fs_cfg["yolo"]["device_type"]);
    ycfg.conf_threshold = (float)fs_cfg["yolo"]["conf_threshold"];
    ycfg.iou_threshold = (float)fs_cfg["yolo"]["iou_threshold"];
    ycfg.intra_op_threads = (int)fs_cfg["yolo"]["intra_op_threads"];
    ycfg.is_obb = true;   // OBB 模型

    gpnp::ObbCornerExtractor::Config ecfg;
    ecfg.template_dir = (std::string)fs_cfg["extractor"]["template_dir"];
    ecfg.norm_size = (int)fs_cfg["extractor"]["norm_size"];
    ecfg.morph_k = (int)fs_cfg["extractor"]["morph"];
    ecfg.denoise_k = (int)fs_cfg["extractor"]["denoise"];
    ecfg.poly_eps = (double)fs_cfg["extractor"]["poly_eps"];
    ecfg.tmpl_score = (double)fs_cfg["extractor"]["tmpl_score"];
    ecfg.corner_tau = (double)fs_cfg["extractor"]["corner_tau"];
    ecfg.snap_r = (double)fs_cfg["extractor"]["snap_r"];
    ecfg.min_area_frac = (double)fs_cfg["extractor"]["min_area_frac"];
    ecfg.min_side = (double)fs_cfg["extractor"]["min_side"];
    ecfg.template_real_size_mm = (double)fs_cfg["extractor"]["template_real_size_mm"];

    const double fx = (double)fs_cfg["camera"]["fx"];
    const double fy = (double)fs_cfg["camera"]["fy"];
    const double cx = (double)fs_cfg["camera"]["cx"];
    const double cy = (double)fs_cfg["camera"]["cy"];
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    const std::string in_dir = (std::string)fs_cfg["input"]["directory"];
    std::string out_dir = "output/obb_pipeline";
    if (!fs_cfg["output"]["directory"].empty()) {
        out_dir = (std::string)fs_cfg["output"]["directory"];
    }
    const bool visualize = (int)fs_cfg["output"]["visualize"] != 0;
    fs_cfg.release();

    // ---- 日志重定向: 详细输出 → <out_dir>/obb_log.txt, 终端只留汇总 ----
    // (含 MonoPnP 每候选 verbose 行; YoloDetector 的 cerr 初始化诊断不走 cout, 保持终端可见)
    fs::create_directories(out_dir);
    std::ofstream log_file(out_dir + "/obb_log.txt", std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "[obb_pipeline] 无法创建日志: " << out_dir << "/obb_log.txt" << std::endl;
        return 1;
    }
    std::streambuf* const old_cout_buf = std::cout.rdbuf(log_file.rdbuf());

    std::cout << "[obb_pipeline] model=" << ycfg.model_path
              << " templates=" << ecfg.template_dir
              << " real_size_mm=" << ecfg.template_real_size_mm
              << "\n[obb_pipeline] input=" << in_dir << " output=" << out_dir << std::endl;

    // ---- 组件构造 ----
    auto detector = gpnp::createYoloDetector(ycfg);   // .rknn 后缀 → RKNN NPU 后端
    gpnp::ObbCornerExtractor extractor(ecfg);
    if (!extractor.templatesReady()) {
        std::cerr << "[obb_pipeline] 模板加载失败: " << ecfg.template_dir << std::endl;
        return 1;
    }
    gpnp::MonoPnPSolver pnp;
    std::cout << "[obb_pipeline] 模板 " << extractor.numTemplates() << " 个, K=\n" << K << std::endl;

    // ---- 收集图像 ----
    std::vector<fs::path> images;
    for (const auto& e : fs::directory_iterator(in_dir)) {
        if (e.is_regular_file() && isImageFile(e.path())) images.push_back(e.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) {
        std::cout << "[obb_pipeline] 目录无图像: " << in_dir << std::endl;
        std::cout.rdbuf(old_cout_buf);   // 恢复后再走终端报错
        std::cerr << "[obb_pipeline] 目录无图像: " << in_dir << std::endl;
        return 1;
    }

    // ---- 帧循环 ----
    int n_frames = 0, n_detected = 0, n_target = 0, n_pose = 0;
    double t_det = 0, t_ext = 0, t_pnp = 0;
    double reproj_sum = 0;
    int reproj_n = 0;

    for (size_t fi = 0; fi < images.size(); ++fi) {
        const auto& path = images[fi];
        cv::Mat img = cv::imread(path.string());
        if (img.empty()) {
            std::cout << "[Frame " << fi << "] 读取失败: " << path.string() << "\n";
            continue;
        }
        ++n_frames;

        // 1) YOLO-OBB 检测
        std::vector<gpnp::ObbDetection> obbs;
        auto t0 = std::chrono::steady_clock::now();
        gpnp::Status st = detector->detectObb(img, obbs);
        auto t1 = std::chrono::steady_clock::now();
        t_det += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (st != gpnp::Status::Success || obbs.empty()) {
            std::cout << "[Frame " << fi << "] DETECT_FAIL (" << path.filename().string() << ")\n";
            continue;
        }
        ++n_detected;

        // 类别互斥过滤: 同图同时检出 class0/class1 时仅传 class0
        {
            bool has0 = false;
            for (const auto& d : obbs) has0 |= (d.class_id == 0);
            if (has0) {
                obbs.erase(std::remove_if(obbs.begin(), obbs.end(),
                                          [](const gpnp::ObbDetection& d) {
                                              return d.class_id != 0;
                                          }),
                           obbs.end());
            }
        }

        // 2) 逐候选角点提取, 首个靶标即用（NMS 输出已按置信度降序）
        gpnp::ObbCornerExtractor::Result res;
        bool got_target = false;
        float used_conf = 0.f;
        t0 = std::chrono::steady_clock::now();
        for (const auto& d : obbs) {
            gpnp::ObbCornerExtractor::Result r = extractor.extract(img, d.quad);
            if (r.is_target && r.success) {
                res = r;
                got_target = true;
                used_conf = d.confidence;
                break;
            }
        }
        t1 = std::chrono::steady_clock::now();
        t_ext += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!got_target) {
            std::cout << "[Frame " << fi << "] NO_TARGET (cands=" << obbs.size() << ")\n";
            continue;
        }
        ++n_target;

        // 3) PnP
        gpnp::PoseEstimate pose;
        double reproj = -1.0;
        if (!res.pts_3d.empty()) {
            t0 = std::chrono::steady_clock::now();
            pose = pnp.solve(res.corners, res.pts_3d, K);
            t1 = std::chrono::steady_clock::now();
            t_pnp += std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (pose.success) {
                // 重投影误差（全点平均）
                cv::Matx33d Kc = toCvMatx(K);
                cv::Mat rvec, tvec, R64;
                R64 = cv::Mat(toCvMatx(pose.R));
                cv::Rodrigues(R64, rvec);
                tvec = cv::Mat(cv::Vec3d(pose.t.x(), pose.t.y(), pose.t.z()));
                std::vector<cv::Point3f> obj;
                obj.reserve(res.pts_3d.size());
                for (const auto& p3 : res.pts_3d) {
                    obj.emplace_back((float)p3.x(), (float)p3.y(), (float)p3.z());
                }
                std::vector<cv::Point2f> proj;
                cv::projectPoints(obj, rvec, tvec, Kc, cv::noArray(), proj);
                double s = 0;
                for (size_t k = 0; k < proj.size(); ++k) {
                    s += cv::norm(proj[k] - res.corners[k]);
                }
                reproj = s / proj.size();
                reproj_sum += reproj;
                ++reproj_n;
            }
        }

        // ---- 输出 ----
        ++n_pose;   // 帧通过标志（含 PnP 失败情形, 下方单独标注）
        std::cout << "[Frame " << fi << "] " << path.filename().string()
                  << " conf=" << used_conf << " score=" << res.score
                  << " corners=" << res.corners.size()
                  << " (dp" << res.n_dp << "+keep" << res.n_keep << "+snap" << res.n_snap << ")"
                  << " extract=" << res.extract_ms << "ms";
        if (!res.pts_3d.empty()) {
            if (pose.success) {
                cv::Mat rvec, R64;
                R64 = cv::Mat(toCvMatx(pose.R));
                cv::Rodrigues(R64, rvec);
                double angle = cv::norm(rvec) * 180.0 / CV_PI;
                std::cout << " | PnP OK t(mm)=[" << pose.t.x() << ", " << pose.t.y() << ", "
                          << pose.t.z() << "] |t|=" << pose.t.norm()
                          << " rvec_angle=" << angle << "deg"
                          << " reproj=" << reproj << "px\n";
            } else {
                std::cout << " | PNP_FAIL\n";
            }
        } else {
            std::cout << " | NO_3D (template_real_size_mm 未配置?)\n";
        }

        // ---- 可视化 ----
        if (visualize) {
            cv::Mat vis = img.clone();

            // 检测四边形（绿）
            std::vector<cv::Point> quad_pts;
            for (const auto& p : res.quad) quad_pts.emplace_back(cvRound(p.x), cvRound(p.y));
            cv::polylines(vis, quad_pts, true, cv::Scalar(0, 200, 0), 2, cv::LINE_AA);

            // 反投影外接方形（黄）
            std::vector<cv::Point> rect_pts;
            for (const auto& p : res.rect_corners) rect_pts.emplace_back(cvRound(p.x), cvRound(p.y));
            cv::polylines(vis, rect_pts, true, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

            // 10 角点（索引着色 + 编号）
            const int n = std::max<size_t>(res.corners.size(), 1);
            for (size_t k = 0; k < res.corners.size(); ++k) {
                cv::Scalar col = cornerColor((int)k, (int)n);
                cv::Point c(cvRound(res.corners[k].x), cvRound(res.corners[k].y));
                cv::circle(vis, c, 5, col, -1, cv::LINE_AA);
                cv::putText(vis, std::to_string(k), c + cv::Point(7, -7),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
            }

            // 位姿坐标轴（PnP 成功时）
            if (pose.success) {
                cv::Mat rvec, tvec, R64;
                R64 = cv::Mat(toCvMatx(pose.R));
                cv::Rodrigues(R64, rvec);
                tvec = cv::Mat(cv::Vec3d(pose.t.x(), pose.t.y(), pose.t.z()));
                const double L = ecfg.template_real_size_mm * 0.2;   // 轴长 = 20% 靶标尺寸
                std::vector<cv::Point3f> axes = {{0, 0, 0},
                                                 {(float)L, 0, 0},
                                                 {0, (float)L, 0},
                                                 {0, 0, (float)L}};
                std::vector<cv::Point2f> ppts;
                cv::projectPoints(axes, rvec, tvec, toCvMatx(K), cv::noArray(), ppts);
                cv::line(vis, ppts[0], ppts[1], cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                cv::line(vis, ppts[0], ppts[2], cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
                cv::line(vis, ppts[0], ppts[3], cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
            }

            std::ostringstream oss;
            oss << out_dir << "/obb_f" << fi << ".png";
            cv::imwrite(oss.str(), vis);
        }
    }

    // ---- 汇总（终端 + 日志双写）----
    std::cout.rdbuf(old_cout_buf);   // 恢复终端输出
    std::ostringstream sum;
    sum << "\n===== obb_pipeline 汇总 =====\n"
        << "帧数: " << n_frames
        << " | 检出: " << n_detected
        << " | 靶标: " << n_target
        << " | 输出位姿帧: " << n_pose << "\n"
        << "平均耗时(ms/帧): 检测 " << (n_frames ? t_det / n_frames : 0.0)
        << " + 提取 " << (n_frames ? t_ext / n_frames : 0.0)
        << " + PnP " << (n_frames ? t_pnp / n_frames : 0.0) << "\n"
        << "平均重投影误差: "
        << (reproj_n ? reproj_sum / reproj_n : -1.0) << " px"
        << " (" << reproj_n << " 帧)\n"
        << "明细日志: " << out_dir << "/obb_log.txt\n";
    std::cout << sum.str();
    log_file << sum.str();
    log_file.close();
    return 0;
}
