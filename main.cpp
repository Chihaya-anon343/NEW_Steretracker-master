/**
 * @file main.cpp
 * @brief GPNP 双目视觉跟踪器入口 —— YOLO检测 → 策略选择 → 位姿解算
 *
 * 运行: ./GPNP [config_path]
 * 配置: config/tracker_config.json
 */

#include "tracker/StereoTracker.hpp"
#include "detection/YoloRoiProvider.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    using namespace gpnp;

    // ========================================================================
    // ① 读取配置文件
    // ========================================================================
    std::string config_path = "config/tracker_config.json";
    if (argc >= 2) config_path = argv[1];

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开配置文件: " << config_path << std::endl;
        return 1;
    }

    // 相机内参 + 基线
    double fx = fs["camera"]["fx"], fy = fs["camera"]["fy"];
    double cx = fs["camera"]["cx"], cy = fs["camera"]["cy"];
    double baseline = fs["camera"]["baseline_mm"];

    // AKAZE 策略参数
    std::string template_path = fs["strategies"]["akaze_gpnp"]["template_path"];
    double tw = fs["strategies"]["akaze_gpnp"]["template_real_width_mm"];
    double th = fs["strategies"]["akaze_gpnp"]["template_real_height_mm"];
    double scale = fs["strategies"]["akaze_gpnp"]["scale"];
    int min_pts = fs["strategies"]["akaze_gpnp"]["gpnp_min_pts"];
    bool use_init_pnp = static_cast<int>(fs["strategies"]["akaze_gpnp"]["use_initial_pnp"]) != 0;

    // BinaryCorner 策略参数
    BinaryCornerExtractor::Config binary_cfg;
    binary_cfg.corners      = fs["strategies"]["binary_corner"]["corners"];
    binary_cfg.kernel_size   = fs["strategies"]["binary_corner"]["kernel_size"];
    binary_cfg.scale         = fs["strategies"]["binary_corner"]["corner_scale"];
    binary_cfg.target_size   = cv::Size(
        fs["strategies"]["binary_corner"]["target_width"],
        fs["strategies"]["binary_corner"]["target_height"]);
    binary_cfg.pixel_to_meter_scale = fs["strategies"]["binary_corner"]["pixel_to_meter_scale"];
    binary_cfg.roi_pad_pixels = fs["strategies"]["binary_corner"]["roi_pad_pixels"];
    binary_cfg.otsu_ratio     = fs["strategies"]["binary_corner"]["otsu_ratio"];
    std::string binary_template_dir = fs["strategies"]["binary_corner"]["template_dir"];

    // TinyTarget 策略参数
    TinyTargetExtractor::Config tiny_cfg;
    tiny_cfg.target_size   = cv::Size(
        fs["strategies"]["tiny_target"]["target_width"],
        fs["strategies"]["tiny_target"]["target_height"]);
    tiny_cfg.scale_factor  = fs["strategies"]["tiny_target"]["scale_factor"];
    tiny_cfg.square_size_m = fs["strategies"]["tiny_target"]["square_size_m"];
    tiny_cfg.roi_pad_pixels = fs["strategies"]["tiny_target"]["roi_pad_pixels"];
    std::string tiny_template_dir = fs["strategies"]["tiny_target"]["template_dir"];

    // YOLO 检测参数
    std::string model_path = fs["yolo"]["model_path"];
    float conf = fs["yolo"]["conf_threshold"];
    int target_cls = fs["yolo"]["target_class_id"];
    float expand = fs["yolo"]["roi_expand_ratio"];
    int min_roi = fs["yolo"]["roi_min_size"];

    // 策略选择阈值（从 JSON 读取，替代原硬编码常量）
    int akaze_min_area = fs["strategies"]["akaze_min_area"];
    int tiny_max_area  = fs["strategies"]["tiny_max_area"];

    // 双 ROI 配置（class 1 ROI 拓展像素 + AKAZE 提取参数）
    int dual_expand = 10;
    double dual_akaze_scale = 0.5;
    cv::FileNode dual_node = fs["strategies"]["dual_roi"];
    if (!dual_node.empty()) {
        dual_expand = dual_node["secondary_expand_pixels"];
        cv::FileNode ak_node = dual_node["akaze"];
        if (!ak_node.empty()) {
            dual_akaze_scale = ak_node["scale"];
        }
    }

    // 输入输出
    std::string left_path  = fs["input"]["left"];
    std::string right_path = fs["input"]["right"];
    bool visualize = static_cast<int>(fs["output"]["visualize"]) != 0;

    // 手动 ROI 配置（若 enabled=true 则跳过 YOLO，直接使用指定 ROI）
    bool use_manual_roi = false;
    RoiRect manual_rl, manual_rr;
    cv::FileNode manual_node = fs["manual_roi"];
    if (!manual_node.empty()) {
        int enabled = manual_node["enabled"];
        if (enabled) {
            use_manual_roi = true;
            manual_rl.x      = manual_node["left"]["x"];
            manual_rl.y      = manual_node["left"]["y"];
            manual_rl.width  = manual_node["left"]["width"];
            manual_rl.height = manual_node["left"]["height"];
            manual_rr.x      = manual_node["right"]["x"];
            manual_rr.y      = manual_node["right"]["y"];
            manual_rr.width  = manual_node["right"]["width"];
            manual_rr.height = manual_node["right"]["height"];
        }
    }

    fs.release();

    // ========================================================================
    // ② 构造输出目录（按图像名分类）
    // ========================================================================
    namespace fsp = std::filesystem;
    std::string img_name = fsp::path(left_path).stem().string();
    {
        auto pos = img_name.find(" - ");
        if (pos != std::string::npos) img_name = img_name.substr(0, pos);
    }
    std::string output_dir = "output/" + img_name;
    if (visualize) fsp::create_directories(output_dir);
    std::cout << "输出目录: " << output_dir << std::endl;

    // ========================================================================
    // ③ 初始化相机 + tracker
    // ========================================================================
    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;
    Eigen::Matrix3d R_rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rl(baseline, 0.0, 0.0);

    TrackerConfig tracker_cfg = makeTrackerConfig(scale, min_pts, use_init_pnp, tw, th,
                                                  akaze_min_area, tiny_max_area,
                                                  dual_expand, dual_akaze_scale);

    try {
        // 加载双目图像
        std::cout << "加载图像: " << left_path << " / " << right_path << std::endl;
        cv::Mat left_img  = cv::imread(left_path,  cv::IMREAD_COLOR);
        cv::Mat right_img = cv::imread(right_path, cv::IMREAD_COLOR);
        if (left_img.empty() || right_img.empty())
            throw std::runtime_error("无法读取输入图像");

        // 初始化 YOLO 检测器
        YoloRoiProvider yolo;
        YoloConfig yolo_cfg = makeYoloConfig(model_path, DeviceType::CPU, conf);
        yolo_cfg.target_class_id = target_cls;
        yolo_cfg.roi_expand_ratio = expand;
        RoiGenerator::Config roi_cfg{target_cls, expand, min_roi};
        bool yolo_ok = yolo.initialize(yolo_cfg, roi_cfg);

        // 初始化 StereoTracker（预创建全部 3 种策略，每帧仅切换指针）
        std::cout << "初始化 StereoTracker（预加载 3 种提取器）..." << std::endl;
        StereoTracker tracker(K, R_rl, t_rl, template_path, tracker_cfg,
                              binary_cfg, binary_template_dir,
                              tiny_cfg, tiny_template_dir);
        tracker.setOutputDir(output_dir);

        // ====================================================================
        // ④ 逐帧处理
        // ====================================================================
        for (int frame = 1; frame <= 2; ++frame) {
            std::cout << "\n===== 第 " << frame << " 帧 =====" << std::endl;

            // 获取 ROI：手动输入 > YOLO 检测
            RoiGroup lg, rg;
            if (use_manual_roi) {
                lg = RoiGroup{manual_rl, {}, false};
                rg = RoiGroup{manual_rr, {}, false};
                std::cout << "  手动 ROI: left=(" << manual_rl.x << "," << manual_rl.y << ","
                          << manual_rl.width << "," << manual_rl.height << "), right=("
                          << manual_rr.x << "," << manual_rr.y << "," << manual_rr.width
                          << "," << manual_rr.height << ")" << std::endl;
            } else if (yolo_ok) {
                std::tie(lg, rg) = yolo.detect(left_img, right_img);
                if (lg.is_dual) {
                    std::cout << "  双 ROI 模式: secondary=(" << lg.secondary.width
                              << "x" << lg.secondary.height << ")" << std::endl;
                }
            }

            // process() 内部自动完成：策略链选择 + ROI padding + 提取 + 位姿解算
            const RoiGroup* plg = lg.valid() ? &lg : nullptr;
            const RoiGroup* prg = rg.valid() ? &rg : nullptr;

            // 处理当前帧
            auto result = tracker.process(left_img, right_img, visualize, plg, prg);

            // 输出结果摘要
            std::cout << "  特征点: " << result.n_kp_left
                      << "  匹配: " << result.n_matched
                      << "  投影: " << result.n_projected
                      << "  模板: " << result.n_template_match;
            if (!result.disparity.empty()) {
                std::vector<double> abs_disp;
                for (double d : result.disparity) abs_disp.push_back(std::abs(d));
                std::cout << "  视差: " << computeMedian(std::move(abs_disp)) << "px";
            }
            std::cout << "  GPNP: " << (result.gpnp_success ? "成功" : "失败")
                      << "  耗时: " << result.total_time_ms() << "ms" << std::endl;
        }

        tracker.printLogs();

    } catch (const std::exception& e) {
        std::cerr << "致命错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
