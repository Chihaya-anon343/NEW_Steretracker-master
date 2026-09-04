/**
 * @file main.cpp
 * @brief GPNP 双目视觉跟踪器入口 —— YOLO检测 → 策略选择 → 位姿解算
 *
 * 运行: ./GPNP [config_path]
 * 配置: config/tracker_config.json
 */

#include "tracker/StereoTracker.hpp"
#include "tracker/MonoTracker.hpp"
#include "detection/YoloRoiProvider.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "utils/AsyncImageSaver.hpp"

// 输入系统 (Phase 1)
#include "input/InputProvider.hpp"
#include "input/InputConfig.hpp"
#include "input/FileStereoSource.hpp"
#include "input/SimulatedSensors.hpp"

// ESKF 多源融合 (可选)
#include "fusion/EskfFusionManager.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <csignal>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <fstream>

namespace {
// Phase 3.2: 优雅退出 — Ctrl+C (SIGINT) / kill (SIGTERM) 置位, 帧循环检测后收尾
volatile std::sig_atomic_t g_stop_requested = 0;
void handleSignal(int) {
    g_stop_requested = 1;
}

// ---- 配置读取辅助 (ESKF 节) ----
double readDouble(const cv::FileNode& n, const char* key, double def) {
    cv::FileNode v = n[key];
    return v.empty() ? def : static_cast<double>(v);
}

/// 读取 3×3 矩阵 [[r00,r01,r02],...]; 失败 (缺省/尺寸不符) 保持 out 原值并返回 false
bool readMat3(const cv::FileNode& node, Eigen::Matrix3d& out) {
    if (node.empty() || node.size() != 3) return false;
    for (int i = 0; i < 3; ++i) {
        cv::FileNode row = node[i];
        if (row.empty() || row.size() != 3) return false;
        for (int j = 0; j < 3; ++j) {
            cv::FileNode e = row[j];
            if (e.empty()) return false;
            out(i, j) = static_cast<double>(e);
        }
    }
    return true;
}

/// 读取 3 维向量 [x,y,z]; 失败保持 out 原值并返回 false
bool readVec3(const cv::FileNode& node, Eigen::Vector3d& out) {
    if (node.empty() || node.size() != 3) return false;
    for (int i = 0; i < 3; ++i) {
        cv::FileNode e = node[i];
        if (e.empty()) return false;
        out(i) = static_cast<double>(e);
    }
    return true;
}
} // namespace

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
    binary_cfg.pixel_to_meter_scale_class0 = fs["strategies"]["binary_corner"]["pixel_to_meter_scale_class0"];
    binary_cfg.pixel_to_meter_scale_class1 = fs["strategies"]["binary_corner"]["pixel_to_meter_scale_class1"];
    binary_cfg.roi_pad_pixels = fs["strategies"]["binary_corner"]["roi_pad_pixels"];
    binary_cfg.otsu_ratio     = fs["strategies"]["binary_corner"]["otsu_ratio"];
    std::string binary_template_dir = fs["strategies"]["binary_corner"]["template_dir"];

    // TinyTarget 策略参数
    TinyTargetExtractor::Config tiny_cfg;
    tiny_cfg.target_size   = cv::Size(
        fs["strategies"]["tiny_target"]["target_width"],
        fs["strategies"]["tiny_target"]["target_height"]);
    tiny_cfg.scale_factor  = fs["strategies"]["tiny_target"]["scale_factor"];
    tiny_cfg.square_size_m_class0 = fs["strategies"]["tiny_target"]["square_size_m_class0"];
    tiny_cfg.square_size_m_class1 = fs["strategies"]["tiny_target"]["square_size_m_class1"];
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
    int dual_trigger_area = fs["strategies"]["dual_trigger_area"];
    bool stereo_mono_fallback = static_cast<int>(fs["strategies"]["stereo_mono_fallback"]) != 0;

    // 近距离回退配置（class0 丢失时用 class1）
    RoiGenerator::CloseRangeConfig close_range_cfg;
    int akaze_min_area_class1 = 0;
    int tiny_max_area_class1  = 0;
    cv::FileNode cr_node = fs["strategies"]["close_range"];
    if (!cr_node.empty()) {
        close_range_cfg.enabled          = static_cast<int>(cr_node["enabled"]) != 0;
        close_range_cfg.class1_min_area  = cr_node["class1_min_area"];
        close_range_cfg.roi_expand_ratio = cr_node["roi_expand_ratio"];
        close_range_cfg.min_expand_pixels = cr_node["min_expand_pixels"];
        akaze_min_area_class1 = static_cast<int>(cr_node["akaze_min_area"]);
        tiny_max_area_class1  = static_cast<int>(cr_node["tiny_max_area"]);
        close_range_cfg.akaze_min_area = akaze_min_area_class1;
        close_range_cfg.tiny_max_area  = tiny_max_area_class1;
    }

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

    // ========================================================================
    // 运行模式 — normal (InputProvider+简洁) / debug (旧路径+详细)
    // ========================================================================
    std::string run_mode = "normal";
    {
        cv::FileNode mn = fs["mode"];
        if (!mn.empty()) run_mode = static_cast<std::string>(mn);
    }
    bool verbose_console  = (run_mode == "debug");
    bool visualize_detailed = verbose_console;  ///< Debug 模式生成 per-strategy 面板，与日志捕获独立
    bool use_input_system = (run_mode == "normal");

    // 输入输出
    std::string left_path  = fs["input"]["left"];
    std::string right_path = fs["input"]["right"];
    bool visualize = static_cast<int>(fs["output"]["visualize"]) != 0;
    bool log_file   = false;
    {
        cv::FileNode lf = fs["output"]["log_file"];
        if (!lf.empty()) log_file = static_cast<int>(lf) != 0;
    }

    // 手动 ROI 配置 — 仅 debug 模式生效
    bool use_manual_roi = false;
    RoiRect manual_rl, manual_rr;
    if (run_mode == "debug") {
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
    }

    // 单目模式配置
    bool mono_mode = false;
    cv::FileNode mono_node = fs["mono_mode"];
    if (!mono_node.empty()) {
        mono_mode = static_cast<int>(mono_node) != 0;
    }

    // ========================================================================
    // ①-b 解析 input_system 配置（normal 模式用）
    // ========================================================================
    input::InputSystemConfig input_sys_cfg;
    int max_frames = (run_mode == "debug") ? 2 : 0;

    cv::FileNode input_sys_node = fs["input_system"];
    if (!input_sys_node.empty()) {
        // Phase 3.1: 线程化采集配置 (Camera 类型自动启用, 可不配置)
        input_sys_cfg.use_threaded_capture =
            static_cast<int>(input_sys_node["use_threaded_capture"]) != 0;
        {
            cv::FileNode rc = input_sys_node["ring_capacity"];
            if (!rc.empty()) input_sys_cfg.ring_capacity = static_cast<int>(rc);
        }

        cv::FileNode img_node = input_sys_node["image"];
        if (!img_node.empty()) {
            std::string img_type = img_node["type"];
            if (img_type == "file") {
                input_sys_cfg.image.type = input::ImageSourceType::File;
                input_sys_cfg.image.left_path = static_cast<std::string>(img_node["left_path"]);
                input_sys_cfg.image.right_path = static_cast<std::string>(img_node["right_path"]);
            } else if (img_type == "directory") {
                input_sys_cfg.image.type = input::ImageSourceType::Directory;
                input_sys_cfg.image.directory_path = static_cast<std::string>(img_node["directory_path"]);
                input_sys_cfg.image.left_pattern = static_cast<std::string>(img_node["left_pattern"]);
                input_sys_cfg.image.right_pattern = static_cast<std::string>(img_node["right_pattern"]);
                if (input_sys_cfg.image.left_pattern.empty()) input_sys_cfg.image.left_pattern = "left";
                if (input_sys_cfg.image.right_pattern.empty()) input_sys_cfg.image.right_pattern = "right";
            } else if (img_type == "sequence") {
                input_sys_cfg.image.type = input::ImageSourceType::Sequence;
                input_sys_cfg.image.directory_path = static_cast<std::string>(img_node["directory_path"]);
                input_sys_cfg.image.sequence_pattern = static_cast<std::string>(img_node["sequence_pattern"]);
                if (input_sys_cfg.image.sequence_pattern.empty()) input_sys_cfg.image.sequence_pattern = "frame";
            } else if (img_type == "camera") {
                input_sys_cfg.image.type = input::ImageSourceType::Camera;
                input_sys_cfg.image.camera_devices = static_cast<std::string>(img_node["camera_devices"]);
                if (input_sys_cfg.image.camera_devices.empty()) input_sys_cfg.image.camera_devices = "0";
                input_sys_cfg.image.target_fps = 0.0;
                {
                    cv::FileNode tf = img_node["target_fps"];
                    if (!tf.empty()) input_sys_cfg.image.target_fps = static_cast<double>(tf);
                }
            }
        }
        if (use_input_system) {
            max_frames = static_cast<int>(input_sys_node["max_frames"]);
            if (max_frames <= 0) max_frames = 999999;
        }
    }

    // ========================================================================
    // ①-c ESKF 融合配置 (可选, 默认关闭) + 合成 IMU/雷达源配置
    // ========================================================================
    fusion::EskfFusionConfig eskf_cfg;
    input::SimulatedImuConfig sim_imu_cfg;
    input::SimulatedRadarConfig sim_radar_cfg;

    cv::FileNode ek = fs["eskf"];
    if (!ek.empty()) {
        eskf_cfg.enabled = static_cast<int>(ek["enabled"]) != 0;
        if (eskf_cfg.enabled) {
            eskf_cfg.imu_rate_hz   = readDouble(ek, "imu_rate_hz", 200.0);
            eskf_cfg.radar_rate_hz = readDouble(ek, "radar_rate_hz", 20.0);
            eskf_cfg.max_imu_gap_s = readDouble(ek, "max_imu_gap_s", 0.1);
            eskf_cfg.max_cam_gap_s = readDouble(ek, "max_cam_gap_s", 1.0);

            // 反向传播 (延迟测量) 参数
            eskf_cfg.backprop_window_s = readDouble(ek, "backprop_window_s", 0.2);
            eskf_cfg.state_hist_hz = static_cast<int>(readDouble(ek, "state_hist_hz", 100.0));
            eskf_cfg.max_output_age_s = readDouble(ek, "max_output_age_s", 0.5);
            {
                cv::FileNode lf = ek["latency_fallback"];
                std::string lf_str = lf.empty() ? "inflate" : static_cast<std::string>(lf);
                eskf_cfg.latency_fallback =
                    (lf_str == "reject") ? fusion::LatencyFallback::Reject
                                         : fusion::LatencyFallback::Inflate;
            }
            eskf_cfg.threaded = static_cast<int>(ek["threaded"]) != 0;

            cv::FileNode nz = ek["noise"];
            if (!nz.empty()) {
                eskf_cfg.params.sigma_acc       = readDouble(nz, "sigma_acc", 0.3);
                eskf_cfg.params.sigma_gyro      = readDouble(nz, "sigma_gyro", 0.02);
                eskf_cfg.params.sigma_acc_bias  = readDouble(nz, "sigma_acc_bias", 0.01);
                eskf_cfg.params.sigma_gyro_bias = readDouble(nz, "sigma_gyro_bias", 0.001);
                eskf_cfg.params.sigma_pos_rw    = readDouble(nz, "sigma_pos_rw", 0.5);
                eskf_cfg.params.cam_pos_noise   = readDouble(nz, "cam_pos_noise", 0.1);
                eskf_cfg.params.cam_rot_noise   = readDouble(nz, "cam_rot_noise", 1.0);
                eskf_cfg.params.use_cam_z_noise_decoupling =
                    static_cast<int>(nz["use_cam_z_noise_decoupling"]) != 0;
                eskf_cfg.params.cam_pos_z_noise = readDouble(nz, "cam_pos_z_noise", 5.0);
                eskf_cfg.params.radar_alt_noise = readDouble(nz, "radar_alt_noise", 0.30);
            }
            cv::FileNode ng = ek["gravity"];
            if (!ng.empty() && ng.size() == 3) {
                Eigen::Vector3d g;
                bool ok = true;
                for (int i = 0; i < 3; ++i) {
                    if (ng[i].empty()) { ok = false; break; }
                    g(i) = static_cast<double>(ng[i]);
                }
                if (ok) eskf_cfg.params.gravity = g;
            }
            readMat3(ek["R_imu_cam"], eskf_cfg.params.R_imu_cam);      // 失败保持默认 (I)
            readVec3(ek["p_imu_in_cam"], eskf_cfg.params.p_imu_in_cam); // 失败保持默认 (0)
            readMat3(ek["R_template_world"], eskf_cfg.R_template_world);

            cv::FileNode ni = ek["init_std"];
            if (!ni.empty()) {
                eskf_cfg.init_std_p  = readDouble(ni, "p", 1.0);
                eskf_cfg.init_std_v  = readDouble(ni, "v", 1.0);
                eskf_cfg.init_std_q  = readDouble(ni, "q", 0.1);
                eskf_cfg.init_std_ba = readDouble(ni, "ba", 0.1);
                eskf_cfg.init_std_bg = readDouble(ni, "bg", 0.01);
            }
        }
    }

    // 合成数据源: input_system.imu.type == "simulated" / altimeter.type == "simulated"
    if (!input_sys_node.empty()) {
        cv::FileNode inode = input_sys_node["imu"];
        if (!inode.empty()
            && static_cast<std::string>(inode["type"]) == "simulated") {
            sim_imu_cfg.enabled    = true;
            sim_imu_cfg.rate_hz    = readDouble(inode, "rate_hz", 200.0);
            sim_imu_cfg.sigma_acc  = readDouble(inode, "sigma_acc", 0.1);
            sim_imu_cfg.sigma_gyro = readDouble(inode, "sigma_gyro", 0.005);
            readVec3(inode["bias_acc"], sim_imu_cfg.bias_acc);
            readVec3(inode["bias_gyro"], sim_imu_cfg.bias_gyro);
        }
        cv::FileNode anode = input_sys_node["altimeter"];
        if (!anode.empty()
            && static_cast<std::string>(anode["type"]) == "simulated") {
            sim_radar_cfg.enabled = true;
            sim_radar_cfg.rate_hz = readDouble(anode, "rate_hz", 20.0);
            sim_radar_cfg.noise_m = readDouble(anode, "noise_m", 0.30);
            sim_radar_cfg.initial_height =
                readDouble(anode, "initial_height", 10.0);
            sim_radar_cfg.inject_jump_every_s =
                readDouble(anode, "inject_jump_every_s", 0.0);
            sim_radar_cfg.jump_m = readDouble(anode, "jump_m", 50.0);
        }
    }

    // 时序连贯性配置 (temporal, 可选; 缺省值见 TemporalConfig 定义)
    TemporalConfig temporal_cfg;
    {
        cv::FileNode tn = fs["temporal"];
        if (!tn.empty()) {
            cv::FileNode en = tn["enabled"];
            if (!en.empty()) temporal_cfg.enabled = static_cast<int>(en) != 0;

            cv::FileNode pcn = tn["pose_chain"];
            if (!pcn.empty()) {
                cv::FileNode v = pcn["max_cache_age_frames"];
                if (!v.empty()) temporal_cfg.max_cache_age_frames = static_cast<int>(v);
                v = pcn["tie_epsilon_px"];
                if (!v.empty()) temporal_cfg.tie_epsilon_px = static_cast<double>(v);
            }

            cv::FileNode mg = tn["motion_gate"];
            if (!mg.empty()) {
                cv::FileNode v = mg["max_trans_ratio"];
                if (!v.empty()) temporal_cfg.max_trans_ratio = static_cast<double>(v);
                v = mg["max_rot_deg"];
                if (!v.empty()) temporal_cfg.max_rot_deg = static_cast<double>(v);
                v = mg["switch_margin"];
                if (!v.empty()) temporal_cfg.switch_margin = static_cast<double>(v);
            }

            cv::FileNode stn = tn["stickiness"];
            if (!stn.empty()) {
                cv::FileNode v = stn["hold_frames"];
                if (!v.empty()) temporal_cfg.hold_frames = static_cast<int>(v);
                v = stn["hysteresis_up"];
                if (!v.empty()) temporal_cfg.hysteresis_up = static_cast<double>(v);
                v = stn["hysteresis_down"];
                if (!v.empty()) temporal_cfg.hysteresis_down = static_cast<double>(v);
                v = stn["locked_fail_limit"];
                if (!v.empty()) temporal_cfg.locked_fail_limit = static_cast<int>(v);
            }

            cv::FileNode v = tn["area_ema_alpha"];
            if (!v.empty()) temporal_cfg.area_ema_alpha = static_cast<double>(v);
        }
    }

    fs.release();

    // ========================================================================
    // 日志文件捕获 — normal 模式 + log_file 时重定向 cout 到字符串流
    // ========================================================================
    std::ostringstream log_body;
    std::streambuf* saved_cout = nullptr;
    bool capture_log = log_file && !verbose_console;   // 仅 normal 模式（非 verbose）
    if (capture_log) {
        saved_cout = std::cout.rdbuf(log_body.rdbuf());  // cout → 日志；保存原始终端缓冲区
        verbose_console = true;                        // 强制 verbose 以便捕获详细处理消息
    }

    // ========================================================================
    // ② 构造输出目录
    // ========================================================================
    namespace fsp = std::filesystem;
    std::string output_dir;
    if (use_input_system) {
        // 从实际输入源目录名派生
        std::string src_dir = input_sys_cfg.image.directory_path;
        if (!src_dir.empty()) {
            std::string folder = fsp::path(src_dir).filename().string();
            if (folder.empty()) folder = fsp::path(src_dir).parent_path().filename().string();
            output_dir = "output/" + (folder.empty() ? "sequence" : folder);
        } else {
            output_dir = "output/sequence";
        }
    } else {
        std::string img_name = fsp::path(left_path).stem().string();
        {
            auto pos = img_name.find(" - ");
            if (pos != std::string::npos) img_name = img_name.substr(0, pos);
        }
        output_dir = "output/" + img_name;
    }
    if (visualize || log_file) fsp::create_directories(output_dir);
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
                                                  akaze_min_area_class1, tiny_max_area_class1,
                                                  dual_expand, dual_akaze_scale);
    tracker_cfg.temporal = temporal_cfg;

    try {
        // ====================================================================
        // 图像加载 —— 新版 InputProvider 或 旧版 cv::imread
        // ====================================================================
        input::InputProvider input_provider;
        cv::Mat left_img, right_img;  // 旧版路径使用

        if (use_input_system) {
            std::cout << "[输入系统] 初始化 InputProvider..." << std::endl;
            if (!input_provider.initialize(input_sys_cfg))
                throw std::runtime_error("InputProvider 初始化失败");
            std::cout << "[输入系统] 就绪, 最大帧数: " << max_frames << std::endl;
        } else {
            // 旧版路径：直接从路径加载图像
            std::cout << "加载图像: " << left_path << " / " << right_path << std::endl;
            left_img  = cv::imread(left_path,  cv::IMREAD_COLOR);
            right_img = cv::imread(right_path, cv::IMREAD_COLOR);
            if (left_img.empty() || right_img.empty())
                throw std::runtime_error("无法读取输入图像");
        }

        // ====================================================================
        // 初始化 YOLO — 手动 ROI 时跳过以节省加载时间
        // ====================================================================
        YoloRoiProvider yolo;
        bool yolo_ok = false;

        if (!use_manual_roi) {
            YoloConfig yolo_cfg = makeYoloConfig(model_path, DeviceType::CPU, conf);
            yolo_cfg.target_class_id = target_cls;
            yolo_cfg.roi_expand_ratio = expand;
            RoiGenerator::Config roi_cfg{target_cls, expand, min_roi, dual_trigger_area};
            yolo_ok = yolo.initialize(yolo_cfg, roi_cfg);
            if (yolo_ok) yolo.setCloseRangeConfig(close_range_cfg);
        }

        // ====================================================================
        // 初始化 Tracker（单目/双目 分支）
        // ====================================================================

        std::unique_ptr<TrackerBase> tracker;

        if (mono_mode) {
            std::cout << "初始化 MonoTracker（单目模式）..." << std::endl;
            auto t = std::make_unique<MonoTracker>(K, template_path, tracker_cfg,
                                                    binary_cfg, binary_template_dir,
                                                    tiny_cfg, tiny_template_dir);
            t->setOutputDir(output_dir);
            t->setVerboseConsole(verbose_console);
            t->setVisualizeDetailed(visualize_detailed);
            tracker = std::move(t);
            std::cout << "单目模式已启用（仅左图）" << std::endl;
        } else {
            std::cout << "初始化 StereoTracker（双目模式）..." << std::endl;
            auto t = std::make_unique<StereoTracker>(K, R_rl, t_rl, template_path, tracker_cfg,
                                                      binary_cfg, binary_template_dir,
                                                      tiny_cfg, tiny_template_dir);
            t->setOutputDir(output_dir);
            t->setVerboseConsole(verbose_console);
            t->setVisualizeDetailed(visualize_detailed);
            tracker = std::move(t);
        }

        // ====================================================================
        // ③-b ESKF 融合 + 合成数据源 (仅 normal 模式; debug 固定2帧无时间序列)
        // ====================================================================
        fusion::EskfFusionManager fusion(eskf_cfg);
        input::SimulatedImu   sim_imu(sim_imu_cfg);
        input::SimulatedRadar sim_radar(sim_radar_cfg);
        bool fusion_active = eskf_cfg.enabled && use_input_system;
        if (eskf_cfg.enabled && !use_input_system) {
            std::cerr << "[ESKF] debug 模式无时间序列, 融合已禁用 (eskf.enabled 忽略)"
                      << std::endl;
        }
        if (fusion_active && eskf_cfg.threaded) {
            fusion.start();
            std::cout << "[ESKF] 融合工作线程已启动 (异步消费)" << std::endl;
        }
        // 合成源真值参考 (最近一帧相机位姿)
        Eigen::Matrix3d last_R_cam_w = Eigen::Matrix3d::Identity();
        double last_cam_height = sim_radar_cfg.initial_height;

        // ====================================================================
        // ④ 逐帧处理
        // ====================================================================
        // Phase 3.2: 位姿成功帧计数 (汇总统计用)
        int processed_ok = 0;

        // 终端每帧一行输出。capture_log 时 cout 被重定向到日志，这里直写原始终端缓冲区
        auto termLine = [&](const std::string& s) {
            if (capture_log && saved_cout) {
                std::string t = s + '\n';
                saved_cout->sputn(t.data(), t.size());
            } else {
                std::cout << s << std::endl;
            }
        };

        // 处理单帧: YOLO/ROI → 策略 → PnP; 返回 PipelineResult (失败帧 success=false)
        auto processFrame = [&](int frame, const cv::Mat& L, const cv::Mat& R) -> PipelineResult {
            PipelineResult result;

            if (mono_mode) {
                auto* mt = static_cast<MonoTracker*>(tracker.get());
                RoiGroup left_group;
                if (use_manual_roi) {
                    left_group = RoiGroup{manual_rl, {}, false};
                } else if (yolo_ok) {
                    left_group = yolo.detectMono(L);
                    if (!left_group.valid()) {
                        tracker->onFrameSkipped();   // skip 帧也让位姿 seed 老化
                        termLine("[Frame " + std::to_string(frame) + "] YOLO未检测到目标");
                        return result;
                    }
                }
                tracker->setFrameNumber(frame);
                result = mt->process(L, visualize, &left_group);
            } else {
                auto* st = static_cast<StereoTracker*>(tracker.get());
                RoiGroup lg, rg;
                if (use_manual_roi) {
                    lg = RoiGroup{manual_rl, {}, false};
                    rg = RoiGroup{manual_rr, {}, false};
                    if (verbose_console) {
                        std::cout << "  手动 ROI: left=(" << manual_rl.x << "," << manual_rl.y << ","
                                  << manual_rl.width << "," << manual_rl.height << "), right=("
                                  << manual_rr.x << "," << manual_rr.y << "," << manual_rr.width
                                  << "," << manual_rr.height << ")" << std::endl;
                    }
                } else if (yolo_ok) {
                    std::tie(lg, rg) = yolo.detect(L, R);
                    if (!lg.valid() && !rg.valid()) {
                        tracker->onFrameSkipped();   // skip 帧也让位姿 seed 老化
                        termLine("[Frame " + std::to_string(frame) + "] YOLO未检测到目标");
                        return result;
                    }
                    if (lg.is_dual && verbose_console) {
                        std::cout << "  双 ROI 模式: secondary=(" << lg.secondary.width
                                  << "x" << lg.secondary.height << ")" << std::endl;
                    }
                }
                tracker->setFrameNumber(frame);
                if (lg.valid() && rg.valid()) {
                    // 双侧检测 → 双目
                    result = st->process(L, R, visualize, &lg, &rg);
                } else if (stereo_mono_fallback && (lg.valid() || rg.valid())) {
                    // 单侧检测 → 单目降级（需 stereo_mono_fallback=true）
                    result = st->processMono(lg.valid() ? L : R, visualize,
                                              lg.valid() ? &lg : &rg);
                } else {
                    tracker->onFrameSkipped();   // skip 帧也让位姿 seed 老化
                    termLine("[Frame " + std::to_string(frame) + "] YOLO未检测到目标");
                    return result;
                }
            }

            // ---- 详细输出（verbose / capture_log 时进日志）----
            if (result.success) ++processed_ok;   // 汇总统计
            if (verbose_console) {
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
                          << "  耗时: " << result.total_time_ms() << "ms";
                if (result.success) {
                    std::cout << "  PnP t(mm)=[" << result.t.x() << ", "
                              << result.t.y() << ", " << result.t.z() << "]";
                }
                std::cout << std::endl;
            }

            return result;
        };

        if (use_input_system) {
            // Phase 3.2: 注册优雅退出信号 — Ctrl+C 后循环自然收尾 (汇总+日志落盘)
            std::signal(SIGINT, handleSignal);
            std::signal(SIGTERM, handleSignal);

            // 新版输入系统路径 —— 数据驱动的帧循环
            int frame = 0;
            input::SensorPacket packet;
            double t_prev = -1.0;   ///< 上一相机帧时刻 (s), 合成源生成区间用
            auto t_start = std::chrono::steady_clock::now();
            while (frame < max_frames && !g_stop_requested) {
                // 有限超时取帧: 让循环周期性醒来检查 Ctrl+C;
                // getNextPacket false 时用 isCaptureStopped 区分"源结束"与"超时重试"
                if (!input_provider.getNextPacket(packet, /*timeout_ms=*/200)) {
                    if (input_provider.isCaptureStopped()) break;  // 源结束/断开/失败
                    continue;                                       // 超时 → 重试
                }
                ++frame;
                double t_cam = packet.timestamp_us * 1e-6;
                try {
                    // ---- ESKF: 合成源按 [t_prev, t_cam] 生成 IMU/雷达样本 ----
                    if (fusion_active) {
                        if (t_prev >= 0.0) {
                            for (const auto& s : sim_imu.generate(t_prev, t_cam, last_R_cam_w))
                                fusion.feedImu(s.t, s.acc, s.gyro);
                            for (const auto& s : sim_radar.generate(t_prev, t_cam, last_cam_height))
                                fusion.feedRadar(s.t, s.height);
                        }
                        // 排干 IMU/雷达至 t_cam (覆盖 YOLO 丢失路径的惯性传播);
                        // 线程化模式下由融合工作线程自动排干, 无需在此显式调用
                        if (!eskf_cfg.threaded)
                            fusion.propagateTo(t_cam);
                    }
                    t_prev = t_cam;   // 提前推进, 异常路径也安全

                    if (verbose_console)
                        std::cout << "\n===== 第 " << frame << " 帧"
                                  << (mono_mode ? " (单目)" : "") << " =====" << std::endl;
                    PipelineResult result = processFrame(frame, packet.left_image, packet.right_image);

                    // ---- ESKF: PnP 位姿 → 相机观测 (同时更新合成源真值参考) ----
                    if (fusion_active) {
                        fusion.feedCameraPose(t_cam, result.R, result.t, result.success);
                        if (result.success) {
                            last_R_cam_w = eskf_cfg.R_template_world * result.R.transpose();
                            last_cam_height = fusion.position()(2);
                        }
                    }

                    // ---- normal 模式: 每帧一行 → 终端 (ESKF 融合为主输出) ----
                    if (!verbose_console || capture_log) {
                        std::ostringstream os;
                        fusion::FusionState st = fusion.getLatestState();
                        if (fusion_active && st.initialized) {
                            const char* qs =
                                (st.quality == fusion::FusionQuality::Normal)   ? "N"
                              : (st.quality == fusion::FusionQuality::Degraded) ? "D"
                              : (st.quality == fusion::FusionQuality::Stale)    ? "S" : "?";
                            os << "[Frame " << frame << "] ESKF"
                               << " p=[" << st.position.x() << ", " << st.position.y() << ", " << st.position.z() << "]m"
                               << " v=[" << st.velocity.x() << ", " << st.velocity.y() << ", " << st.velocity.z() << "]"
                               << " q=[" << st.quaternion(0) << ", " << st.quaternion(1) << ", " << st.quaternion(2) << ", " << st.quaternion(3) << "]"
                               << " Q=" << qs;
                            if (result.success) {
                                os << " | PnP t(mm)=[" << result.t.x() << ", "
                                   << result.t.y() << ", " << result.t.z() << "]";
                            }
                            termLine(os.str());
                        } else if (result.success) {
                            Eigen::AngleAxisd aa(result.R);
                            Eigen::Vector3d rvec = aa.angle() * aa.axis();
                            std::string sname = result.strategy_name.empty()
                                ? "Unknown" : result.strategy_name;
                            os << "[Frame " << frame << "] " << sname
                               << "  n=" << result.n_matched
                               << "  r=[" << rvec.x() << ", " << rvec.y() << ", " << rvec.z() << "]"
                               << "  t=[" << result.t.x() << ", " << result.t.y() << ", " << result.t.z() << "]";
                            termLine(os.str());
                        } else {
                            termLine("[Frame " + std::to_string(frame) + "] FAILED");
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Frame " << frame << "] 异常: " << e.what() << std::endl;
                    // 继续下一帧，不中断整条序列
                }
            }

            // ---- 停止融合工作线程 (若已启动) ----
            if (fusion_active && eskf_cfg.threaded) fusion.stop();

            // ---- Phase 3.2: 收尾汇总 — 停止采集线程 → 打印统计 ----
            input_provider.shutdown();
            auto termOut = [&](const std::string& s) {
                if (capture_log && saved_cout) saved_cout->sputn(s.data(), s.size());
                std::cout << s;   // capture_log 时同时进入日志文件
            };
            if (g_stop_requested) {
                termOut("用户中断 (Ctrl+C)，停止采集\n");
            } else if (frame == 0) {
                std::cerr << "警告: 输入系统未产生任何帧" << std::endl;
            }
            if (frame > 0) {
                auto t_end = std::chrono::steady_clock::now();
                double secs = std::chrono::duration<double>(t_end - t_start).count();
                input::InputProvider::InputStats ist = input_provider.stats();
                termOut("===== 运行汇总 =====\n");
                termOut("处理帧数: " + std::to_string(frame)
                        + "  位姿成功: " + std::to_string(processed_ok)
                        + " (" + std::to_string(100.0 * processed_ok / frame) + "%)\n");
                if (secs > 0.0) {
                    termOut("实际处理 FPS: " + std::to_string(frame / secs) + "\n");
                }
                if (ist.captured > 0) {
                    termOut("输入统计 (线程化采集): 采集 " + std::to_string(ist.captured)
                            + " 帧, 消费 " + std::to_string(ist.consumed)
                            + " 帧, 丢弃 " + std::to_string(ist.dropped)
                            + " 帧 ("
                            + std::to_string(100.0 * ist.dropped / ist.captured)
                            + "%)\n");
                }
                if (fusion_active) {
                    auto fst = fusion.stats();
                    termOut("ESKF 统计: IMU样本=" + std::to_string(fst.imu_samples)
                            + "  相机更新=" + std::to_string(fst.cam_updates)
                            + "  FDI忽略=" + std::to_string(fst.cam_ignored)
                            + "  姿态跳过=" + std::to_string(fst.cam_rot_skipped)
                            + "  雷达接受=" + std::to_string(fst.radar_accepted)
                            + "  雷达拒绝=" + std::to_string(fst.radar_rejected)
                            + "  延迟兜底=" + std::to_string(fst.cam_late_fallback) + "\n");
                }
            }
        } else {
            // 旧版路径 —— 固定帧数循环（向后兼容）
            for (int frame = 1; frame <= max_frames; ++frame) {
                try {
                    if (verbose_console)
                        std::cout << "\n===== 第 " << frame << " 帧"
                                  << (mono_mode ? " (单目)" : "") << " =====" << std::endl;
                    processFrame(frame, left_img, right_img);
                } catch (const std::exception& e) {
                    std::cerr << "[Frame " << frame << "] 异常: " << e.what() << std::endl;
                }
            }
        }

        // M4: 帧循环结束 → 等待异步图像全部落盘, 避免程序退出时丢失
        gpnp::utils::AsyncImageSaver::flush();

        if (verbose_console)
            tracker->printLogs();

        // ---- 日志文件输出（normal + log_file 时已捕获 cout）----
        if (capture_log) {
            std::cout.rdbuf(saved_cout);   // 恢复终端输出
            verbose_console = false;

            // 配置摘要头
            std::ostringstream hdr;
            hdr << "================================================================\n"
                << " Steretracker 追踪日志\n"
                << "================================================================\n"
                << "运行模式: " << run_mode << (mono_mode ? " (单目)" : " (双目)") << "\n"
                << "配置路径: " << config_path << "\n";
            if (use_input_system) {
                hdr << "输入源目录: " << input_sys_cfg.image.directory_path
                    << "  type=";
                switch (input_sys_cfg.image.type) {
                    case input::ImageSourceType::File:       hdr << "file"; break;
                    case input::ImageSourceType::Directory:  hdr << "directory"; break;
                    case input::ImageSourceType::Sequence:   hdr << "sequence"; break;
                    case input::ImageSourceType::Camera:     hdr << "camera"; break;
                    default:                                 hdr << "unknown"; break;
                }
                hdr << "\n";
            } else {
                hdr << "左图: " << left_path << "\n"
                    << "右图: " << right_path << "\n";
            }
            hdr << "输出目录: " << output_dir << "\n"
                << "相机: fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy
                << " baseline_mm=" << baseline << "\n"
                << "YOLO: model=" << model_path << " conf=" << conf
                << " class=" << target_cls << " expand=" << expand << "\n"
                << "AKAZE模板: " << template_path << "  " << tw << "x" << th << "mm"
                << "  scale=" << scale << "  min_pts=" << min_pts << "\n"
                << "BC模板: " << binary_template_dir << "  corners=" << binary_cfg.corners
                << "  kernel=" << binary_cfg.kernel_size
                << "  p2m_c0=" << binary_cfg.pixel_to_meter_scale_class0
                << "  p2m_c1=" << binary_cfg.pixel_to_meter_scale_class1 << "\n"
                << "TT模板: " << tiny_template_dir << "  scale_factor=" << tiny_cfg.scale_factor
                << "  sq_c0=" << tiny_cfg.square_size_m_class0
                << "  sq_c1=" << tiny_cfg.square_size_m_class1 << "\n"
                << "面积阈值: akaze_min=" << akaze_min_area
                << "  tiny_max=" << tiny_max_area
                << "  class1_akaze=" << akaze_min_area_class1
                << "  class1_tiny=" << tiny_max_area_class1 << "\n"
                << "close_range: enabled=" << close_range_cfg.enabled
                << "  class1_min=" << close_range_cfg.class1_min_area
                << "  expand_ratio=" << close_range_cfg.roi_expand_ratio << "\n"
                << "================================================================\n\n";

            std::string log_path = output_dir + "/tracking_log.txt";
            std::ofstream ofs(log_path);
            if (ofs.is_open()) {
                ofs << hdr.str() << log_body.str();
                ofs.close();
                std::cout << "[Log] Saved: " << log_path << std::endl;
            } else {
                std::cerr << "[Log] Failed to open: " << log_path << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "致命错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
