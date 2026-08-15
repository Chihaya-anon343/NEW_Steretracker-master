#include "../framework/TestAssert.hpp"
#include "fusion/EskfFusionManager.hpp"
#include "fusion/FusionTypes.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace gpnp;
using namespace gpnp::fusion;

// ============================================================================
// ESKF 多模态集成测试: 螺旋上升轨迹 + IMU/相机/雷达三路合成数据 + 错误注入
//
// 被测模块: fusion::EskfFusionManager (线程化 threaded=true)
// 数据来源: 纯代码合成 —— 一条螺旋上升地面真值轨迹, 由运动学反推 IMU 比力/角速度,
//           相机位姿(PnP 约定)与雷达高度; 可注入 4 类错误 (相机跳变/失效, 雷达跳变, 相机延迟)。
// 判定方式: 软验证 (跑通全流程 + smoke 断言), 轨迹/误差写 CSV 供可视化脚本绘图。
// ============================================================================

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kG  = 9.81;

// ---- 螺旋轨迹参数 ----
struct TrajParams {
    double radius   = 2.0;    // 螺旋半径 (m)
    double omega    = 0.5;    // 角速度 (rad/s)
    double vz       = 0.5;    // 爬升速度 (m/s)
    double h0       = 10.0;   // 初始高度 (m)
    double duration = 20.0;   // 时长 (s)
};

// ---- 错误事件 (type: cam_jump | cam_fail | radar_jump | cam_delay) ----
struct ErrorEvent {
    std::string type;
    double t_start;
    double t_end;       // 单点错误用 t_end == t_start
    double magnitude;   // 跳变幅度(m) / 延迟(s)
};

// ---- 采样记录 (写 CSV 用) ----
struct TrajSample {
    double t;
    Eigen::Vector3d gt;       // 真值位置 (m)
    Eigen::Vector3d fused;    // 融合位置 (m)
    double err;               // 3D 误差 (m)
    int quality;              // FusionQuality 枚举值
};

// ============================================================================
// 螺旋轨迹合成器: 地面真值位姿 + 反推 IMU / 相机 / 雷达
// ============================================================================
class SpiralSynthesizer {
public:
    SpiralSynthesizer(const TrajParams& p, const std::vector<ErrorEvent>& events, unsigned seed)
        : params_(p), events_(events), rng_(seed) {}

    // 地面真值位姿 (相机→世界旋转 + 位置); yaw 沿切线, 水平飞行
    void gtPose(double t, Eigen::Matrix3d& R_cam_w, Eigen::Vector3d& p) const {
        double x = params_.radius * std::cos(params_.omega * t);
        double y = params_.radius * std::sin(params_.omega * t);
        double z = params_.h0 + params_.vz * t;
        p = Eigen::Vector3d(x, y, z);
        double yaw = params_.omega * t + kPi / 2.0;
        R_cam_w = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }

    // 世界系加速度 (向心加速度)
    Eigen::Vector3d accWorld(double t) const {
        double ax = -params_.radius * params_.omega * params_.omega * std::cos(params_.omega * t);
        double ay = -params_.radius * params_.omega * params_.omega * std::sin(params_.omega * t);
        return Eigen::Vector3d(ax, ay, 0.0);
    }

    // IMU 样本 (比力 + 角速度, 相机系, SI)
    void imuAt(double t, Eigen::Vector3d& acc, Eigen::Vector3d& gyro) {
        Eigen::Matrix3d R_cam_w;
        Eigen::Vector3d p;
        gtPose(t, R_cam_w, p);

        Eigen::Vector3d a = accWorld(t);
        Eigen::Vector3d f_world = a - Eigen::Vector3d(0.0, 0.0, -kG);  // 比力 = a - g
        acc = R_cam_w.transpose() * f_world;
        acc += noise3(sigma_acc_);

        gyro = Eigen::Vector3d(0.0, 0.0, params_.omega);   // 仅 yaw 率
        gyro += noise3(sigma_gyro_);
    }

    // 相机观测 (PnP 约定: R_tpl_cam=模板→相机, t_cam_mm=mm), 含噪声 + 错误
    CameraObservation cameraObsAt(double t) {
        CameraObservation obs;

        // 相机延迟段: 曝光时刻 = t - delay, 送达时刻 = t (照片 delay 秒前拍, 现在送达)
        double delay = inEvent("cam_delay", t) ? delayOf("cam_delay") : 0.0;
        double t_exp = t - delay;
        obs.t_exposure = t_exp;
        obs.t_arrival  = t;
        obs.valid      = true;

        // 位姿取曝光时刻的真值
        Eigen::Matrix3d R_cam_w;
        Eigen::Vector3d p;
        gtPose(t_exp, R_cam_w, p);

        // 错误: 相机位置跳变 (+3m x)
        if (inEvent("cam_jump", t_exp)) p += Eigen::Vector3d(3.0, 0.0, 0.0);
        // 错误: 相机失效
        if (inEvent("cam_fail", t_exp)) obs.valid = false;

        // 位置噪声 (PnP 误差)
        p += noise3(sigma_cam_pos_);

        // 转 PnP 约定: convertPose 中 R_cam_w = R_tpl_camᵀ, p_cam_w = R_cam_w·(-t/1000)
        obs.R_tpl_cam = R_cam_w.transpose();
        obs.t_cam_mm  = -1000.0 * R_cam_w.transpose() * p;
        return obs;
    }

    // 雷达高度 (m), 含噪声 + 跳变
    double radarAt(double t) {
        Eigen::Matrix3d R_cam_w;
        Eigen::Vector3d p;
        gtPose(t, R_cam_w, p);
        double h = p.z();
        if (inEvent("radar_jump", t)) h += 50.0;   // 雷达跳变 +50m
        h += nd_(rng_) * sigma_radar_;
        return h;
    }

    const std::vector<ErrorEvent>& events() const { return events_; }

private:
    Eigen::Vector3d noise3(double sigma) {
        return Eigen::Vector3d(nd_(rng_) * sigma, nd_(rng_) * sigma, nd_(rng_) * sigma);
    }
    bool inEvent(const std::string& type, double t) const {
        for (const auto& e : events_)
            if (e.type == type && t >= e.t_start - 1e-9 && t <= e.t_end + 1e-9)
                return true;
        return false;
    }
    double delayOf(const std::string& type) const {
        for (const auto& e : events_)
            if (e.type == type) return e.magnitude;
        return 0.0;
    }

    TrajParams params_;
    std::vector<ErrorEvent> events_;
    double sigma_acc_     = 0.1;    // 合成 IMU 加速度噪声 σ (m/s²)
    double sigma_gyro_    = 0.005;  // 合成 IMU 角速度噪声 σ (rad/s)
    double sigma_radar_   = 0.3;    // 合成雷达高度噪声 σ (m)
    double sigma_cam_pos_ = 0.1;    // 合成相机位置噪声 σ (m)
    std::mt19937 rng_;
    std::normal_distribution<double> nd_{0.0, 1.0};
};

// ---- 轮询等待 (线程化, 防 flaky) ----
bool waitUntil(const std::function<bool()>& pred, double timeout_s = 5.0) {
    using clock = std::chrono::steady_clock;
    const auto deadline =
        clock::now() + std::chrono::milliseconds((int)(timeout_s * 1000.0));
    while (clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return pred();
}

// ---- CSV 输出 ----
void writeTrajCsv(const std::string& path, const std::vector<TrajSample>& samples) {
    std::ofstream f(path);
    f << "t,gt_x,gt_y,gt_z,fused_x,fused_y,fused_z,err_3d,quality\n";
    for (const auto& s : samples) {
        f << s.t << ','
          << s.gt.x() << ',' << s.gt.y() << ',' << s.gt.z() << ','
          << s.fused.x() << ',' << s.fused.y() << ',' << s.fused.z() << ','
          << s.err << ',' << s.quality << '\n';
    }
    std::printf("[test_eskf_multimodal] 已写 %s (%zu 采样点)\n", path.c_str(), samples.size());
}

void writeEventsCsv(const std::string& path, const std::vector<ErrorEvent>& events) {
    std::ofstream f(path);
    f << "type,t_start,t_end,magnitude\n";
    for (const auto& e : events)
        f << e.type << ',' << e.t_start << ',' << e.t_end << ',' << e.magnitude << '\n';
    std::printf("[test_eskf_multimodal] 已写 %s (%zu 错误事件)\n", path.c_str(), events.size());
}

} // namespace

// ============================================================================
// 用例: 螺旋上升多模态线程化 ESKF 完整流程 (软验证)
// ============================================================================
static void test_eskf_multimodal_threaded() {
    // 1. 配置 (threaded=true, 噪声对齐 config/tracker_config_eskf.json)
    EskfFusionConfig cfg;
    cfg.enabled           = true;
    cfg.threaded          = true;
    cfg.imu_rate_hz       = 200.0;
    cfg.radar_rate_hz     = 20.0;
    cfg.max_imu_gap_s     = 0.1;
    cfg.max_cam_gap_s     = 1.0;
    cfg.backprop_window_s = 0.2;
    cfg.state_hist_hz     = 100;
    cfg.max_output_age_s  = 0.5;
    cfg.latency_fallback  = LatencyFallback::Inflate;
    cfg.init_std_p  = 1.0;
    cfg.init_std_v  = 1.0;
    cfg.init_std_q  = 0.1;
    cfg.init_std_ba = 0.1;
    cfg.init_std_bg = 0.01;
    cfg.params.sigma_acc       = 0.3;
    cfg.params.sigma_gyro      = 0.02;
    cfg.params.sigma_acc_bias  = 0.01;
    cfg.params.sigma_gyro_bias = 0.001;
    cfg.params.sigma_pos_rw    = 0.5;
    cfg.params.cam_pos_noise   = 0.1;
    cfg.params.cam_rot_noise   = 1.0;
    cfg.params.radar_alt_noise = 0.30;
    cfg.params.gravity         = Eigen::Vector3d(0.0, 0.0, -kG);

    // 2. 螺旋轨迹 + 4 类错误注入
    TrajParams tp;
    std::vector<ErrorEvent> events = {
        {"radar_jump",  5.0,  5.2, 50.0},   // 雷达高度跳变 +50m (0.2s 窗口 ≈ 4 样本)
        {"cam_jump",    8.0,  8.5,  3.0},   // 相机位置跳变 +3m x (0.5s)
        {"cam_fail",   13.0, 13.6,  0.0},   // 相机失效 0.6s (间隔 0.8s < max_cam_gap_s=1.0, 避免触发 reset 清空 stats)
        {"cam_delay",  16.0, 18.0,  0.1},   // 相机延迟 0.1s (反向传播)
    };
    SpiralSynthesizer synth(tp, events, 42u);

    // 3. 融合器 (线程化)
    EskfFusionManager fusion(cfg);
    fusion.start();

    // 4. 逐时间步喂入 (以 IMU 节拍 200Hz 驱动)
    const double dt_imu    = 1.0 / 200.0;
    const double dt_radar  = 1.0 / 20.0;
    const double dt_cam    = 1.0 / 10.0;
    const double dt_sample = 1.0 / 20.0;

    std::vector<TrajSample> samples;
    samples.reserve((size_t)(tp.duration / dt_sample) + 2);

    double next_radar  = 0.0;
    double next_cam    = 0.0;
    double next_sample = 0.0;
    int    iter = 0;

    for (double t = 0.0; t <= tp.duration + 1e-9; t += dt_imu) {
        // IMU (200Hz)
        Eigen::Vector3d acc, gyro;
        synth.imuAt(t, acc, gyro);
        fusion.feedImu(t, acc, gyro);

        // 雷达 (20Hz)
        if (t >= next_radar - 1e-9) {
            fusion.feedRadar(t, synth.radarAt(t));
            next_radar += dt_radar;
        }

        // 相机 (10Hz)
        if (t >= next_cam - 1e-9) {
            fusion.feedCameraPose(synth.cameraObsAt(t));
            next_cam += dt_cam;
        }

        // 采样记录 (20Hz)
        if (t >= next_sample - 1e-9) {
            FusionState s = fusion.getLatestState();
            Eigen::Matrix3d R;
            Eigen::Vector3d gt;
            synth.gtPose(t, R, gt);
            TrajSample rec;
            rec.t      = t;
            rec.gt     = gt;
            rec.fused  = s.position;
            rec.err    = (s.position - gt).norm();
            rec.quality = (int)s.quality;
            samples.push_back(rec);
            next_sample += dt_sample;
        }

        // 轻微节流: 让 worker 跟上 (防缓冲溢出丢最旧)
        if (++iter % 100 == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    // 5. 等 worker 排干缓冲 (最后一个相机帧已处理), 再 stop
    bool drained = waitUntil([&] {
        FusionState s = fusion.getLatestState();
        return s.initialized && s.last_cam_t >= tp.duration - 0.2;
    });
    TEST_ASSERT_MSG(drained, "融合工作线程未在超时内排干缓冲");

    fusion.stop();

    // 6. smoke 断言 (软验证)
    TEST_ASSERT(fusion.initialized());
    auto st = fusion.stats();
    TEST_ASSERT(st.imu_samples    > 0);   // IMU 被积分
    TEST_ASSERT(st.cam_updates    > 0);   // 相机被更新
    TEST_ASSERT(st.radar_rejected > 0);   // 雷达跳变被 RadarAltimeter 拒
    TEST_ASSERT(st.cam_ignored    > 0);   // 相机跳变被 FDI 拒

    std::printf("[test_eskf_multimodal] imu=%d cam_update=%d cam_ignored=%d cam_rot_skip=%d "
                "radar_acc=%d radar_rej=%d late_fb=%d\n",
                st.imu_samples, st.cam_updates, st.cam_ignored, st.cam_rot_skipped,
                st.radar_accepted, st.radar_rejected, st.cam_late_fallback);

    // 7. 写 CSV (可视化)
    writeTrajCsv("eskf_traj.csv", samples);
    writeEventsCsv("eskf_events.csv", events);
}

// ============================================================================
// 测试入口
// ============================================================================

REGISTER_TEST(test_eskf_multimodal_threaded);

int main() {
    return gpnp_test::TestRegistry::instance().runAll();
}
