#pragma once

/**
 * @file SimulatedSensors.hpp
 * @brief 合成 IMU / 雷达数据源 (离线回放与链路验证用)。
 *
 * Module: input
 * Function: 在无真实硬件 (串口/CAN) 时, 按配置频率生成带时间戳的 IMU 角速度/
 *           加速度与雷达高度样本, 用于验证 ESKF 融合链路 (对齐/预测/更新/FDI)。
 *
 * 简化假设 (链路验证而非物理仿真):
 *   - 载体近似悬停: IMU 比力 = -g 旋转到相机系 + 白噪声 + 偏置
 *   - 雷达真值高度 = 调用方提供的相机高度 + 白噪声, 可选周期性跳变注入
 *     (验证 RadarAltimeter 的跳变拒绝)
 *
 * 接入方式: 由 main.cpp 按相机帧时间轴驱动 generate(t_from, t_to, ...),
 *           产出样本后喂给 fusion::EskfFusionManager::feedImu/feedRadar。
 *           真实硬件源 (Phase 2) 将来以同样的"带时间戳样本流"模式接入。
 * Dependencies: Eigen, C++17 (<random>)
 */

#include <Eigen/Dense>

#include <random>
#include <vector>

namespace gpnp {
namespace input {

// ============================================================================
// 合成 IMU
// ============================================================================

struct SimulatedImuConfig {
    bool enabled = false;
    double rate_hz   = 200.0;             ///< 输出频率 (Hz)
    double sigma_acc = 0.1;               ///< 加速度噪声 σ (m/s²)
    double sigma_gyro = 0.005;            ///< 角速度噪声 σ (rad/s)
    Eigen::Vector3d bias_acc  = Eigen::Vector3d::Zero();  ///< 加计偏置
    Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();  ///< 陀螺偏置
    unsigned seed = 42;                   ///< 随机数种子 (可复现)
};

class SimulatedImu {
public:
    explicit SimulatedImu(const SimulatedImuConfig& cfg);

    struct Sample {
        double t;
        Eigen::Vector3d acc;   ///< 比力 (相机系, m/s²)
        Eigen::Vector3d gyro;  ///< 角速度 (相机系, rad/s)
    };

    /// 生成 (t_from, t_to] 区间的样本。
    /// @param R_cam_w  相机→世界旋转 (用于把比力 -g 转到相机系; 未初始化时单位阵)
    std::vector<Sample> generate(double t_from, double t_to,
                                 const Eigen::Matrix3d& R_cam_w);

    void reset();

private:
    SimulatedImuConfig cfg_;
    double next_t_ = -1.0;   ///< 下一个样本时刻
    std::mt19937 rng_;
};

// ============================================================================
// 合成雷达
// ============================================================================

struct SimulatedRadarConfig {
    bool enabled = false;
    double rate_hz    = 20.0;    ///< 输出频率 (Hz)
    double noise_m    = 0.30;    ///< 高度噪声 σ (m)
    double initial_height = 10.0; ///< 初始高度 (m, 真值未提供时)
    double inject_jump_every_s = 0.0;  ///< >0: 每 N 秒注入一次跳变 (验证检验器)
    double jump_m     = 50.0;    ///< 跳变幅度 (m)
    unsigned seed = 42;
};

class SimulatedRadar {
public:
    explicit SimulatedRadar(const SimulatedRadarConfig& cfg);

    struct Sample {
        double t;
        double height;   ///< 高度 (m)
    };

    /// 生成 (t_from, t_to] 区间的样本。
    /// @param truth_height 当前真值高度 (通常取最近一帧相机位姿高度)
    std::vector<Sample> generate(double t_from, double t_to, double truth_height);

    void reset();

private:
    SimulatedRadarConfig cfg_;
    double next_t_ = -1.0;
    double last_jump_t_ = -1.0;
    std::mt19937 rng_;
};

} // namespace input
} // namespace gpnp
