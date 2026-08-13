/**
 * @file SimulatedSensors.cpp
 * @brief 合成 IMU / 雷达数据源实现。
 *
 * IMU 简化模型 (悬停近似):
 *   世界系比力 s_world = a_world - g。悬停时 a_world ≈ 0 → s_world ≈ -g = (0,0,+9.81)。
 *   相机系比力 s_cam = R_cam_wᵀ · s_world, 再加白噪声与偏置。
 *   陀螺 = 白噪声 + 偏置 (悬停无角速度)。
 *
 * 雷达简化模型:
 *   height = truth_height + N(0, σ); inject_jump_every_s > 0 时周期性叠加 jump_m,
 *   用于验证 RadarAltimeter 的跳变拒绝与失效/恢复状态机。
 */

#include "input/SimulatedSensors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gpnp {
namespace input {

// ============================================================================
// SimulatedImu
// ============================================================================

SimulatedImu::SimulatedImu(const SimulatedImuConfig& cfg)
    : cfg_(cfg), rng_(cfg_.seed)
{
    if (cfg_.rate_hz <= 0.0) cfg_.rate_hz = 200.0;
}

std::vector<SimulatedImu::Sample>
SimulatedImu::generate(double t_from, double t_to, const Eigen::Matrix3d& R_cam_w)
{
    std::vector<Sample> out;
    if (!cfg_.enabled || t_to <= t_from) return out;

    const double dt = 1.0 / cfg_.rate_hz;

    // 相机系比力 (悬停近似): s_cam = R_cam_wᵀ · (0,0,+9.81)
    Eigen::Vector3d s_cam = R_cam_w.transpose() * Eigen::Vector3d(0.0, 0.0, 9.81);

    // 首次生成: 从 t_from + dt 起步; 跨帧继续时接着上次节奏
    if (next_t_ <= t_from || next_t_ <= 0.0) next_t_ = t_from + dt;

    std::normal_distribution<double> acc_noise(0.0, cfg_.sigma_acc);
    std::normal_distribution<double> gyro_noise(0.0, cfg_.sigma_gyro);

    while (next_t_ <= t_to + 1e-9)
    {
        Sample s;
        s.t   = next_t_;
        s.acc = s_cam + cfg_.bias_acc
              + Eigen::Vector3d(acc_noise(rng_), acc_noise(rng_), acc_noise(rng_));
        s.gyro = cfg_.bias_gyro
               + Eigen::Vector3d(gyro_noise(rng_), gyro_noise(rng_), gyro_noise(rng_));
        out.push_back(s);
        next_t_ += dt;
    }
    return out;
}

void SimulatedImu::reset()
{
    next_t_ = -1.0;
}

// ============================================================================
// SimulatedRadar
// ============================================================================

SimulatedRadar::SimulatedRadar(const SimulatedRadarConfig& cfg)
    : cfg_(cfg), rng_(cfg_.seed)
{
    if (cfg_.rate_hz <= 0.0) cfg_.rate_hz = 20.0;
}

std::vector<SimulatedRadar::Sample>
SimulatedRadar::generate(double t_from, double t_to, double truth_height)
{
    std::vector<Sample> out;
    if (!cfg_.enabled || t_to <= t_from) return out;

    const double dt = 1.0 / cfg_.rate_hz;

    if (next_t_ <= t_from || next_t_ <= 0.0) next_t_ = t_from + dt;

    std::normal_distribution<double> noise(0.0, cfg_.noise_m);

    while (next_t_ <= t_to + 1e-9)
    {
        double h = truth_height + noise(rng_);
        if (std::isfinite(h) && h < 0.0) h = 0.0;  // 高度非负

        // 周期跳变注入 (验证检验器): 每 inject_jump_every_s 秒一次
        if (cfg_.inject_jump_every_s > 0.0
            && (last_jump_t_ < 0.0 || next_t_ - last_jump_t_ >= cfg_.inject_jump_every_s))
        {
            h += cfg_.jump_m;
            last_jump_t_ = next_t_;
        }

        out.push_back(Sample{next_t_, h});
        next_t_ += dt;
    }
    return out;
}

void SimulatedRadar::reset()
{
    next_t_ = -1.0;
    last_jump_t_ = -1.0;
}

} // namespace input
} // namespace gpnp
