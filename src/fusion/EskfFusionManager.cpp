/**
 * @file EskfFusionManager.cpp
 * @brief ESKF 多源信息融合适配层实现。
 *
 * 时间对齐方案 (drain-to-camera-time):
 *   主循环以相机帧为节拍。每帧到达 t_cam 时:
 *     1. 排干 IMU 缓冲中所有 t ≤ t_cam 的样本, 逐样本 predict 积分传播至 t_cam
 *        (保留 IMU 高频信息, 优于"插值到单样本"的消费模型)
 *     2. 排干雷达缓冲中所有 t ≤ t_cam 的样本, 逐样本 validate + update_altitude
 *     3. 相机位姿更新 (若 PnP 成功) → update_camera_pose_hybrid (内置 FDI)
 *     4. PnP 失败 / YOLO 丢失: 状态由 IMU 继续传播 (ESKF 的核心价值)
 *
 * 边界处理:
 *   - IMU 样本间隔 > max_imu_gap_s → 丢弃该样本并重新锚定 (防异常 dt 污染)
 *   - 相机有效位姿间隔 > max_cam_gap_s → 重置滤波器 (重新 lazy init)
 *   - 缓冲上限: IMU ~2s, 雷达 ~1s, 溢出丢最旧
 */

#include "fusion/EskfFusionManager.hpp"

#include <cmath>

namespace gpnp {
namespace fusion {

namespace {

// IMU 缓冲容量: 2 秒 + 余量
constexpr int kImuBufCap   = 2000;  // 200Hz × 2s × 5 余量, 溢出丢最旧
// 雷达缓冲容量: 1 秒 + 余量
constexpr int kRadarBufCap = 200;

} // namespace

// ============================================================================
// 构造 / 重置
// ============================================================================

EskfFusionManager::EskfFusionManager(const EskfFusionConfig& cfg)
    : cfg_(cfg)
    , eskf_(Eigen::VectorXd::Zero(16), Eigen::MatrixXd::Identity(15, 15) * 0.1,
            cfg.params)
    , radar_validator_(cfg.params.radar_alt_noise)
{
}

void EskfFusionManager::reset()
{
    imu_buf_.clear();
    radar_buf_.clear();
    last_prop_t_ = -1.0;
    last_cam_t_  = -1.0;
    initialized_ = false;
    stats_ = Stats{};
    eskf_ = eskf::ESKF_VIO(Eigen::VectorXd::Zero(16),
                           Eigen::MatrixXd::Identity(15, 15) * 0.1, cfg_.params);
    radar_validator_ = eskf::RadarAltimeter(cfg_.params.radar_alt_noise);
}

// ============================================================================
// 输入接口
// ============================================================================

void EskfFusionManager::feedImu(double t_sec,
                                const Eigen::Vector3d& acc,
                                const Eigen::Vector3d& gyro)
{
    if (!cfg_.enabled || !std::isfinite(t_sec)) return;
    imu_buf_.push_back(ImuSample{t_sec, acc, gyro});
    while ((int)imu_buf_.size() > kImuBufCap) imu_buf_.pop_front();
}

void EskfFusionManager::feedRadar(double t_sec, double height_m)
{
    if (!cfg_.enabled || !std::isfinite(t_sec) || !std::isfinite(height_m)) return;
    radar_buf_.push_back(RadarSample{t_sec, height_m});
    while ((int)radar_buf_.size() > kRadarBufCap) radar_buf_.pop_front();
}

void EskfFusionManager::feedCameraPose(double t_cam_sec,
                                       const Eigen::Matrix3d& R_tpl_cam,
                                       const Eigen::Vector3d& t_cam_mm,
                                       bool valid)
{
    if (!cfg_.enabled || !std::isfinite(t_cam_sec)) return;

    // 1. 排干 IMU/雷达至 t_cam (无论视觉是否成功, 惯性传播照常)
    propagateTo(t_cam_sec);

    if (!valid) return;  // 视觉失败 → 状态由 IMU 继续传播

    // 2. 位姿转换 (世界系 = 模板系, Z 向上)
    Eigen::Matrix3d R_cam_w;
    Eigen::Vector3d p_cam_w;
    convertPose(R_tpl_cam, t_cam_mm, cfg_.R_template_world, R_cam_w, p_cam_w);

    // 3. 间隔超限 → 重置 (视觉长时间丢失后重新初始化)
    if (initialized_ && last_cam_t_ >= 0.0
        && (t_cam_sec - last_cam_t_) > cfg_.max_cam_gap_s)
    {
        reset();
    }

    // 4. 首帧 → lazy init; 后续 → hybrid 更新 (位置永远更新 + 姿态自洽追加)
    if (!initialized_)
    {
        lazyInit(R_cam_w, p_cam_w);
        last_cam_t_ = t_cam_sec;
        return;
    }

    stats_.cam_updates++;

    // 姿态观测为四元数 [w,x,y,z] (相机→世界)
    Eigen::Quaterniond q_cam_w(R_cam_w);
    q_cam_w.normalize();
    Eigen::Matrix<double, 4, 1> q_meas;
    q_meas << q_cam_w.w(), q_cam_w.x(), q_cam_w.y(), q_cam_w.z();

    auto [nis, z_opt, dq_opt, action] =
        eskf_.update_camera_pose_hybrid(p_cam_w, q_meas);
    (void)nis; (void)z_opt; (void)dq_opt;

    if      (action == "ignore")      stats_.cam_ignored++;
    else if (action == "rot_skipped") stats_.cam_rot_skipped++;

    last_cam_t_ = t_cam_sec;
}

void EskfFusionManager::propagateTo(double t_sec)
{
    if (!cfg_.enabled || !std::isfinite(t_sec)) return;

    drainImu(t_sec);
    drainRadar(t_sec);

    // 尾部残余: 从最后一个 IMU 样本传播到 t_cam (无测量纯传播)
    if (initialized_ && last_prop_t_ >= 0.0 && last_prop_t_ < t_sec - 1e-9)
    {
        eskf_.predict(std::nullopt, std::nullopt, t_sec - last_prop_t_);
        last_prop_t_ = t_sec;
    }
}

// ============================================================================
// 输出接口
// ============================================================================

bool EskfFusionManager::initialized() const { return initialized_; }

Eigen::Vector3d EskfFusionManager::position() const
{
    return eskf_.x.segment<3>(0);
}

Eigen::Vector3d EskfFusionManager::velocity() const
{
    return eskf_.x.segment<3>(3);
}

Eigen::Matrix3d EskfFusionManager::rotation() const
{
    Eigen::Quaterniond q(eskf_.x.segment<4>(6));
    q.normalize();
    return q.toRotationMatrix();
}

Eigen::Matrix<double, 4, 1> EskfFusionManager::quaternion() const
{
    Eigen::Quaterniond q(eskf_.x.segment<4>(6));
    q.normalize();
    Eigen::Matrix<double, 4, 1> qv;
    qv << q.w(), q.x(), q.y(), q.z();
    return qv;
}

EskfFusionManager::Stats EskfFusionManager::stats() const { return stats_; }

// ============================================================================
// 内部实现
// ============================================================================

void EskfFusionManager::drainImu(double t_cam)
{
    while (!imu_buf_.empty() && imu_buf_.front().t <= t_cam + 1e-9)
    {
        const ImuSample& s = imu_buf_.front();

        double dt;
        if (last_prop_t_ < 0.0)
        {
            // 首个样本: 锚定时间原点, 不积分
            last_prop_t_ = s.t;
            dt = 0.0;
        }
        else
        {
            dt = s.t - last_prop_t_;
            if (dt < 0.0) dt = 0.0;  // 乱序保护
        }

        // 间隙超限: 丢弃该样本, 重新锚定 (防异常 dt 污染协方差)
        if (dt > cfg_.max_imu_gap_s)
        {
            imu_buf_.pop_front();
            last_prop_t_ = s.t;
            continue;
        }

        if (initialized_ && dt > 1e-9)
        {
            eskf_.predict_adaptive(s.acc, s.gyro, dt);
            stats_.imu_samples++;
        }

        imu_buf_.pop_front();
        last_prop_t_ = s.t;
    }
}

void EskfFusionManager::drainRadar(double t_cam)
{
    while (!radar_buf_.empty() && radar_buf_.front().t <= t_cam + 1e-9)
    {
        const RadarSample s = radar_buf_.front();
        radar_buf_.pop_front();

        if (!initialized_) continue;  // 需先由相机位姿完成初始化

        // 预测高度与协方差 (沿估计重力方向投影, 与 update_altitude 同约定)
        const Eigen::Vector3d& g = cfg_.params.gravity;
        Eigen::Vector3d n_hat = (g.norm() < 0.1)
                                    ? Eigen::Vector3d(0.0, 0.0, 1.0)
                                    : (-g.normalized());
        double z_pred   = eskf_.x.segment<3>(0).dot(n_hat);
        double P_z_pred = n_hat.dot(eskf_.P.block<3, 3>(0, 0) * n_hat);

        auto [ok, nis, info] =
            radar_validator_.validate(s.height, z_pred, P_z_pred, s.t);
        (void)nis;

        if (ok && !info.radar_failed)
        {
            eskf_.update_altitude(s.height, info.effective_noise_std);
            stats_.radar_accepted++;
        }
        else
        {
            stats_.radar_rejected++;
        }
    }
}

void EskfFusionManager::lazyInit(const Eigen::Matrix3d& R_cam_w,
                                 const Eigen::Vector3d& p_cam_w)
{
    Eigen::Quaterniond q(R_cam_w);
    q.normalize();

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(16);
    x0.segment<3>(0) = p_cam_w;
    x0(6) = q.w(); x0(7) = q.x(); x0(8) = q.y(); x0(9) = q.z();
    // 速度 / 偏置默认 0

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(15, 15);
    P0.block<3, 3>(0, 0)  = Eigen::Matrix3d::Identity() * (cfg_.init_std_p  * cfg_.init_std_p);
    P0.block<3, 3>(3, 3)  = Eigen::Matrix3d::Identity() * (cfg_.init_std_v  * cfg_.init_std_v);
    P0.block<3, 3>(6, 6)  = Eigen::Matrix3d::Identity() * (cfg_.init_std_q  * cfg_.init_std_q);
    P0.block<3, 3>(9, 9)  = Eigen::Matrix3d::Identity() * (cfg_.init_std_ba * cfg_.init_std_ba);
    P0.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (cfg_.init_std_bg * cfg_.init_std_bg);

    eskf_ = eskf::ESKF_VIO(x0, P0, cfg_.params);
    initialized_ = true;
    last_prop_t_ = -1.0;  // 重新锚定, 下一个 IMU 样本作为积分起点
}

void EskfFusionManager::convertPose(const Eigen::Matrix3d& R_tpl_cam,
                                    const Eigen::Vector3d& t_cam_mm,
                                    const Eigen::Matrix3d& R_template_world,
                                    Eigen::Matrix3d& R_cam_w_out,
                                    Eigen::Vector3d& p_cam_w_out)
{
    // R_tpl_cam: 模板→相机 (p_cam = R·p_tpl + t), t 单位 mm
    // 相机在模板系: p_cam_tpl = -Rᵀ·t; 相机姿态: R_cam_tpl = Rᵀ
    // 世界系 = 模板系 (Z 向上) + 可选修正旋转 R_template_world
    R_cam_w_out = R_template_world * R_tpl_cam.transpose();
    p_cam_w_out = R_cam_w_out * (-t_cam_mm / 1000.0);  // mm → m
}

} // namespace fusion
} // namespace gpnp
