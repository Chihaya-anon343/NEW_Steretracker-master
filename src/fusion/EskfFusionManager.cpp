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

#include <algorithm>
#include <cmath>
#include <utility>

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

EskfFusionManager::~EskfFusionManager()
{
    stop();
}

void EskfFusionManager::reset()
{
    imu_buf_.clear();
    radar_buf_.clear();
    cam_buf_.clear();
    imu_hist_.clear();
    state_hist_.clear();
    last_prop_t_ = -1.0;
    last_cam_t_  = -1.0;
    last_snap_t_ = -1.0;
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
    {
        std::lock_guard<std::mutex> lock(mtx_);
        imu_buf_.push_back(ImuSample{t_sec, acc, gyro});
        while ((int)imu_buf_.size() > kImuBufCap) imu_buf_.pop_front();
    }
    cv_.notify_one();
}

void EskfFusionManager::feedRadar(double t_sec, double height_m)
{
    if (!cfg_.enabled || !std::isfinite(t_sec) || !std::isfinite(height_m)) return;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        radar_buf_.push_back(RadarSample{t_sec, height_m});
        while ((int)radar_buf_.size() > kRadarBufCap) radar_buf_.pop_front();
    }
    cv_.notify_one();
}

void EskfFusionManager::feedCameraPose(double t_cam_sec,
                                       const Eigen::Matrix3d& R_tpl_cam,
                                       const Eigen::Vector3d& t_cam_mm,
                                       bool valid)
{
    // 兼容旧接口: 无延迟语义, 曝光 = 送达
    CameraObservation obs;
    obs.t_exposure = t_cam_sec;
    obs.t_arrival  = t_cam_sec;
    obs.R_tpl_cam  = R_tpl_cam;
    obs.t_cam_mm   = t_cam_mm;
    obs.valid      = valid;
    feedCameraPose(obs);
}

void EskfFusionManager::feedCameraPose(const CameraObservation& obs)
{
    if (!cfg_.enabled || !std::isfinite(obs.t_arrival)) return;

    if (!threaded_)
    {
        // 单线程同步 (向后兼容): 立即处理
        std::lock_guard<std::mutex> lock(mtx_);
        processCameraObs(obs);
    }
    else
    {
        // 线程化: 异步入队, 由融合工作线程处理
        {
            std::lock_guard<std::mutex> lock(mtx_);
            cam_buf_.push_back(obs);
        }
        cv_.notify_one();
    }
}

void EskfFusionManager::processCameraObs(const CameraObservation& obs)
{
    // 调用方须持有 mtx_ (sync 路径的 feedCameraPose 或融合工作线程)。

    // 1. 排干 IMU/雷达至送达时刻 (无论视觉是否成功, 惯性传播照常)
    propagateInternal(obs.t_arrival);

    if (!obs.valid) return;  // 视觉失败 → 状态由 IMU 继续传播

    // 2. 位姿转换 (世界系 = 模板系, Z 向上)
    Eigen::Matrix3d R_cam_w;
    Eigen::Vector3d p_cam_w;
    convertPose(obs.R_tpl_cam, obs.t_cam_mm, cfg_.R_template_world, R_cam_w, p_cam_w);

    // 3. 间隔超限 → 重置 (视觉长时间丢失后重新初始化)
    if (initialized_ && last_cam_t_ >= 0.0
        && (obs.t_exposure - last_cam_t_) > cfg_.max_cam_gap_s)
    {
        reset();
    }

    // 4. 首帧 → lazy init; 后续 → hybrid 更新 (位置永远更新 + 姿态自洽追加)
    if (!initialized_)
    {
        lazyInit(R_cam_w, p_cam_w);
        last_cam_t_ = obs.t_exposure;
        return;
    }

    // 姿态观测为四元数 [w,x,y,z] (相机→世界)
    Eigen::Quaterniond q_cam_w(R_cam_w);
    q_cam_w.normalize();
    Eigen::Matrix<double, 4, 1> q_meas;
    q_meas << q_cam_w.w(), q_cam_w.x(), q_cam_w.y(), q_cam_w.z();

    // 5. 延迟测量处理: 反向传播为主, 协方差膨胀兜底
    const double t0    = obs.t_exposure;
    const double t_now = last_prop_t_;   // 当前已传播到的时刻
    const double latency = t_now - t0;

    bool applied = false;
    // 仅真实延迟 (>1ms) 才反向传播; 否则按到达时刻直接更新, 与旧同步路径一致,
    // 避免无延迟时回退丢失同窗口已应用的雷达更新。
    if (latency > 1e-3)
        applied = applyCameraBackprop(t0, t_now, p_cam_w, q_meas);

    // cam_updates 仅在"真正应用了更新"时计数 (backprop 成功 / 非 Reject 的
    // 直接或膨胀更新); 被 Reject 丢弃的观测不算更新, 只计 cam_ignored。
    if (applied)
    {
        stats_.cam_updates++;
    }
    else
    {
        const bool late = latency > 1e-3;
        if (late && cfg_.latency_fallback == LatencyFallback::Reject)
        {
            stats_.cam_ignored++;
        }
        else
        {
            // 无延迟 → 直接更新; 延迟超窗 → 协方差膨胀兜底 (一阶有界近似)
            // 膨胀仅对真实延迟生效; 亚毫秒 (≤1ms) 视为无延迟, eff_latency=0,
            // 保持与 t0=t1 直接处理逐位一致。
            stats_.cam_updates++;
            const double eff_latency = late ? std::max(0.0, latency) : 0.0;
            const double orig_pos = eskf_.params.cam_pos_noise;
            const double orig_rot = eskf_.params.cam_rot_noise;
            eskf_.params.cam_pos_noise = std::sqrt(
                orig_pos * orig_pos
                + cfg_.params.sigma_pos_rw * cfg_.params.sigma_pos_rw * eff_latency);
            eskf_.params.cam_rot_noise = std::sqrt(
                orig_rot * orig_rot
                + cfg_.params.sigma_gyro * cfg_.params.sigma_gyro * eff_latency);

            auto [nis, z_opt, dq_opt, action] =
                eskf_.update_camera_pose_hybrid(p_cam_w, q_meas);
            (void)nis; (void)z_opt; (void)dq_opt;

            eskf_.params.cam_pos_noise = orig_pos;
            eskf_.params.cam_rot_noise = orig_rot;

            if      (action == "ignore")      stats_.cam_ignored++;
            else if (action == "rot_skipped") stats_.cam_rot_skipped++;
            if (late) stats_.cam_late_fallback++;
        }
    }

    last_cam_t_ = t0;
}

void EskfFusionManager::propagateTo(double t_sec)
{
    if (!cfg_.enabled || !std::isfinite(t_sec)) return;
    std::lock_guard<std::mutex> lock(mtx_);
    propagateInternal(t_sec);
}

void EskfFusionManager::propagateInternal(double t_sec)
{
    // 调用方须持有 mtx_。
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
    // 状态向量四元数为 [w,x,y,z] 序 (eskf_vio.hpp:187), 必须用 4 标量构造,
    // Eigen 的向量构造会把 [w,x,y,z] 误当 [x,y,z,w] (单位四元数会错成 [0,1,0,0])
    Eigen::Quaterniond q(eskf_.x(6), eskf_.x(7), eskf_.x(8), eskf_.x(9));
    q.normalize();
    return q.toRotationMatrix();
}

Eigen::Matrix<double, 4, 1> EskfFusionManager::quaternion() const
{
    // 同上: [w,x,y,z] 序 4 标量构造
    Eigen::Quaterniond q(eskf_.x(6), eskf_.x(7), eskf_.x(8), eskf_.x(9));
    q.normalize();
    Eigen::Matrix<double, 4, 1> qv;
    qv << q.w(), q.x(), q.y(), q.z();
    return qv;
}

EskfFusionManager::Stats EskfFusionManager::stats() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return stats_;
}

FusionState EskfFusionManager::getLatestState() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    FusionState s;
    s.initialized = initialized_;
    if (!initialized_)
    {
        s.quality = FusionQuality::Uninitialized;
        return s;
    }

    s.position  = eskf_.x.segment<3>(0);
    s.velocity  = eskf_.x.segment<3>(3);
    // [w,x,y,z] 序 4 标量构造 (同上)
    Eigen::Quaterniond q(eskf_.x(6), eskf_.x(7), eskf_.x(8), eskf_.x(9));
    q.normalize();
    s.quaternion << q.w(), q.x(), q.y(), q.z();
    s.cov_trace  = eskf_.P.block<3, 3>(0, 0).trace();
    s.last_cam_t = last_cam_t_;

    // 距上次相机更新的时长 (s) → 可信度分级
    const double age = last_prop_t_ - last_cam_t_;
    if (last_cam_t_ < 0.0)
        s.quality = FusionQuality::Degraded;
    else if (age > cfg_.max_cam_gap_s)
        s.quality = FusionQuality::Stale;
    else if (age > cfg_.max_output_age_s)
        s.quality = FusionQuality::Degraded;
    else
        s.quality = FusionQuality::Normal;
    return s;
}

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
            recordHistory(s);   // 记入反向传播历史 (快照 + 重放窗口)
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

// ============================================================================
// 反向传播支持
// ============================================================================

void EskfFusionManager::recordHistory(const ImuSample& s)
{
    const double snap_dt = cfg_.state_hist_hz > 0 ? 1.0 / cfg_.state_hist_hz : 0.01;

    imu_hist_.push_back(s);
    if (last_snap_t_ < 0.0 || s.t - last_snap_t_ >= snap_dt)
    {
        state_hist_.push_back(StateSnap{s.t, eskf_});
        last_snap_t_ = s.t;
    }
    // 按窗口裁剪 (丢弃过早样本, 保证回退窗口有界)
    const double cutoff = s.t - cfg_.backprop_window_s;
    while (!imu_hist_.empty() && imu_hist_.front().t < cutoff - 1e-9)
        imu_hist_.pop_front();
    while (!state_hist_.empty() && state_hist_.front().t < cutoff - 1e-9)
        state_hist_.pop_front();
}

bool EskfFusionManager::applyCameraBackprop(double t_exposure, double t_now,
                                            const Eigen::Vector3d& p_cam_w,
                                            const Eigen::Matrix<double, 4, 1>& q_meas)
{
    // 找到 ≤ t_exposure 的最近快照
    const StateSnap* snap = nullptr;
    for (const auto& s : state_hist_)
    {
        if (s.t <= t_exposure + 1e-9) snap = &s;
        else break;
    }
    if (!snap) return false;   // 无可用快照 (延迟超窗) → 走兜底

    const double snap_dt = cfg_.state_hist_hz > 0 ? 1.0 / cfg_.state_hist_hz : 0.01;

    // 1. 回退到快照
    eskf_ = snap->eskf;
    double t = snap->t;

    // 2. 重放 IMU 到 t_exposure
    for (const auto& imu : imu_hist_)
    {
        if (imu.t <= snap->t + 1e-9) continue;
        if (imu.t > t_exposure + 1e-9) break;
        const double dt = imu.t - t;
        if (dt > 0) eskf_.predict_adaptive(imu.acc, imu.gyro, dt);
        t = imu.t;
    }
    if (t < t_exposure - 1e-9)   // 尾部残余到精确 t_exposure (纯传播)
    {
        eskf_.predict(std::nullopt, std::nullopt, t_exposure - t);
        t = t_exposure;
    }

    // 3. 在 t_exposure 应用相机更新
    auto [nis, z_opt, dq_opt, action] = eskf_.update_camera_pose_hybrid(p_cam_w, q_meas);
    (void)nis; (void)z_opt; (void)dq_opt;
    if      (action == "ignore")      stats_.cam_ignored++;
    else if (action == "rot_skipped") stats_.cam_rot_skipped++;

    // 4. 重放 IMU t_exposure → t_now, 同时重建历史 (新快照反映修正后状态)
    std::deque<ImuSample> new_imu_hist;
    std::deque<StateSnap> new_state_hist;
    last_snap_t_ = t;
    for (const auto& imu : imu_hist_)
    {
        if (imu.t <= t_exposure + 1e-9) continue;
        const double dt = imu.t - t;
        if (dt > 0) eskf_.predict_adaptive(imu.acc, imu.gyro, dt);
        t = imu.t;
        new_imu_hist.push_back(imu);
        if (imu.t - last_snap_t_ >= snap_dt)
        {
            new_state_hist.push_back(StateSnap{imu.t, eskf_});
            last_snap_t_ = imu.t;
        }
    }

    // 5. 尾部残余到 t_now
    if (t < t_now - 1e-9)
    {
        eskf_.predict(std::nullopt, std::nullopt, t_now - t);
        t = t_now;
    }
    last_prop_t_ = t_now;

    // 6. 替换历史
    imu_hist_   = std::move(new_imu_hist);
    state_hist_ = std::move(new_state_hist);

    return true;
}

// ============================================================================
// 线程化 (Phase 4)
// ============================================================================

void EskfFusionManager::start()
{
    if (threaded_) return;
    threaded_ = true;
    running_  = true;
    worker_   = std::thread(&EskfFusionManager::fusionLoop, this);
}

void EskfFusionManager::stop()
{
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void EskfFusionManager::fusionLoop()
{
    while (running_)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] {
            return !running_ || !imu_buf_.empty() || !radar_buf_.empty()
                   || !cam_buf_.empty();
        });
        if (!running_) break;

        // 1. 处理相机观测 (按到达序; 内部 propagateInternal + lazyInit/反向传播)
        while (!cam_buf_.empty())
        {
            CameraObservation obs = cam_buf_.front();
            cam_buf_.pop_front();
            processCameraObs(obs);
        }

        // 2. 无相机时把已缓冲的 IMU/雷达继续传播到最新 (保持状态新鲜)
        // 仅在已初始化后排干: 未初始化时排干会丢弃缓冲 IMU 并推进 last_prop_t_,
        // 与同步路径 (等首帧相机按 t_arrival 排干) 发散。
        double t_max = -1.0;
        if (!imu_buf_.empty())   t_max = std::max(t_max, imu_buf_.back().t);
        if (!radar_buf_.empty()) t_max = std::max(t_max, radar_buf_.back().t);
        if (t_max >= 0.0 && initialized_ && t_max > last_prop_t_)
            propagateInternal(t_max);
    }
}

} // namespace fusion
} // namespace gpnp
