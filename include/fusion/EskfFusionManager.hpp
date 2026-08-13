#pragma once

/**
 * @file EskfFusionManager.hpp
 * @brief ESKF 多源信息融合适配层 (Error-State Kalman Filter Fusion Manager)
 *
 * Module: fusion
 * Function: 将位姿解算系统 (PnP) 输出的相机位姿、输入系统的 IMU/雷达数据
 *           接入 eskf/eskf_vio.hpp 的 ESKF_VIO 滤波器, 输出平滑融合位姿。
 *
 * 设计要点:
 *   - 不改动第三方库 eskf/eskf_vio.hpp (header-only, 接口已完备)
 *   - "排干式" 时间对齐 (drain-to-camera-time): 以相机帧为节拍, 每帧
 *     将时间戳 ≤ t_cam 的 IMU 样本逐样本 predict 积分传播至 t_cam;
 *     雷达样本逐样本 validate + update_altitude
 *   - lazy init: 首个有效相机位姿直接置名义态, 不经过 Kalman update
 *   - 世界系 = 模板系, Z 轴向上 (重力 (0,0,-9.81), 雷达高度沿 Z)
 *   - 单位约定: 位姿解算输出 t 为 mm, 内部统一 SI (m)
 *
 * Input:   PnP 位姿 {R(模板→相机), t(mm)}, IMU (acc/gyro, SI),
 *          Radar 高度 (m), 时间戳 (double 秒)
 * Output:  融合位置/速度/姿态 (SI, 相机→世界), 统计信息
 * Dependencies: eskf/eskf_vio.hpp (Eigen), C++17
 */

#include "eskf/eskf_vio.hpp"
#include "fusion/FusionTypes.hpp"

#include <Eigen/Dense>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace gpnp {
namespace fusion {

// ============================================================================
// 融合配置
// ============================================================================

/// 相机延迟测量兜底策略 (延迟超出反向传播窗口时)
enum class LatencyFallback {
    Inflate,   ///< 协方差膨胀: 把延迟折算成额外观测噪声, 按到达时刻应用
    Reject     ///< 直接丢弃该相机观测
};

struct EskfFusionConfig {
    bool enabled = false;              ///< 是否启用 ESKF 融合

    // ---- 时间对齐参数 ----
    double imu_rate_hz   = 200.0;      ///< IMU 标称频率 (合成源速率 + 缓冲容量估计)
    double radar_rate_hz = 20.0;       ///< 雷达标称频率
    double max_imu_gap_s = 0.1;        ///< IMU 相邻样本间隔超限 → 丢弃该样本
    double max_cam_gap_s = 1.0;        ///< 相机帧间隔超限 → 重置滤波器 (重新 lazy init)

    // ---- ESKF 噪声参数 (直接透传给 eskf::ESKFParams) ----
    eskf::ESKFParams params;

    // ---- 初始化标准差 (→ P0) ----
    double init_std_p  = 1.0;          ///< 位置 (m)
    double init_std_v  = 1.0;          ///< 速度 (m/s)
    double init_std_q  = 0.1;          ///< 姿态 (rad)
    double init_std_ba = 0.1;          ///< 加计偏置 (m/s²)
    double init_std_bg = 0.01;         ///< 陀螺偏置 (rad/s)

    // ---- 坐标约定 ----
    /// 模板系 → 世界系修正旋转 (默认单位阵: 世界系 = 模板系, Z 轴向上)
    Eigen::Matrix3d R_template_world = Eigen::Matrix3d::Identity();

    // ---- 反向传播 (延迟测量) 参数 ----
    double backprop_window_s = 0.2;         ///< 状态快照/IMU 历史回退窗口 (s)
    int    state_hist_hz    = 100;          ///< 状态快照记录频率 (Hz)
    LatencyFallback latency_fallback = LatencyFallback::Inflate;  ///< 延迟超窗兜底策略
    double max_output_age_s = 0.5;          ///< 相机更新间隔超此值 → DEGRADED (仍输出惯导)

    // ---- 线程化 (Phase 4) ----
    bool threaded = false;                  ///< 是否启用内部融合工作线程 (异步消费)
};

// ============================================================================
// 融合输出状态
// ============================================================================

/// 融合状态可信度分级
enum class FusionQuality {
    Uninitialized,  ///< 尚未完成 lazy init (无有效相机位姿)
    Normal,         ///< 相机更新新鲜, 输出可信
    Degraded,       ///< 相机丢失但未超 max_cam_gap_s, 仅惯性/雷达传播
    Stale           ///< 相机丢失超过 max_cam_gap_s, 已重置/不可信
};

/// 对外输出的融合状态快照 (线程安全读取)
struct FusionState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();   ///< 位置 (m, 世界系)
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();   ///< 速度 (m/s)
    Eigen::Matrix<double, 4, 1> quaternion =
        Eigen::Matrix<double, 4, 1>::Zero();               ///< 姿态 [w,x,y,z]
    double cov_trace = 0.0;                                ///< 位置协方差迹 (可信度信号)
    double last_cam_t = -1.0;                              ///< 最近相机观测时刻 (s)
    FusionQuality quality = FusionQuality::Uninitialized;
    bool initialized = false;
};

// ============================================================================
// ESKF 融合管理器
// ============================================================================

class EskfFusionManager {
public:
    explicit EskfFusionManager(const EskfFusionConfig& cfg);
    ~EskfFusionManager();

    // ========================================================================
    // 输入接口 (主循环调用; 时间戳统一为 double 秒)
    // ========================================================================

    /// 投喂 IMU 样本 (acc/gyro 均为 SI 单位, IMU 本体坐标系)。
    /// 仅入缓冲, 实际传播由 propagateTo()/feedCameraPose() 排干时执行。
    void feedImu(double t_sec, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro);

    /// 投喂雷达高度样本 (m)。仅入缓冲, 在排干时 validate + update_altitude。
    void feedRadar(double t_sec, double height_m);

    /// 投喂相机位姿 (PnP 输出: R = 模板→相机, t 单位 mm; valid=false 表示该帧
    /// 视觉失败, 状态仅由 IMU 继续传播)。内部执行:
    ///   排干 IMU/雷达至 t_sec → (首帧 lazy init) 或 update_camera_pose_hybrid
    void feedCameraPose(double t_cam_sec,
                        const Eigen::Matrix3d& R_tpl_cam,
                        const Eigen::Vector3d& t_cam_mm,
                        bool valid);

    /// 投喂相机位姿 (带曝光/送达双时间戳, 支持延迟测量反向传播)。
    /// obs.t_exposure = 曝光时刻 (t0), obs.t_arrival = 送达时刻 (t1)。
    void feedCameraPose(const CameraObservation& obs);

    /// 排干 IMU/雷达缓冲, 将状态传播至 t_sec (无相机位姿帧时调用)。
    void propagateTo(double t_sec);

    // ========================================================================
    // 输出接口
    // ========================================================================

    /// 是否已完成首次初始化 (首个有效相机位姿)
    bool initialized() const;

    /// 融合位置 (m, 世界系)
    Eigen::Vector3d position() const;

    /// 融合速度 (m/s, 世界系)
    Eigen::Vector3d velocity() const;

    /// 融合姿态: 相机系 → 世界系旋转矩阵
    Eigen::Matrix3d rotation() const;

    /// 融合姿态: 四元数 [w, x, y, z]
    Eigen::Matrix<double, 4, 1> quaternion() const;

    /// 融合状态快照 (含可信度分级 + 协方差迹)。
    /// 注: 单线程模式下无锁; 多线程化 (Phase 4 融合线程) 后由读侧加锁。
    FusionState getLatestState() const;

    /// 原始滤波器引用 (诊断/高级使用)
    const eskf::ESKF_VIO& filter() const { return eskf_; }

    // ---- 统计 ----
    struct Stats {
        int imu_samples     = 0;  ///< 已传播的 IMU 样本数
        int cam_updates     = 0;  ///< 相机更新总数 (含被拒)
        int cam_ignored     = 0;  ///< 相机 FDI REJECTED, 完全跳过
        int cam_rot_skipped = 0;  ///< 位置已更新, 姿态 NIS 超门限跳过
        int radar_accepted  = 0;  ///< 雷达接受更新数
        int radar_rejected  = 0;  ///< 雷达拒绝数 (跳变/NIS/无效)
        int cam_late_fallback = 0; ///< 相机延迟超窗 → 协方差膨胀/Reject 兜底次数
    };
    Stats stats() const;

    /// 重置滤波器: 清空缓冲 + 回到未初始化态
    void reset();

    // ========================================================================
    // 线程化 (Phase 4): 启动/停止内部融合工作线程
    // ========================================================================

    /// 启动融合工作线程。启动后 feedImu/feedRadar/feedCameraPose 变为异步入队,
    /// 融合线程独立以 IMU 节拍预测、相机/雷达异步更新; 相机缺席时状态继续传播。
    void start();

    /// 停止工作线程并 join (析构时自动调用)。
    void stop();

    /// 是否运行在线程化模式 (start() 后为 true)。
    bool threaded() const { return threaded_; }

private:
    /// 反向传播快照: 某时刻的完整滤波器状态 (ESKF_VIO 整体拷贝)。
    struct StateSnap {
        double t;
        eskf::ESKF_VIO eskf;
    };

    /// 排干 IMU 缓冲至 t_sec (逐样本 predict)
    void drainImu(double t_sec);

    /// 排干雷达缓冲至 t_sec (validate + update_altitude)
    void drainRadar(double t_sec);

    /// 用首个有效相机位姿初始化名义态 + P0
    void lazyInit(const Eigen::Matrix3d& R_cam_w, const Eigen::Vector3d& p_cam_w);

    /// 位姿转换: PnP 输出 → 相机在世界系位姿 (m)
    static void convertPose(const Eigen::Matrix3d& R_tpl_cam,
                            const Eigen::Vector3d& t_cam_mm,
                            const Eigen::Matrix3d& R_template_world,
                            Eigen::Matrix3d& R_cam_w_out,
                            Eigen::Vector3d& p_cam_w_out);

    /// 记录 IMU 样本到重放历史, 并按 state_hist_hz 记状态快照 + 按窗口裁剪。
    void recordHistory(const ImuSample& s);

    /// 反向传播核心: 回退到 ≤ t_exposure 的最近快照, 重放 IMU, 在 t0 应用
    /// 相机更新, 再重放到当前, 并重建历史。返回是否成功回退 (否则走兜底)。
    bool applyCameraBackprop(double t_exposure, double t_now,
                             const Eigen::Vector3d& p_cam_w,
                             const Eigen::Matrix<double, 4, 1>& q_meas);

    /// 相机观测处理体 (无锁, 调用方须持有 mtx_)。原 feedCameraPose(obs) 主体。
    void processCameraObs(const CameraObservation& obs);

    /// propagateTo 的处理体 (无锁, 调用方须持有 mtx_)。
    void propagateInternal(double t_sec);

    /// 融合工作线程主循环。
    void fusionLoop();

    // 线程化状态
    std::thread worker_;
    std::atomic<bool> running_{false};
    bool threaded_ = false;
    mutable std::mutex mtx_;                 ///< 保护输入缓冲 + 滤波状态
    std::condition_variable cv_;
    std::deque<CameraObservation> cam_buf_;  ///< 待处理相机观测 (mtx_ 保护)

    EskfFusionConfig cfg_;
    eskf::ESKF_VIO eskf_;

    // 缓冲 + 对齐状态
    std::deque<ImuSample>   imu_buf_;
    std::deque<RadarSample> radar_buf_;
    double last_prop_t_ = -1.0;   ///< 状态已传播到的时刻 (s); <0 = 未传播过
    double last_cam_t_  = -1.0;   ///< 上一相机位姿帧时刻 (s)

    // 反向传播历史 (窗口 = backprop_window_s)
    std::deque<ImuSample> imu_hist_;    ///< 窗口内原始 IMU 样本 (重放用)
    std::deque<StateSnap> state_hist_;  ///< 窗口内滤波器状态快照 (回退用)
    double last_snap_t_ = -1.0;         ///< 上一快照时刻 (s)

    // 滤波器子模块
    eskf::RadarAltimeter radar_validator_;

    // 状态
    bool initialized_ = false;   ///< 已完成 lazy init (首个有效相机位姿)

    // 统计
    Stats stats_;
};

} // namespace fusion
} // namespace gpnp
