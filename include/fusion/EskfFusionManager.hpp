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

#include <Eigen/Dense>

#include <deque>
#include <optional>
#include <vector>

namespace gpnp {
namespace fusion {

// ============================================================================
// 融合配置
// ============================================================================

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
};

// ============================================================================
// ESKF 融合管理器
// ============================================================================

class EskfFusionManager {
public:
    explicit EskfFusionManager(const EskfFusionConfig& cfg);

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
    };
    Stats stats() const;

    /// 重置滤波器: 清空缓冲 + 回到未初始化态
    void reset();

private:
    struct ImuSample {
        double t;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
    };
    struct RadarSample {
        double t;
        double height;
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

    EskfFusionConfig cfg_;
    eskf::ESKF_VIO eskf_;

    // 缓冲 + 对齐状态
    std::deque<ImuSample>   imu_buf_;
    std::deque<RadarSample> radar_buf_;
    double last_prop_t_ = -1.0;   ///< 状态已传播到的时刻 (s); <0 = 未传播过
    double last_cam_t_  = -1.0;   ///< 上一相机位姿帧时刻 (s)

    // 滤波器子模块
    eskf::RadarAltimeter radar_validator_;

    // 状态
    bool initialized_ = false;   ///< 已完成 lazy init (首个有效相机位姿)

    // 统计
    Stats stats_;
};

} // namespace fusion
} // namespace gpnp
