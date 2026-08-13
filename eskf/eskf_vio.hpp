// =============================================================
// 文件: eskf_vio.hpp (单头文件版)
// 说明: ESKF (Error-State Kalman Filter) + VIO (视觉惯性里程计) C++ 单头文件库
//       包含滤波器参数、相机FDI、在线重力估计、雷达高度计检验及核心滤波类
//
// 主要模块:
//   - ESKFParams:            IMU/相机/雷达 噪声参数配置
//   - CameraFDIInfo/Result/Log: 相机故障检测与隔离(FDI)数据结构
//   - GravityEstimateInfo:   重力估计诊断信息结构体
//   - GravityEstimator:      在线重力滑动窗口估计器
//        基于滑动窗口内轨迹最小二乘 + Tikhonov 正则化 + 自适应平滑
//   - RadarAltimeter:        雷达高度计数据检验器
//        高度跳变检测 + NIS 卡方检验 + 滑动窗口统计
//   - ESKF_VIO:              核心误差状态卡尔曼滤波器
//        支持 IMU 预测(含二阶积分修正) + Camera 6-DOF 位姿更新
//        + Radar 高度更新，内置 FDI 保护机制
//
// 使用方式: 仅需包含本头文件（header-only，所有实现均为 inline），
//           多个编译单元包含均安全（ODR 满足）。
//           接口与 <qing-yun>/include/eskf_vio.hpp + src/eskf_vio.cpp
//           完全一致（源码同步自 qing-yun 仓库，无任何行为改动）。
//
// 依赖: Eigen3, C++17
// 编译选项建议: -std=c++17
// =============================================================

#ifndef ESKF_VIO_HPP
#define ESKF_VIO_HPP

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <string>
#include <optional>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <deque>
#include <limits>
#include <utility>
#include <tuple>

namespace eskf {

// ============================================================
// 参数定义
// ============================================================
struct ESKFParams {
    // IMU 连续时间噪声 (m/s²/√Hz 或 rad/s/√Hz)
    double sigma_acc       = 0.3;
    double sigma_gyro      = 0.02;

    // 偏置随机游走
    double sigma_acc_bias  = 0.01;
    double sigma_gyro_bias = 0.001;

    // 偏置衰减系数
    double p_acc  = 0.0;
    double p_gyro = 0.0;

    // 位置随机游走噪声 —— 补偿前向 Euler 离散化误差
    // 单位 m/s/√Hz。设 0 可退化为纯理论模型。
    // 来源: Python eskf_mask.py:sigma_pos_rw
    double sigma_pos_rw = 0.5;

    // 相机位姿观测噪声
    double cam_pos_noise = 0.1;
    double cam_rot_noise = 1.0;

    // Camera Z 轴噪声解耦 (让 camera 在 Z 轴给雷达让步)
    // 来源: Python eskf_mask.py:use_cam_z_noise_decoupling
    bool   use_cam_z_noise_decoupling = false;
    double cam_pos_z_noise = 5.0;

    // 重力向量（滑动窗口在线估计会动态更新此字段）
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);

    // 雷达高度计噪声 (Z 轴单维观测标准差, 单位 m)
    // 来源: Python eskf_mask.py:radar_alt_noise
    double radar_alt_noise = 0.30;

    // IMU → Camera 安装外参
    // R_imu_cam: IMU 本体坐标系 → Camera 本体坐标系的旋转矩阵
    // p_imu_in_cam: IMU 在 Camera 坐标系下的平移向量（杆臂，单位 m）
    // 默认值为单位阵和零向量，保持向后兼容
    Eigen::Matrix3d R_imu_cam    = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_imu_in_cam = Eigen::Vector3d::Zero();

    // Camera FDI 参数
    double cam_fdi_max_scale       = 10.0;
    double cam_fdi_ewma_alpha      = 0.0;   // 0 -> 禁用
    int    cam_fdi_extreme_count   = 1;
    double cam_fdi_recover_factor  = 0.6;
    int    cam_fdi_max_log         = 5000;
    // hybrid 路径（当前生产路径）：姿态追加门限因子，门限 = χ²₉₅/rot_detect_scale。
    // 4.0 = 有效门限 7.8147/4 = 1.954（±2σ 内自洽才追加），py_eskf 已验证推荐值。
    // 注：6-DOF camera_fdi_evaluate 中该值同时是姿态分类缩放（乘 nis_rot_used），
    // 语义不同但运行期二选一（fusion_thread 现调用 hybrid），无冲突。
    double cam_fdi_rot_detect_scale = 4.0;
};

// ============================================================
// 日志 / 结果结构体
// ============================================================
struct CameraFDIInfo {
    std::string state;  // NORMAL | SUSPECT | REJECTED
    std::unordered_map<std::string, double> numeric;
    std::string reason;
};

struct CameraFDIResult {
    std::string action;  // accept | adaptive_pos | adaptive_rot | adaptive_both | ignore
    double nis_pos = 0.0;
    double nis_rot = 0.0;
    Eigen::Matrix3d S_pos = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d S_rot = Eigen::Matrix3d::Zero();
    CameraFDIInfo info;
};

struct CameraFDILogEntry {
    int id = 0;
    std::string action;
    std::optional<double> nis_pos;
    std::optional<double> nis_rot;
    Eigen::Matrix<double, 6, 1> z;
    Eigen::Matrix<double, 4, 1> dq;
    std::string info_str;
};

// ============================================================
// 重力估计信息
// ============================================================
struct GravityEstimateInfo {
    int n_samples = 0;
    int n_window  = 0;
    bool outlier  = false;
    Eigen::Vector3d g_prior = Eigen::Vector3d(0.0, 0.0, -9.81);
};

// ============================================================
// 雷达高度计验证信息
// ============================================================
struct RadarAltValidateInfo {
    std::string reason = "pending";
    double nis = 0.0;
    double innovation = 0.0;
    double S = 0.0;
    double accept_rate = 0.0;
    double window_mean = 0.0;
    double window_std = 0.0;
    double jump_magnitude = 0.0;
    double nis_threshold = 3.841;
    double jump_threshold = 5.0;
    bool should_skip_update = false;   // radar_failed → 主循环应跳过 update_altitude
    bool radar_failed = false;         // 雷达当前是否标记为失效
    double effective_noise_std = 0.30; // 动态有效噪声标准差（传递给 update_altitude）
    int consecutive_reject = 0;        // 当前连续拒绝计数（仅用于诊断日志）
    int consecutive_accept = 0;        // 当前连续接受计数（仅用于诊断日志）
};

struct RadarAltStatistics {
    int total_calls = 0;
    int accepted = 0;
    int rejected = 0;
    double accept_rate = 0.0;
    int reject_jump = 0;
    int reject_nis = 0;
    int reject_no_data = 0;
    double nis_threshold = 7.879;
    double jump_threshold = 30.0;
    double radar_noise_std = 0.30;
};

// ============================================================
// 工具函数 (内联)
// ============================================================

inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d M;
    M <<     0.0, -v.z(),  v.y(),
          v.z(),    0.0, -v.x(),
         -v.y(),  v.x(),    0.0;
    return M;
}

// 四元数按 [w, x, y, z] 顺序
inline Eigen::Matrix<double, 4, 1> quat_mul(
    const Eigen::Matrix<double, 4, 1> &q1,
    const Eigen::Matrix<double, 4, 1> &q2)
{
    double w1 = q1(0), x1 = q1(1), y1 = q1(2), z1 = q1(3);
    double w2 = q2(0), x2 = q2(1), y2 = q2(2), z2 = q2(3);
    Eigen::Matrix<double, 4, 1> r;
    r(0) = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    r(1) = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    r(2) = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    r(3) = w1*z2 + x1*y2 - y1*x2 + z1*w2;
    return r;
}

inline Eigen::Matrix<double, 4, 1> quat_inv(
    const Eigen::Matrix<double, 4, 1> &q)
{
    Eigen::Matrix<double, 4, 1> r;
    r(0) =  q(0);
    r(1) = -q(1);
    r(2) = -q(2);
    r(3) = -q(3);
    return r;
}

// 四元数指数映射：3-vector -> 四元数 [w,x,y,z]
inline Eigen::Matrix<double, 4, 1> quat_exp(
    const Eigen::Vector3d &delta)
{
    double theta = delta.norm();
    Eigen::Matrix<double, 4, 1> q;
    if (theta < 1e-6) {
        double qw = 1.0 - (theta * theta) / 8.0;
        Eigen::Vector3d qv = 0.5 * delta
                             - (theta * theta) / 48.0 * delta;
        q(0) = qw;
        q(1) = qv(0);
        q(2) = qv(1);
        q(3) = qv(2);
        double n = q.norm();
        if (n > 0) q /= n;
        return q;
    } else {
        Eigen::Vector3d axis = delta / theta;
        q(0) = std::cos(theta / 2.0);
        Eigen::Vector3d v = axis * std::sin(theta / 2.0);
        q(1) = v(0); q(2) = v(1); q(3) = v(2);
        return q;
    }
}

// 四元数 -> 3x3 旋转矩阵 [w,x,y,z]
inline Eigen::Matrix3d quat_to_rot(
    const Eigen::Matrix<double, 4, 1> &q)
{
    double w = q(0), x = q(1), y = q(2), z = q(3);
    Eigen::Matrix3d R;
    R << 1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
         2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
         2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y);
    return R;
}

inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// Sigmoid 激活函数，将输入映射到 [0,1]
inline double sigmoid(double x, double k = 1.0, double x0 = 0.5) {
    return 1.0 / (1.0 + std::exp(-k * (x - x0)));
}

// ============================================================
// 内部工具：矩阵 SVD 伪逆（detail 命名空间，头文件内安全）
// ============================================================
namespace detail {

inline Eigen::Matrix<double, 6, 6>
svd_pinv_6x6(const Eigen::Matrix<double, 6, 6> &M)
{
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
        M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixV()
           * svd.singularValues().asDiagonal().toDenseMatrix().inverse()
           * svd.matrixU().transpose();
}

inline Eigen::Matrix3d
svd_pinv_3x3(const Eigen::Matrix3d &M)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixV()
           * svd.singularValues().asDiagonal().toDenseMatrix().inverse()
           * svd.matrixU().transpose();
}

} // namespace detail

// ============================================================
// 重力估计器：轨迹最小二乘 + EWMA 递推（在线方案）
// 对标 tilt-exp 文档《在线重力估计与转回世界系输出》
//   短窗口(60帧)轨迹LS → g_new；EWMA a=1/k 递推 → g_est（等效全局加权平均）
//   收敛判据看 g_est 方向变化量；recover_rotation() 恢复转回矩阵
// ============================================================
class GravityEstimator {
public:
    // 单帧重力观测样本
    struct GravitySample {
        double t_sec = 0.0;
        Eigen::Vector3d p_cam = Eigen::Vector3d::Zero();
        // IMU 步数据: list of (dt, acc_u_world)
        std::vector<std::pair<double, Eigen::Vector3d>> imu_steps;
    };

    GravityEstimator(int window_size = 60);

    // ---- 样本管理 ----
    void add_sample(double t_sec,
                    const Eigen::Vector3d &p_cam,
                    const std::vector<std::pair<double, Eigen::Vector3d>> &imu_steps);

    int  sample_count() const;
    int  window_size() const;
    bool is_ready(int stride = 1) const;

    // ---- 核心估计 ----
    std::pair<std::optional<Eigen::Vector3d>, GravityEstimateInfo>
    estimate(double tikhonov_lambda = 1e-3, int stride = 1);

    // 可观性分析
    struct ObservabilityResult {
        bool x_obs = false, y_obs = false, z_obs = false;
        bool all_obs = false;
        double pos_disp = 0.0;
        double time_span = 0.0;
        std::string detail;
        double obs_score_x = 0.0;
        double obs_score_y = 0.0;
        double obs_score_z = 0.0;
    };
    ObservabilityResult compute_observability(int stride = 5);

    // ---- EWMA 递推平滑 ----
    // a = 1/k（k=更新次数）：等效对历史 g_new 等权平均（全局加权平均的在线等价）
    // 返回 (g_est, a)
    std::pair<Eigen::Vector3d, double>
    ewma_smooth(const Eigen::Vector3d &g_new);

    // ---- 便捷更新接口：短窗口 LS → EWMA 递推 ----
    // t_sec: 当前帧时刻（秒），写入收敛历史；<0 时不记录
    std::pair<std::optional<Eigen::Vector3d>, GravityEstimateInfo>
    update(double tikhonov_lambda = 1e-3,
           int estimate_stride = 50,
           int obs_stride = 5,
           bool freeze_enabled = true,
           double t_sec = -1.0);

    // ---- 收敛判据 / 转回矩阵 ----
    // 最近 window_s 秒内 g_est 方向变化（度）；历史不足返回 +inf
    double convergence_angle_deg(double window_s = 5.0) const;
    bool is_converged(double window_s = 5.0, double thresh_deg = 0.03) const;
    // 从估计重力恢复转回矩阵 R_est（Rodrigues 最小旋转：标准重力方向 → ĝ）
    // 输出变换: p_w = R_estᵀ·p'（见 tilt-exp 文档 §6）
    Eigen::Matrix3d recover_rotation() const;

    // ---- 查询接口 ----
    const Eigen::Vector3d& get_gravity() const;
    int  update_count() const;
    void reset();

private:
    int window_size_;
    std::deque<GravitySample> samples_;
    Eigen::Vector3d gravity_est_;
    bool initialized_;
    int update_count_;

    // EWMA 递推状态
    int k_ = 0;                                     // 更新计数（a = 1/k）
    std::deque<std::pair<double, Eigen::Vector3d>> g_hist_;  // (t, g_est) 收敛历史

    // 可观性阈值（compute_observability 用）
    double obs_score_threshold_ = 0.45;
};

// ============================================================
// 雷达高度计数据检验器
// 来源: Python radar_altimeter.py — RadarAltimeter
// ============================================================
class RadarAltimeter {
public:
    RadarAltimeter(double radar_noise_std = 0.30,
                   double nis_threshold = 7.879,
                   double jump_threshold = 30.0,
                   int window_size = 20,
                   int fail_threshold = 20,
                   int recovery_threshold = 10);

    // 检验雷达高度观测是否可信
    // 返回: (is_valid, nis_value, info)
    std::tuple<bool, double, RadarAltValidateInfo>
    validate(double z_meas, double z_pred,
             double P_z_pred, double t = -1.0);

    RadarAltStatistics get_statistics() const;

    // 运行时可调参数
    double radar_noise_std;
    double nis_threshold;
    double jump_threshold;

private:
    int window_size_;
    double last_valid_z_;
    double last_valid_t_;
    std::deque<double> recent_valid_z_;
    std::deque<double> recent_valid_t_;

    int total_calls_;
    int accepted_;
    int rejected_;
    int reject_jump_;
    int reject_nis_;
    int reject_no_data_;

    // 失效状态机
    int consecutive_reject_ = 0;
    int consecutive_accept_ = 0;
    int fail_threshold_ = 20;
    int recovery_threshold_ = 10;
    bool radar_failed_ = false;

    void _check_failure_state();
    void _check_recovery_state();
};

// ============================================================
// ESKF_VIO 主类
// ============================================================
class ESKF_VIO {
public:
    // x: 16-dim 名义状态 [p(3), v(3), q(4), b_a(3), b_g(3)]
    // P: 15x15 误差协方差
    ESKF_VIO(const Eigen::VectorXd &x0,
             const Eigen::MatrixXd &P0,
             const ESKFParams &params_in = ESKFParams());

    // 获取 Camera FDI 日志
    const std::vector<CameraFDILogEntry>& get_camera_fdi_log() const;

    // ============================================================
    // Camera FDI 评估
    // ============================================================
    std::pair<Eigen::Matrix<double, 6, 6>, CameraFDIResult>
    camera_fdi_evaluate(const Eigen::Matrix<double, 6, 1> &z,
                        const Eigen::Matrix<double, 6, 15> &H,
                        const Eigen::Matrix<double, 6, 6> &R);

    // ============================================================
    // IMU 预测 (基础版)
    // ============================================================
    bool predict(const std::optional<Eigen::Vector3d> &acc_opt,
                 const std::optional<Eigen::Vector3d> &gyro_opt,
                 double dt);

    // ============================================================
    // IMU 预测 (自适应噪声版)
    // 来源: Python eskf_mask.py:predict_adaptive()
    // ============================================================
    bool predict_adaptive(const Eigen::Vector3d &acc,
                          const Eigen::Vector3d &gyro,
                          double dt,
                          Eigen::Vector3d *acc_world_out = nullptr);

    // ============================================================
    // Camera 位姿更新（6D 观测，基础版）
    // 返回：(NIS_total, z_opt, dq_opt)
    // ============================================================
    std::tuple<double,
               std::optional<Eigen::Matrix<double, 6, 1>>,
               std::optional<Eigen::Matrix<double, 4, 1>>>
    update_camera_pose(const std::optional<Eigen::Vector3d> &p_meas_opt,
                       const std::optional<Eigen::Matrix<double, 4, 1>> &q_meas_opt);

    // ============================================================
    // Camera 位姿更新（hybrid：位置永远更新 + 姿态自洽追加）
    // 返回：(nis_total, z6d_opt, dq_opt, action)
    //   action ∈ "no_data" | "ignore" | "rot_skipped" | "hybrid"
    //   - ignore：位置 FDI REJECTED，完全跳过（位置也不更新）
    //   - rot_skipped：位置已更新，姿态 NIS 超门限跳过
    //   - hybrid：位置 + 姿态均更新
    // 参照：py_eskf/eskf_vio.py::update_camera_pose_hybrid（rot_gate=4.0）
    // ============================================================
    std::tuple<double,
               std::optional<Eigen::Matrix<double, 6, 1>>,
               std::optional<Eigen::Matrix<double, 4, 1>>,
               std::string>
    update_camera_pose_hybrid(
        const std::optional<Eigen::Vector3d> &p_meas_opt,
        const std::optional<Eigen::Matrix<double, 4, 1>> &q_meas_opt);

    // ============================================================
    // Camera 位姿更新 (卡方检验异常值拒绝版)
    // 来源: Python eskf_mask.py:update_camera_pose_chi2()
    // 返回：(accepted, has_dx)
    // ============================================================
    std::pair<bool, bool>
    update_camera_pose_chi2(const Eigen::Vector3d &p_meas,
                            const Eigen::Matrix<double, 4, 1> &q_meas);

    // ============================================================
    // Camera 位姿更新 (自适应噪声 + 卡方拒绝版)
    // 来源: Python eskf_mask.py:update_camera_pose_adaptive()
    // 返回：(accepted, has_dx)
    // ============================================================
    std::pair<bool, bool>
    update_camera_pose_adaptive(const Eigen::Vector3d &p_meas,
                                const Eigen::Matrix<double, 4, 1> &q_meas);

    // ============================================================
    // Radar 高度更新（1D 观测，仅 z 轴，基础版）
    // 返回：(NIS, residual)
    // ============================================================
    std::pair<double, std::optional<double>>
    update_radar_height(double radar_height);

    // ============================================================
    // Radar 高度更新（完整版，对标 Python update_altitude）
    // 来源: Python eskf_mask.py:update_altitude()
    // 返回：(nis, innovation, S)
    // noise_std_override: 外部动态传入观测噪声标准差，nullopt 时使用默认 radar_alt_noise
    // ============================================================
    std::tuple<double, double, double>
    update_altitude(double z_meas,
                    std::optional<double> noise_std_override = std::nullopt);

    // ============================================================
    // 获取/设置自适应噪声参数
    // ============================================================
    bool adaptive_noise_enabled() const;
    void set_adaptive_noise_enabled(bool enabled);
    double adaptive_noise_alpha() const;
    void set_adaptive_noise_alpha(double alpha);
    int adaptive_noise_window() const;
    void set_adaptive_noise_window(int window_size);

    // 卡方检验参数
    bool chi2_reject_enabled() const;
    void set_chi2_reject_enabled(bool enabled);
    double chi2_threshold_95() const;
    void set_chi2_threshold_95(double threshold);

    // V17 统计
    int total_updates() const;
    int rejected_updates() const;
    double last_chi2() const;

    // 公开成员
    Eigen::VectorXd x;    // 16-dim 名义状态
    Eigen::MatrixXd P;    // 15x15 误差协方差
    Eigen::MatrixXd Q;    // 15x15 过程噪声
    ESKFParams params;

private:
    // ---- Camera FDI 内部状态 ----
    std::vector<CameraFDILogEntry> camera_fdi_log;
    int    cam_fdi_update_id_;
    double cam_fdi_max_scale_;
    double cam_fdi_ewma_alpha_;
    int    cam_fdi_extreme_count_limit_;
    double cam_fdi_recover_factor_;
    int    cam_fdi_max_log_;
    double cam_fdi_rot_detect_scale_;

    std::optional<double> cam_nis_pos_ewma_;
    std::optional<double> cam_nis_rot_ewma_;
    int    cam_extreme_count_;
    std::string cam_fdi_state_;

    // ---- Radar FDI 内部状态 ----
    std::optional<double> radar_nis_ewma_;

    // ---- V17 自适应噪声 ----
    bool   adaptive_noise_enabled_ = false;
    double adaptive_noise_alpha_   = 0.9;
    int    adaptive_noise_window_  = 50;
    std::deque<double> acc_norm_hist_;
    std::deque<double> gyro_norm_hist_;
    double smoothed_sigma_acc_  = 0.3;
    double smoothed_sigma_gyro_ = 0.02;
    bool   smoothed_initialized_ = false;

    // ---- V17 卡方拒绝 ----
    bool   chi2_reject_enabled_ = false;
    double chi2_threshold_95_   = 12.6;   // χ²(6) 95%分位数
    int    total_updates_ = 0;
    int    rejected_updates_ = 0;
    double last_chi2_ = 0.0;

    // ---- 内部方法 ----
    bool _predict_nominal(const std::optional<Eigen::Vector3d> &acc_opt,
                          const std::optional<Eigen::Vector3d> &gyro_opt,
                          double dt);

    Eigen::Matrix<double, 15, 15> _inject(const Eigen::Matrix<double, 15, 1> &dx);

    std::tuple<Eigen::Vector3d, Eigen::Vector3d,
               Eigen::Matrix<double, 4, 1>,
               Eigen::Vector3d, Eigen::Vector3d>
    _unpack() const;

    void _pack(const Eigen::Vector3d &p,
               const Eigen::Vector3d &v,
               const Eigen::Matrix<double, 4, 1> &q,
               const Eigen::Vector3d &b_a,
               const Eigen::Vector3d &b_g);
};

// ============================================================
// 实现部分（header-only：全部为 inline）
// ============================================================

// ============================================================
// GravityEstimator 实现
// 来源: Python gravity_estimator.py
// ============================================================

inline
GravityEstimator::GravityEstimator(int window_size)
    : window_size_(window_size)
    , gravity_est_(0.0, 0.0, -9.81)
    , initialized_(false)
    , update_count_(0)
{
    if (window_size_ < 10) window_size_ = 10;
}

inline
void GravityEstimator::add_sample(
    double t_sec,
    const Eigen::Vector3d &p_cam,
    const std::vector<std::pair<double, Eigen::Vector3d>> &imu_steps)
{
    GravitySample s;
    s.t_sec     = t_sec;
    s.p_cam     = p_cam;
    s.imu_steps = imu_steps;
    samples_.push_back(s);

    while ((int)samples_.size() > window_size_)
        samples_.pop_front();
}

inline
int GravityEstimator::sample_count() const
{
    return (int)samples_.size();
}

inline
int GravityEstimator::window_size() const
{
    return window_size_;
}

inline
bool GravityEstimator::is_ready(int stride) const
{
    (void)stride;
    return (int)samples_.size() >= std::max(10, 2 * stride + 1);
}

inline
const Eigen::Vector3d& GravityEstimator::get_gravity() const
{
    return gravity_est_;
}

inline
int GravityEstimator::update_count() const
{
    return update_count_;
}

inline
void GravityEstimator::reset()
{
    samples_.clear();
    gravity_est_ = Eigen::Vector3d(0.0, 0.0, -9.81);
    initialized_ = false;
    update_count_ = 0;
    k_ = 0;
    g_hist_.clear();
}

// ---- 轨迹最小二乘重力估计 ----
inline
std::pair<std::optional<Eigen::Vector3d>, GravityEstimateInfo>
GravityEstimator::estimate(double tikhonov_lambda, int stride)
{
    GravityEstimateInfo info;
    int n = (int)samples_.size();
    info.n_samples = n;
    info.n_window  = window_size_;

    if (n < 10)
    {
        return {std::nullopt, info};
    }

    (void)stride;  // stride 在简化版中未使用，保留接口兼容

    // 提取时间戳和位置
    std::vector<double> timestamps(n);
    std::vector<Eigen::Vector3d> positions(n);
    for (int i = 0; i < n; ++i)
    {
        timestamps[i] = samples_[i].t_sec;
        positions[i]  = samples_[i].p_cam;
    }

    double t0 = timestamps[0];

    // 使用 IMU 步数据精确积分 p_imu
    std::vector<Eigen::Vector3d> p_imu(n, Eigen::Vector3d::Zero());
    Eigen::Vector3d v_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_acc = Eigen::Vector3d::Zero();

    for (int i = 0; i < n; ++i)
    {
        if (i == 0)
        {
            p_imu[i].setZero();
        }
        else
        {
            for (const auto &step : samples_[i].imu_steps)
            {
                double dt_s            = step.first;
                const Eigen::Vector3d &acc_s = step.second;
                if (dt_s > 1e-8)
                {
                    Eigen::Vector3d v_prev = v_acc;
                    v_acc = v_acc + acc_s * dt_s;
                    p_acc = p_acc + v_prev * dt_s
                            + 0.5 * acc_s * dt_s * dt_s;
                }
            }
            p_imu[i] = p_acc;
        }
    }

    // 观测值 y = p_cam - p_imu
    Eigen::Vector3d g_est_result = Eigen::Vector3d::Zero();
    Eigen::Vector3d residual_std = Eigen::Vector3d::Zero();

    // 先验重力
    Eigen::Vector3d g_prior = gravity_est_;
    info.g_prior = g_prior;

    for (int ax = 0; ax < 3; ++ax)
    {
        // 构建设计矩阵 A: [1, dt_i, 0.5*dt_i^2]
        Eigen::MatrixXd A(n, 3);
        Eigen::VectorXd b(n);

        for (int i = 0; i < n; ++i)
        {
            double dt_i = timestamps[i] - t0;
            A(i, 0) = 1.0;
            A(i, 1) = dt_i;
            A(i, 2) = 0.5 * dt_i * dt_i;
            b(i)    = positions[i](ax) - p_imu[i](ax);
        }

        // 权重: 帧间时间跨度
        double total_w = 0.0;
        Eigen::VectorXd weights(n);
        weights(0) = 1.0;
        total_w    = 1.0;
        for (int i = 1; i < n; ++i)
        {
            double frame_dt = timestamps[i] - timestamps[i - 1];
            weights(i)      = std::max(frame_dt, 0.001);
            total_w        += weights(i);
        }
        weights = weights / total_w * (double)n;

        // WLS: A^T W A 和 A^T W b
        Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
        Eigen::Vector3d ATb = Eigen::Vector3d::Zero();
        for (int i = 0; i < n; ++i)
        {
            double w_i   = weights(i);
            Eigen::Vector3d a_row = A.row(i).transpose();
            ATA += w_i * a_row * a_row.transpose();
            ATb += w_i * a_row * b(i);
        }

        // Tikhonov 正则化: 只约束 g 分量 (index 2)
        double lam = tikhonov_lambda;
        double g_prior_val = g_prior(ax);

        if (lam > 0)
        {
            ATA(2, 2) += lam;
            ATb(2)    += lam * g_prior_val;
        }

        // 求解
        Eigen::Vector3d coeffs;
        bool solved = false;
        if (ATA.determinant() != 0.0)
        {
            coeffs = ATA.inverse() * ATb;
            solved = true;
        }
        else
        {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
            if (svd.singularValues()(2) > 1e-12)
            {
                coeffs = svd.solve(ATb);
                solved = true;
            }
        }

        if (solved)
        {
            g_est_result(ax) = coeffs(2);
        }
        else
        {
            g_est_result(ax) = g_prior_val;
        }

        // 残差标准差
        Eigen::VectorXd residual = A * coeffs - b;
        residual_std(ax) = std::sqrt(
            (residual.array() * residual.array()).sum() / (double)n);
    }

    // 约束模长到 9.81
    double g_norm = g_est_result.norm();
    if (g_norm > 0.1)
    {
        g_est_result = g_est_result * (9.81 / g_norm);
    }

    return {g_est_result, info};
}

// ---- 可观性分析 ----
inline
GravityEstimator::ObservabilityResult
GravityEstimator::compute_observability(int stride)
{
    ObservabilityResult result;
    int n = (int)samples_.size();

    if (n < 10)
    {
        result.detail = "insufficient_data";
        return result;
    }

    (void)stride;

    // 提取位置
    std::vector<Eigen::Vector3d> positions(n);
    std::vector<double> timestamps(n);
    for (int i = 0; i < n; ++i)
    {
        positions[i]  = samples_[i].p_cam;
        timestamps[i] = samples_[i].t_sec;
    }

    double time_span = timestamps.back() - timestamps.front();
    result.time_span = time_span;

    // 位置位移
    Eigen::Vector3d p0 = positions[0];
    double max_disp = 0.0;
    Eigen::Vector3d max_disp_per_axis(0, 0, 0);
    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector3d d = positions[i] - p0;
        double dist = d.norm();
        if (dist > max_disp) max_disp = dist;
        for (int ax = 0; ax < 3; ++ax)
            if (std::abs(d(ax)) > max_disp_per_axis(ax))
                max_disp_per_axis(ax) = std::abs(d(ax));
    }
    result.pos_disp = max_disp;

    // 可观性评分
    result.obs_score_x = std::min(time_span / 2.0, 1.0)
                         * std::min(max_disp_per_axis(0) / 0.2, 1.0);
    result.obs_score_y = std::min(time_span / 2.0, 1.0)
                         * std::min(max_disp_per_axis(1) / 0.2, 1.0);
    result.obs_score_z = std::min(time_span / 2.0, 1.0)
                         * std::min(max_disp_per_axis(2) / 0.2, 1.0);

    result.x_obs = result.obs_score_x > obs_score_threshold_;
    result.y_obs = result.obs_score_y > obs_score_threshold_;
    result.z_obs = result.obs_score_z > obs_score_threshold_;
    result.all_obs = result.x_obs && result.y_obs && result.z_obs;

    result.detail = "time_span=" + std::to_string(time_span)
                    + "s, pos_disp=" + std::to_string(max_disp) + "m";

    return result;
}

// ---- EWMA 递推平滑 ----
inline
std::pair<Eigen::Vector3d, double>
GravityEstimator::ewma_smooth(const Eigen::Vector3d &g_new)
{
    update_count_++;
    k_++;

    if (!initialized_)
    {
        gravity_est_ = g_new;
        initialized_ = true;
        return {gravity_est_, 1.0};
    }

    // a = 1/k：等效对历史 g_new 等权平均（全局加权平均的在线等价）。
    // 后期 k 大 → 每帧更新量 ∝ 1/k → 自动趋零，输出参考系自然稳定。
    double a = 1.0 / (double)k_;
    gravity_est_ = (1.0 - a) * gravity_est_ + a * g_new;

    // 约束模长
    double g_norm = gravity_est_.norm();
    if (g_norm > 0.1)
    {
        gravity_est_ = gravity_est_ * (9.81 / g_norm);
    }

    return {gravity_est_, a};
}

// ---- 便捷更新接口：短窗口 LS → EWMA 递推 ----
inline
std::pair<std::optional<Eigen::Vector3d>, GravityEstimateInfo>
GravityEstimator::update(double tikhonov_lambda,
                         int estimate_stride,
                         int obs_stride,
                         bool freeze_enabled,
                         double t_sec)
{
    (void)estimate_stride;
    (void)obs_stride;
    (void)freeze_enabled;

    if (!is_ready(obs_stride))
        return {std::nullopt, GravityEstimateInfo{}};

    // 短窗口轨迹最小二乘估计
    auto [g_new_opt, est_info] = estimate(tikhonov_lambda);

    if (!g_new_opt.has_value())
        return {std::nullopt, est_info};

    // EWMA 递推平滑
    auto [g_smooth, a] = ewma_smooth(g_new_opt.value());

    // 记录收敛历史（带时间戳，供 convergence_angle_deg 使用）
    if (t_sec >= 0.0)
    {
        g_hist_.push_back({t_sec, gravity_est_});
        while ((int)g_hist_.size() > 100000)
            g_hist_.pop_front();
    }

    GravityEstimateInfo result_info = est_info;
    result_info.n_window = (int)(a * 1000);

    return {g_smooth, result_info};
}

// ---- 收敛判据：最近 window_s 秒内 g_est 方向变化（度）----
inline
double GravityEstimator::convergence_angle_deg(double window_s) const
{
    if (g_hist_.size() < 2 || gravity_est_.norm() < 0.1)
        return std::numeric_limits<double>::infinity();

    double t_now = g_hist_.back().first;
    double t_ref = t_now - window_s;

    // 找到时间 <= t_ref 的最新历史记录
    const Eigen::Vector3d *g_ref = nullptr;
    for (auto it = g_hist_.rbegin(); it != g_hist_.rend(); ++it)
    {
        if (it->first <= t_ref)
        {
            g_ref = &it->second;
            break;
        }
    }
    if (!g_ref)
        return std::numeric_limits<double>::infinity();

    double cos_v = gravity_est_.dot(*g_ref)
                   / (gravity_est_.norm() * g_ref->norm() + 1e-12);
    cos_v = clamp(cos_v, -1.0, 1.0);
    return std::acos(cos_v) * 180.0 / std::acos(-1.0);
}

inline
bool GravityEstimator::is_converged(double window_s, double thresh_deg) const
{
    return convergence_angle_deg(window_s) < thresh_deg;
}

// ---- 从估计重力恢复转回矩阵 R_est ----
// Rodrigues 最小旋转：把标准重力方向 [0,0,-1] 转到估计方向 ĝ
// 输出变换: p_w = R_estᵀ·p'，q_w = q_est⁻¹ ⊗ q'（tilt-exp 文档 §6）
inline
Eigen::Matrix3d GravityEstimator::recover_rotation() const
{
    Eigen::Vector3d a(0.0, 0.0, -1.0);          // g_world 归一化方向
    Eigen::Vector3d b = gravity_est_.normalized();
    if (b.norm() < 0.1) b = a;
    return Eigen::Quaterniond::FromTwoVectors(a, b).toRotationMatrix();
}

// ============================================================
// RadarAltimeter 实现
// 来源: Python radar_altimeter.py
// ============================================================

inline
RadarAltimeter::RadarAltimeter(double radar_noise_std,
                               double nis_threshold,
                               double jump_threshold,
                               int window_size,
                               int fail_threshold,
                               int recovery_threshold)
    : radar_noise_std(radar_noise_std)
    , nis_threshold(nis_threshold)
    , jump_threshold(jump_threshold)
    , window_size_(window_size)
    , last_valid_z_(0.0)
    , last_valid_t_(-1.0)
    , total_calls_(0)
    , accepted_(0)
    , rejected_(0)
    , reject_jump_(0)
    , reject_nis_(0)
    , reject_no_data_(0)
    , fail_threshold_(fail_threshold)
    , recovery_threshold_(recovery_threshold)
{
    if (window_size_ < 3) window_size_ = 3;
    if (fail_threshold_ < 1) fail_threshold_ = 1;
    if (recovery_threshold_ < 1) recovery_threshold_ = 1;
}

inline
std::tuple<bool, double, RadarAltValidateInfo>
RadarAltimeter::validate(double z_meas, double z_pred,
                         double P_z_pred, double t)
{
    total_calls_++;
    RadarAltValidateInfo info;
    info.nis_threshold = nis_threshold;
    info.jump_threshold = jump_threshold;
    info.radar_failed   = radar_failed_;
    info.should_skip_update = radar_failed_;
    info.effective_noise_std = radar_noise_std;

    // 检查 NaN 或 无效值
    if (!std::isfinite(z_meas))
    {
        rejected_++;
        reject_no_data_++;
        consecutive_reject_ += 1;
        _check_failure_state();
        info.reason         = "no_data";
        info.radar_failed   = radar_failed_;
        info.should_skip_update = radar_failed_;
        return {false, 0.0, info};
    }

    // 1. 高度跳变检测
    if (total_calls_ > 1 && std::isfinite(last_valid_z_))
    {
        double jump = std::abs(z_meas - last_valid_z_);
        if (jump > jump_threshold)
        {
            rejected_++;
            reject_jump_++;
            consecutive_reject_ += 1;
            _check_failure_state();
            info.reason         = "jump";
            info.jump_magnitude = jump;
            info.radar_failed   = radar_failed_;
            info.should_skip_update = radar_failed_;
            return {false, 0.0, info};
        }
    }

    // 2. NIS 检验
    double innovation = z_meas - z_pred;
    double S = P_z_pred + radar_noise_std * radar_noise_std;

    if (S < 1e-12)
    {
        rejected_++;
        reject_nis_++;
        consecutive_reject_ += 1;
        _check_failure_state();
        info.reason       = "nis_degenerate";
        info.innovation   = innovation;
        info.S            = S;
        info.radar_failed = radar_failed_;
        info.should_skip_update = radar_failed_;
        return {false, 0.0, info};
    }

    double nis = innovation * innovation / S;

    if (nis > nis_threshold)
    {
        rejected_++;
        reject_nis_++;
        consecutive_reject_ += 1;
        _check_failure_state();
        info.reason       = "nis";
        info.nis          = nis;
        info.innovation   = innovation;
        info.S            = S;
        info.radar_failed = radar_failed_;
        info.should_skip_update = radar_failed_;
        return {false, nis, info};
    }

    // 通过检验 — 接受
    accepted_++;
    consecutive_reject_ = 0;       // 重置连续拒绝计数
    consecutive_accept_ += 1;
    _check_recovery_state();

    last_valid_z_ = z_meas;
    if (t >= 0)
        last_valid_t_ = t;

    recent_valid_z_.push_back(z_meas);
    if (t >= 0)
        recent_valid_t_.push_back(t);

    while ((int)recent_valid_z_.size() > window_size_)
        recent_valid_z_.pop_front();
    while ((int)recent_valid_t_.size() > window_size_)
        recent_valid_t_.pop_front();

    info.reason     = "accepted";
    info.nis        = nis;
    info.innovation = innovation;
    info.S          = S;
    info.accept_rate = (double)accepted_ / (double)std::max(total_calls_, 1);
    info.radar_failed   = radar_failed_;
    info.should_skip_update = radar_failed_;
    info.effective_noise_std = radar_noise_std;

    // 窗口统计
    if (!recent_valid_z_.empty())
    {
        double sum = 0.0;
        for (double v : recent_valid_z_) sum += v;
        info.window_mean = sum / (double)recent_valid_z_.size();

        if (recent_valid_z_.size() >= 2)
        {
            double sum_sq = 0.0;
            for (double v : recent_valid_z_)
                sum_sq += (v - info.window_mean) * (v - info.window_mean);
            info.window_std = std::sqrt(
                sum_sq / (double)recent_valid_z_.size());
        }
    }

    return {true, nis, info};
}

inline
void RadarAltimeter::_check_failure_state()
{
    if (!radar_failed_ && consecutive_reject_ >= fail_threshold_)
    {
        radar_failed_ = true;
    }
}

inline
void RadarAltimeter::_check_recovery_state()
{
    if (radar_failed_ && consecutive_accept_ >= recovery_threshold_)
    {
        radar_failed_ = false;
    }
}

inline
RadarAltStatistics RadarAltimeter::get_statistics() const
{
    RadarAltStatistics s;
    s.total_calls    = total_calls_;
    s.accepted       = accepted_;
    s.rejected       = rejected_;
    s.accept_rate    = (double)accepted_
                       / (double)std::max(total_calls_, 1);
    s.reject_jump    = reject_jump_;
    s.reject_nis     = reject_nis_;
    s.reject_no_data = reject_no_data_;
    s.nis_threshold  = nis_threshold;
    s.jump_threshold = jump_threshold;
    s.radar_noise_std = radar_noise_std;
    return s;
}

// ============================================================
// ESKF_VIO 实现
// 来源: Python eskf_mask.py + 原 eskf_vio.hpp
// ============================================================

inline
ESKF_VIO::ESKF_VIO(const Eigen::VectorXd &x0,
                   const Eigen::MatrixXd &P0,
                   const ESKFParams &params_in)
    : x(x0), P(P0), params(params_in)
{
    if (x.size() != 16)
        throw std::runtime_error("x0 must be 16-dim");
    if (P.rows() != 15 || P.cols() != 15)
        throw std::runtime_error("P0 must be 15x15");

    Q = Eigen::MatrixXd::Identity(15, 15) * 1e-4;

    // Camera FDI 参数初始化
    cam_fdi_max_scale_           = params.cam_fdi_max_scale;
    cam_fdi_ewma_alpha_          = params.cam_fdi_ewma_alpha;
    cam_fdi_extreme_count_limit_ = params.cam_fdi_extreme_count;
    cam_fdi_recover_factor_      = params.cam_fdi_recover_factor;
    cam_fdi_max_log_             = params.cam_fdi_max_log;
    cam_fdi_rot_detect_scale_    = params.cam_fdi_rot_detect_scale;

    // Camera FDI 状态初始化
    cam_nis_pos_ewma_  = std::nullopt;
    cam_nis_rot_ewma_  = std::nullopt;
    cam_extreme_count_ = 0;
    cam_fdi_state_     = "NORMAL";
    cam_fdi_update_id_ = 0;

    // Radar FDI 状态初始化
    radar_nis_ewma_ = std::nullopt;
}

inline
const std::vector<CameraFDILogEntry>&
ESKF_VIO::get_camera_fdi_log() const
{
    return camera_fdi_log;
}

// ---- Camera FDI 评估 ----
inline
std::pair<Eigen::Matrix<double, 6, 6>, CameraFDIResult>
ESKF_VIO::camera_fdi_evaluate(const Eigen::Matrix<double, 6, 1> &z,
                              const Eigen::Matrix<double, 6, 15> &H,
                              const Eigen::Matrix<double, 6, 6> &R)
{
    // S = H P H^T + R
    Eigen::Matrix<double, 6, 6> S = H * P * H.transpose() + R;

    Eigen::Vector3d z_pos = z.segment<3>(0);
    Eigen::Vector3d z_rot = z.segment<3>(3);
    Eigen::Matrix3d S_pos = S.block<3, 3>(0, 0);
    Eigen::Matrix3d S_rot = S.block<3, 3>(3, 3);

    // 求逆
    Eigen::Matrix3d inv_S_pos, inv_S_rot;

    if (S_pos.determinant() != 0.0)
        inv_S_pos = S_pos.inverse();
    else
        inv_S_pos = detail::svd_pinv_3x3(S_pos);

    if (S_rot.determinant() != 0.0)
        inv_S_rot = S_rot.inverse();
    else
        inv_S_rot = detail::svd_pinv_3x3(S_rot);

    double nis_pos = z_pos.transpose() * inv_S_pos * z_pos;
    double nis_rot = z_rot.transpose() * inv_S_rot * z_rot;

    double nis_pos_used = nis_pos;
    double nis_rot_used = nis_rot;

    // EWMA 平滑
    if (cam_fdi_ewma_alpha_ > 0.0)
    {
        double alpha = cam_fdi_ewma_alpha_;
        if (!cam_nis_pos_ewma_.has_value())
        {
            cam_nis_pos_ewma_ = nis_pos;
            cam_nis_rot_ewma_ = nis_rot;
        }
        else
        {
            cam_nis_pos_ewma_ = alpha * nis_pos
                              + (1.0 - alpha) * (*cam_nis_pos_ewma_);
            cam_nis_rot_ewma_ = alpha * nis_rot
                              + (1.0 - alpha) * (*cam_nis_rot_ewma_);
        }
        nis_pos_used = *cam_nis_pos_ewma_;
        nis_rot_used = *cam_nis_rot_ewma_;
    }

    nis_rot_used *= cam_fdi_rot_detect_scale_;

    // 卡方门限（3 DOF）
    const double chi2_pos_95  = 7.8147279032511;
    const double chi2_pos_999 = 16.266236196238;

    bool pos_extreme = (nis_pos_used > chi2_pos_999);
    bool rot_extreme = (nis_rot_used > chi2_pos_999);
    bool pos_susp    = (nis_pos_used > chi2_pos_95);
    bool rot_susp    = (nis_rot_used > chi2_pos_95);

    double chi2_recover = cam_fdi_recover_factor_ * chi2_pos_95;
    bool pos_good = (nis_pos_used < chi2_recover);
    bool rot_good = (nis_rot_used < chi2_recover);

    // 状态机更新
    if (pos_extreme || rot_extreme)
        cam_extreme_count_ = std::min(cam_extreme_count_ + 1, 1000000);
    else
        cam_extreme_count_ = std::max(0, cam_extreme_count_ - 1);

    if (cam_fdi_state_ == "NORMAL")
    {
        if (pos_susp || rot_susp) cam_fdi_state_ = "SUSPECT";
        if (cam_extreme_count_ >= cam_fdi_extreme_count_limit_)
        {
            cam_fdi_state_ = "REJECTED";
            cam_extreme_count_ = 0;
        }
    }
    else if (cam_fdi_state_ == "SUSPECT")
    {
        if (cam_extreme_count_ >= cam_fdi_extreme_count_limit_)
        {
            cam_fdi_state_ = "REJECTED";
            cam_extreme_count_ = 0;
        }
        else if (pos_good && rot_good)
        {
            cam_fdi_state_ = "NORMAL";
            cam_extreme_count_ = 0;
        }
    }
    else if (cam_fdi_state_ == "REJECTED")
    {
        if (pos_good && rot_good)
        {
            cam_fdi_state_ = "NORMAL";
            cam_extreme_count_ = 0;
        }
    }

    Eigen::Matrix<double, 6, 6> R_eff = R;
    CameraFDIResult res;
    res.info.state = cam_fdi_state_;

    if (cam_fdi_state_ == "REJECTED")
    {
        // 2026-08-12：回退 08-11 的降级方案（REJECTED 恢复完全跳过，action="ignore"）。
        // 依据：《ESKF 融合 — 修改记录与改进方向》实机/仿真验证——视觉端坏帧（姿态
        // 翻转，等价角速度 ~2500°/s）由 FDI 正确拒掉是预期行为（拒检直接丢弃），
        // 降级吸收（R_eff×max_scale）会弱吸收坏帧，反而拖累状态。
        res.action   = "ignore";
        res.nis_pos  = nis_pos;
        res.nis_rot  = nis_rot;
        res.S_pos    = S_pos;
        res.S_rot    = S_rot;
        res.info.reason = "state_rejected";
        return {R_eff, res};
    }

    double max_scale = cam_fdi_max_scale_;
    double scale_pos = pos_susp
        ? std::min(std::max(nis_pos_used / chi2_pos_95, 1.0), max_scale)
        : 1.0;
    double scale_rot = rot_susp
        ? std::min(std::max(nis_rot_used / chi2_pos_95, 1.0), max_scale)
        : 1.0;

    R_eff.block<3, 3>(0, 0) *= scale_pos;
    R_eff.block<3, 3>(3, 3) *= scale_rot;

    if      (scale_pos > 1.0 && scale_rot > 1.0) res.action = "adaptive_both";
    else if (scale_pos > 1.0)                     res.action = "adaptive_pos";
    else if (scale_rot > 1.0)                     res.action = "adaptive_rot";
    else                                          res.action = "accept";

    res.nis_pos = nis_pos;
    res.nis_rot = nis_rot;
    res.S_pos   = S_pos;
    res.S_rot   = S_rot;
    res.info.numeric["scale_pos"]   = scale_pos;
    res.info.numeric["scale_rot"]   = scale_rot;
    res.info.numeric["pos_susp"]    = pos_susp ? 1.0 : 0.0;
    res.info.numeric["rot_susp"]    = rot_susp ? 1.0 : 0.0;
    res.info.numeric["nis_pos_used"] = nis_pos_used;
    res.info.numeric["nis_rot_used"] = nis_rot_used;

    return {R_eff, res};
}

// ---- IMU 预测（基础版）----
inline
bool ESKF_VIO::predict(const std::optional<Eigen::Vector3d> &acc_opt,
                       const std::optional<Eigen::Vector3d> &gyro_opt,
                       double dt)
{
    return _predict_nominal(acc_opt, gyro_opt, dt);
}

// ---- IMU 预测（自适应噪声版）----
inline
bool ESKF_VIO::predict_adaptive(const Eigen::Vector3d &acc,
                                const Eigen::Vector3d &gyro,
                                double dt,
                                Eigen::Vector3d *acc_world_out)
{
    if (!adaptive_noise_enabled_)
    {
        bool ok = _predict_nominal(acc, gyro, dt);
        if (ok && acc_world_out)
        {
            auto [p, v, q, b_a, b_g] = _unpack();
            (void)p; (void)v; (void)b_g;
            Eigen::Vector3d acc_u   = acc - b_a;
            Eigen::Vector3d acc_cam = params.R_imu_cam * acc_u;
            Eigen::Matrix3d R = quat_to_rot(q);
            *acc_world_out = R * acc_cam;
        }
        return ok;
    }

    // 存储范数历史
    double acc_norm  = acc.norm();
    double gyro_norm = gyro.norm();
    acc_norm_hist_.push_back(acc_norm);
    gyro_norm_hist_.push_back(gyro_norm);

    while ((int)acc_norm_hist_.size() > adaptive_noise_window_)
        acc_norm_hist_.pop_front();
    while ((int)gyro_norm_hist_.size() > adaptive_noise_window_)
        gyro_norm_hist_.pop_front();

    if ((int)acc_norm_hist_.size() >= 10)
    {
        // 计算历史标准差
        double acc_mean  = 0.0, gyro_mean = 0.0;
        for (double v : acc_norm_hist_)  acc_mean  += v;
        for (double v : gyro_norm_hist_) gyro_mean += v;
        acc_mean  /= (double)acc_norm_hist_.size();
        gyro_mean /= (double)gyro_norm_hist_.size();

        double acc_var  = 0.0, gyro_var = 0.0;
        for (double v : acc_norm_hist_)
            acc_var += (v - acc_mean) * (v - acc_mean);
        for (double v : gyro_norm_hist_)
            gyro_var += (v - gyro_mean) * (v - gyro_mean);
        acc_var  /= (double)acc_norm_hist_.size();
        gyro_var /= (double)gyro_norm_hist_.size();

        double acc_std  = std::sqrt(acc_var);
        double gyro_std = std::sqrt(gyro_var);

        double base_acc  = params.sigma_acc;
        double base_gyro = params.sigma_gyro;
        double scale_acc = clamp(acc_std / (base_acc + 1e-6), 0.5, 3.0);
        double scale_gyro = clamp(gyro_std / (base_gyro + 1e-6), 0.5, 3.0);

        double target_sigma_acc  = base_acc  * scale_acc;
        double target_sigma_gyro = base_gyro * scale_gyro;

        if (!smoothed_initialized_)
        {
            smoothed_sigma_acc_  = target_sigma_acc;
            smoothed_sigma_gyro_ = target_sigma_gyro;
            smoothed_initialized_ = true;
        }
        else
        {
            double alpha = adaptive_noise_alpha_;
            smoothed_sigma_acc_  = alpha * smoothed_sigma_acc_
                                 + (1.0 - alpha) * target_sigma_acc;
            smoothed_sigma_gyro_ = alpha * smoothed_sigma_gyro_
                                 + (1.0 - alpha) * target_sigma_gyro;
        }

        // 临时覆盖噪声参数
        double orig_acc  = params.sigma_acc;
        double orig_gyro = params.sigma_gyro;
        params.sigma_acc  = smoothed_sigma_acc_;
        params.sigma_gyro = smoothed_sigma_gyro_;

        bool ok = _predict_nominal(acc, gyro, dt);

        params.sigma_acc  = orig_acc;
        params.sigma_gyro = orig_gyro;

        if (ok && acc_world_out)
        {
            auto [p, v, q, b_a, b_g] = _unpack();
            (void)p; (void)v; (void)b_g;
            Eigen::Vector3d acc_u   = acc - b_a;
            Eigen::Vector3d acc_cam = params.R_imu_cam * acc_u;
            Eigen::Matrix3d R = quat_to_rot(q);
            *acc_world_out = R * acc_cam;
        }
        return ok;
    }
    else
    {
        bool ok = _predict_nominal(acc, gyro, dt);
        if (ok && acc_world_out)
        {
            auto [p, v, q, b_a, b_g] = _unpack();
            (void)p; (void)v; (void)b_g;
            Eigen::Vector3d acc_u   = acc - b_a;
            Eigen::Vector3d acc_cam = params.R_imu_cam * acc_u;
            Eigen::Matrix3d R = quat_to_rot(q);
            *acc_world_out = R * acc_cam;
        }
        return ok;
    }
}

// ---- Camera 位姿更新（基础版）----
inline
std::tuple<double,
           std::optional<Eigen::Matrix<double, 6, 1>>,
           std::optional<Eigen::Matrix<double, 4, 1>>>
ESKF_VIO::update_camera_pose(
    const std::optional<Eigen::Vector3d> &p_meas_opt,
    const std::optional<Eigen::Matrix<double, 4, 1>> &q_meas_opt)
{
    cam_fdi_update_id_ += 1;

    if (!p_meas_opt.has_value() || !q_meas_opt.has_value())
    {
        return {-1.0, std::nullopt, std::nullopt};
    }

    // 构造测量噪声 R（支持 Z 轴解耦）
    // 2026-08-12：cam_*_noise 为 σ，R 应为 σ²（与 update_camera_pose_adaptive
    // 的 R_mat 一致）；原 σ 直接当方差使位置 R 放大 ~4.8×、旋转 ~17×，融合过度保守。
    // 依据：《ESKF 融合 — 修改记录与改进方向》§1.1/§1.3（R² 修复保留，门限不动）
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
    if (params.use_cam_z_noise_decoupling)
    {
        R(0, 0) = params.cam_pos_noise * params.cam_pos_noise;
        R(1, 1) = params.cam_pos_noise * params.cam_pos_noise;
        R(2, 2) = params.cam_pos_z_noise * params.cam_pos_z_noise;  // Z 轴放大噪声
    }
    else
    {
        for (int i = 0; i < 3; ++i)
            R(i, i) = params.cam_pos_noise * params.cam_pos_noise;
    }
    for (int i = 0; i < 3; ++i)
        R(i + 3, i + 3) = params.cam_rot_noise * params.cam_rot_noise;

    // 观测矩阵 H
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

    // 计算残差
    auto [p, v, q, b_a, b_g] = _unpack();
    (void)v; (void)b_a; (void)b_g;

    // 杆臂补偿：相机观测 (p_meas) 是相机系原点在世界系中的位置
    // 状态 p 是 IMU 在世界系中的位置（初始由相机位姿近似）
    // 预期相机位置 = p - R(q) * p_imu_in_cam
    // 残差 dp = p_meas - 预期相机位置
    Eigen::Matrix3d Rmat = quat_to_rot(q);
    Eigen::Vector3d cam_expected = p - Rmat * params.p_imu_in_cam;
    Eigen::Vector3d dp = *p_meas_opt - cam_expected;

    // 右误差约定（body 系）：dq = q_nom⁻¹ ⊗ q_meas，
    // 与 F 矩阵 -[ω]× 及注入 q ⊗ exp(δθ/2) 一致（对应 Python eskf_vio.py 修复）
    Eigen::Matrix<double, 4, 1> dq = quat_mul(quat_inv(q), *q_meas_opt);
    if (dq(0) < 0.0) dq *= -1.0;  // 短路径检查

    // 四元数残差 -> 3D 旋转向量
    double qw    = clamp(dq(0), -1.0, 1.0);
    double angle = 2.0 * std::acos(qw);
    Eigen::Vector3d dtheta;
    if (angle < 1e-6)
    {
        dtheta = 2.0 * dq.segment<3>(1);
    }
    else
    {
        double sin_half = std::sqrt(std::max(0.0, 1.0 - qw * qw));
        if (sin_half < 1e-8)
        {
            dtheta = 2.0 * dq.segment<3>(1);
        }
        else
        {
            Eigen::Vector3d axis = dq.segment<3>(1) / sin_half;
            dtheta = angle * axis;
        }
    }

    Eigen::Matrix<double, 6, 1> z;
    z.segment<3>(0) = dp;
    z.segment<3>(3) = dtheta;

    // FDI 评估
    auto [R_eff, fdi_res] = camera_fdi_evaluate(z, H, R);
    std::string action = fdi_res.action;
    double nis_pos     = fdi_res.nis_pos;
    double nis_rot     = fdi_res.nis_rot;

    // 记录日志
    CameraFDILogEntry entry;
    entry.id       = cam_fdi_update_id_;
    entry.action   = action;
    entry.nis_pos  = nis_pos;
    entry.nis_rot  = nis_rot;
    entry.z        = z;
    entry.dq       = dq;
    entry.info_str = fdi_res.info.state;
    camera_fdi_log.push_back(entry);
    if ((int)camera_fdi_log.size() > cam_fdi_max_log_)
        camera_fdi_log.erase(camera_fdi_log.begin());

    if (action == "ignore")
    {
        double NIS_total = (std::isfinite(nis_pos) ? nis_pos : 0.0)
                         + (std::isfinite(nis_rot) ? nis_rot : 0.0);
        return {NIS_total, std::nullopt, std::nullopt};
    }

    // 计算 Kalman 增益
    Eigen::Matrix<double, 6, 6> S_eff = H * P * H.transpose() + R_eff;
    Eigen::Matrix<double, 15, 6> K;

    if (S_eff.determinant() != 0.0)
        K = P * H.transpose() * S_eff.inverse();
    else
        K = P * H.transpose() * detail::svd_pinv_6x6(S_eff);

    Eigen::Matrix<double, 15, 1> dx = K * z;

    // 注入误差到名义态
    Eigen::Matrix<double, 15, 15> G_reset =
        Eigen::Matrix<double, 15, 15>::Identity();
    bool got_G_reset = false;
    try
    {
        G_reset = _inject(dx);
        got_G_reset = true;
    }
    catch (...)
    {
        try { _inject(dx); } catch (...) {}
    }

    // Joseph 形式协方差更新
    Eigen::Matrix<double, 15, 15> I    =
        Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix<double, 15, 15> I_KH = I - K * H;
    P = I_KH * P * I_KH.transpose() + K * R_eff * K.transpose();

    if (got_G_reset)
    {
        try { P = G_reset * P * G_reset.transpose(); } catch (...) {}
    }

    // 对称化 + 数值稳定
    P = 0.5 * (P + P.transpose());
    P += 1e-9 * Eigen::Matrix<double, 15, 15>::Identity();

    double NIS_total = (std::isfinite(nis_pos) ? nis_pos : 0.0)
                     + (std::isfinite(nis_rot) ? nis_rot : 0.0);
    return {NIS_total, z, dq};
}

// ============================================================
// Camera 位姿更新（hybrid：位置永远更新 + 姿态自洽追加）
// 参照 py_eskf/eskf_vio.py::update_camera_pose_hybrid（rot_gate=4.0）。
// 动机：位置永远可信（尾部 posNIS 全 <5），姿态间歇性翻转（视觉端 PNP
// 解翻转，相邻帧跳 58°~123°）。6-DOF 一刀切拒检把"位置好姿态坏"的帧
// 整个扔掉 → x/y 惯性漂移 ±330m。解耦后：位置 3-DOF 永远更新（FDI 只吃
// 位置 NIS），姿态 3-DOF 仅 rotNIS 自洽（×rot_detect_scale < χ²₉₅）时追加。
// 返回：(nis_total, z6d_opt, dq_opt, action)
// ============================================================
inline
std::tuple<double,
           std::optional<Eigen::Matrix<double, 6, 1>>,
           std::optional<Eigen::Matrix<double, 4, 1>>,
           std::string>
ESKF_VIO::update_camera_pose_hybrid(
    const std::optional<Eigen::Vector3d> &p_meas_opt,
    const std::optional<Eigen::Matrix<double, 4, 1>> &q_meas_opt)
{
    cam_fdi_update_id_ += 1;

    if (!p_meas_opt.has_value() || !q_meas_opt.has_value())
    {
        return {-1.0, std::nullopt, std::nullopt, "no_data"};
    }

    // ---- 1. 位置 3-DOF 更新（H=[I3 0..0]，R=diag(σx²,σy²,σz²)）----
    Eigen::Matrix<double, 3, 15> H_pos = Eigen::Matrix<double, 3, 15>::Zero();
    H_pos.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d R_pos = Eigen::Matrix3d::Zero();
    R_pos(0, 0) = params.cam_pos_noise * params.cam_pos_noise;
    R_pos(1, 1) = params.cam_pos_noise * params.cam_pos_noise;
    R_pos(2, 2) = (params.use_cam_z_noise_decoupling
                       ? params.cam_pos_z_noise * params.cam_pos_z_noise
                       : params.cam_pos_noise * params.cam_pos_noise);

    // 杆臂补偿：残差 z_pos = p_meas - (p - R(q)·p_imu_in_cam)
    auto [p, v, q, b_a, b_g] = _unpack();
    (void)v; (void)b_a; (void)b_g;
    Eigen::Matrix3d Rmat = quat_to_rot(q);
    Eigen::Vector3d cam_expected = p - Rmat * params.p_imu_in_cam;
    Eigen::Vector3d z_pos = *p_meas_opt - cam_expected;

    Eigen::Matrix3d S_pos = H_pos * P * H_pos.transpose() + R_pos;
    Eigen::Matrix3d inv_S_pos;
    if (S_pos.determinant() != 0.0)
        inv_S_pos = S_pos.inverse();
    else
        inv_S_pos = detail::svd_pinv_3x3(S_pos);
    double nis_pos = z_pos.transpose() * inv_S_pos * z_pos;

    // ---- 位置 FDI（3 态状态机，只吃位置 NIS；镜像 camera_fdi_evaluate）----
    const double chi2_pos_95  = 7.8147279032511;
    const double chi2_pos_999 = 16.266236196238;

    double nis_pos_used = nis_pos;
    // EWMA 平滑（α>0 才启用；只更新 pos，不动 rot EWMA）
    if (cam_fdi_ewma_alpha_ > 0.0)
    {
        double alpha = cam_fdi_ewma_alpha_;
        if (!cam_nis_pos_ewma_.has_value())
            cam_nis_pos_ewma_ = nis_pos;
        else
            cam_nis_pos_ewma_ = alpha * nis_pos
                              + (1.0 - alpha) * (*cam_nis_pos_ewma_);
        nis_pos_used = *cam_nis_pos_ewma_;
    }

    bool pos_extreme = (nis_pos_used > chi2_pos_999);
    bool pos_susp    = (nis_pos_used > chi2_pos_95);
    double chi2_recover = cam_fdi_recover_factor_ * chi2_pos_95;
    bool pos_good    = (nis_pos_used < chi2_recover);

    if (pos_extreme)
        cam_extreme_count_ = std::min(cam_extreme_count_ + 1, 1000000);
    else
        cam_extreme_count_ = std::max(0, cam_extreme_count_ - 1);

    if (cam_fdi_state_ == "NORMAL")
    {
        if (pos_susp) cam_fdi_state_ = "SUSPECT";
        if (cam_extreme_count_ >= cam_fdi_extreme_count_limit_)
        {
            cam_fdi_state_ = "REJECTED";
            cam_extreme_count_ = 0;
        }
    }
    else if (cam_fdi_state_ == "SUSPECT")
    {
        if (cam_extreme_count_ >= cam_fdi_extreme_count_limit_)
        {
            cam_fdi_state_ = "REJECTED";
            cam_extreme_count_ = 0;
        }
        else if (pos_good)
        {
            cam_fdi_state_ = "NORMAL";
            cam_extreme_count_ = 0;
        }
    }
    else if (cam_fdi_state_ == "REJECTED")
    {
        if (pos_good)
        {
            cam_fdi_state_ = "NORMAL";
            cam_extreme_count_ = 0;
        }
    }

    // 日志（与 6-DOF 同构；py 在状态机后记录 "pos_update"）
    {
        CameraFDILogEntry entry;
        entry.id       = cam_fdi_update_id_;
        entry.action   = "pos_update";
        entry.nis_pos  = nis_pos;
        entry.nis_rot  = 0.0;
        entry.z.setZero();
        entry.z.segment<3>(0) = z_pos;
        entry.dq << 1.0, 0.0, 0.0, 0.0;   // 无姿态残差（py: dq=None）
        entry.info_str = cam_fdi_state_;
        camera_fdi_log.push_back(entry);
        if ((int)camera_fdi_log.size() > cam_fdi_max_log_)
            camera_fdi_log.erase(camera_fdi_log.begin());
    }

    if (cam_fdi_state_ == "REJECTED")
    {
        // 位置状态机拒绝：完全跳过（位置也不更新），同 6-DOF REJECTED 语义。
        // 2026-08-12 已回退降级吸收方案，勿再实施。
        double nis_total = std::isfinite(nis_pos) ? nis_pos : 0.0;
        return {nis_total, std::nullopt, std::nullopt, "ignore"};
    }

    // 位置异常时 R 按 NIS 比例放大（自适应吸收，与 6-DOF 同构）
    double scale_pos = pos_susp
        ? std::min(std::max(nis_pos_used / chi2_pos_95, 1.0), cam_fdi_max_scale_)
        : 1.0;
    Eigen::Matrix3d R_pos_eff = R_pos * scale_pos;

    Eigen::Matrix<double, 3, 3> S_k = H_pos * P * H_pos.transpose() + R_pos_eff;
    Eigen::Matrix<double, 15, 3> K_pos;
    if (S_k.determinant() != 0.0)
        K_pos = P * H_pos.transpose() * S_k.inverse();
    else
        K_pos = P * H_pos.transpose() * detail::svd_pinv_3x3(S_k);
    Eigen::Matrix<double, 15, 1> dx_pos = K_pos * z_pos;

    Eigen::Matrix<double, 15, 15> G_reset =
        Eigen::Matrix<double, 15, 15>::Identity();
    bool got_G_reset = false;
    try
    {
        G_reset = _inject(dx_pos);
        got_G_reset = true;
    }
    catch (...)
    {
        try { _inject(dx_pos); } catch (...) {}
    }

    // Joseph 形式协方差更新
    Eigen::Matrix<double, 15, 15> I =
        Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix<double, 15, 15> I_KH = I - K_pos * H_pos;
    P = I_KH * P * I_KH.transpose() + K_pos * R_pos_eff * K_pos.transpose();
    if (got_G_reset)
    {
        try { P = G_reset * P * G_reset.transpose(); } catch (...) {}
    }
    P = 0.5 * (P + P.transpose());
    P += 1e-9 * Eigen::Matrix<double, 15, 15>::Identity();

    // ---- 2. 姿态 3-DOF 追加更新（rotNIS 自洽才做）----
    // 位置注入经 P 耦合改变了 q，姿态残差必须基于更新后 q（py 先 _unpack 再算 dq）
    std::tie(p, v, q, b_a, b_g) = _unpack();
    Eigen::Matrix<double, 4, 1> dq = quat_mul(quat_inv(q), *q_meas_opt);
    if (dq(0) < 0.0) dq *= -1.0;  // 短路径检查

    // 四元数残差 -> 3D 旋转向量
    double qw    = clamp(dq(0), -1.0, 1.0);
    double angle = 2.0 * std::acos(qw);
    Eigen::Vector3d z_rot;
    if (angle < 1e-6)
    {
        z_rot = 2.0 * dq.segment<3>(1);
    }
    else
    {
        double sin_half = std::sqrt(std::max(0.0, 1.0 - qw * qw));
        if (sin_half < 1e-8)
        {
            z_rot = 2.0 * dq.segment<3>(1);
        }
        else
        {
            Eigen::Vector3d axis = dq.segment<3>(1) / sin_half;
            z_rot = angle * axis;
        }
    }

    Eigen::Matrix<double, 3, 15> H_rot = Eigen::Matrix<double, 3, 15>::Zero();
    H_rot.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_rot =
        Eigen::Matrix3d::Identity()
        * (params.cam_rot_noise * params.cam_rot_noise);

    Eigen::Matrix3d S_rot = H_rot * P * H_rot.transpose() + R_rot;
    Eigen::Matrix3d inv_S_rot;
    if (S_rot.determinant() != 0.0)
        inv_S_rot = S_rot.inverse();
    else
        inv_S_rot = detail::svd_pinv_3x3(S_rot);
    double nis_rot = z_rot.transpose() * inv_S_rot * z_rot;

    // 姿态门限：nis_rot × rot_detect_scale >= χ²₉₅ 则跳过追加（位置已更新）。
    // rot_detect_scale=4.0 → 有效门限 7.8147/4 = 1.954（±2σ 内自洽才追加）
    if (nis_rot * cam_fdi_rot_detect_scale_ >= chi2_pos_95)
    {
        double nis_total = std::isfinite(nis_pos) ? nis_pos : 0.0;
        return {nis_total, std::nullopt, std::nullopt, "rot_skipped"};
    }

    Eigen::Matrix<double, 3, 3> S_rot_k =
        H_rot * P * H_rot.transpose() + R_rot;
    Eigen::Matrix<double, 15, 3> K_rot;
    if (S_rot_k.determinant() != 0.0)
        K_rot = P * H_rot.transpose() * S_rot_k.inverse();
    else
        K_rot = P * H_rot.transpose() * detail::svd_pinv_3x3(S_rot_k);
    Eigen::Matrix<double, 15, 1> dx_rot = K_rot * z_rot;

    G_reset = Eigen::Matrix<double, 15, 15>::Identity();
    got_G_reset = false;
    try
    {
        G_reset = _inject(dx_rot);
        got_G_reset = true;
    }
    catch (...)
    {
        try { _inject(dx_rot); } catch (...) {}
    }

    I_KH = I - K_rot * H_rot;
    P = I_KH * P * I_KH.transpose() + K_rot * R_rot * K_rot.transpose();
    if (got_G_reset)
    {
        try { P = G_reset * P * G_reset.transpose(); } catch (...) {}
    }
    P = 0.5 * (P + P.transpose());
    P += 1e-9 * Eigen::Matrix<double, 15, 15>::Identity();

    Eigen::Matrix<double, 6, 1> z6d;
    z6d.segment<3>(0) = z_pos;
    z6d.segment<3>(3) = z_rot;

    double nis_total = (std::isfinite(nis_pos) ? nis_pos : 0.0)
                     + (std::isfinite(nis_rot) ? nis_rot : 0.0);
    return {nis_total, z6d, dq, "hybrid"};
}

// ---- Camera 位姿更新（卡方检验异常值拒绝版）----
inline
std::pair<bool, bool>
ESKF_VIO::update_camera_pose_chi2(const Eigen::Vector3d &p_meas,
                                  const Eigen::Matrix<double, 4, 1> &q_meas)
{
    if (!chi2_reject_enabled_)
    {
        // 回退到基础版
        auto [nis, z_opt, dq_opt] = update_camera_pose(p_meas, q_meas);
        bool accepted = z_opt.has_value();
        return {accepted, accepted};
    }

    auto [p, v, q, b_a, b_g] = _unpack();
    (void)v; (void)b_a; (void)b_g;

    // 杆臂补偿
    Eigen::Matrix3d Rcam = quat_to_rot(q);
    Eigen::Vector3d cam_expected = p - Rcam * params.p_imu_in_cam;
    Eigen::Vector3d dp = p_meas - cam_expected;
    // 右误差约定（body 系）：dq = q_nom⁻¹ ⊗ q_meas
    Eigen::Matrix<double, 4, 1> dq_v = quat_mul(quat_inv(q), q_meas);
    if (dq_v(0) < 0.0) dq_v *= -1.0;  // 短路径检查
    Eigen::Vector3d dtheta = 2.0 * dq_v.segment<3>(1);

    Eigen::Matrix<double, 6, 1> z;
    z.segment<3>(0) = dp;
    z.segment<3>(3) = dtheta;

    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

    // 构造 R（支持 Z 轴解耦）
    Eigen::Matrix<double, 6, 6> R_mat =
        Eigen::Matrix<double, 6, 6>::Zero();
    if (params.use_cam_z_noise_decoupling)
    {
        R_mat(0, 0) = params.cam_pos_noise * params.cam_pos_noise;
        R_mat(1, 1) = params.cam_pos_noise * params.cam_pos_noise;
        R_mat(2, 2) = params.cam_pos_z_noise * params.cam_pos_z_noise;
    }
    else
    {
        double pn2 = params.cam_pos_noise * params.cam_pos_noise;
        for (int i = 0; i < 3; ++i) R_mat(i, i) = pn2;
    }
    double rn2 = params.cam_rot_noise * params.cam_rot_noise;
    for (int i = 0; i < 3; ++i) R_mat(i + 3, i + 3) = rn2;

    Eigen::Matrix<double, 6, 6> S = H * P * H.transpose() + R_mat;
    S += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();

    // 马氏距离
    double chi2 = z.transpose() * S.inverse() * z;
    last_chi2_ = chi2;
    total_updates_++;

    if (chi2 > chi2_threshold_95_)
    {
        rejected_updates_++;
        P += 0.1 * Eigen::Matrix<double, 15, 15>::Identity()
             * (chi2 - chi2_threshold_95_) / chi2_threshold_95_;
        return {false, false};
    }

    // 正常更新
    Eigen::Matrix<double, 15, 6> K = P * H.transpose() * S.inverse();
    Eigen::Matrix<double, 15, 1> dx = K * z;

    Eigen::Matrix<double, 15, 15> I_mat =
        Eigen::Matrix<double, 15, 15>::Identity();
    P = (I_mat - K * H) * P * (I_mat - K * H).transpose()
        + K * R_mat * K.transpose();

    _inject(dx);
    return {true, true};
}

// ---- Camera 位姿更新（自适应噪声 + 卡方拒绝版）----
inline
std::pair<bool, bool>
ESKF_VIO::update_camera_pose_adaptive(
    const Eigen::Vector3d &p_meas,
    const Eigen::Matrix<double, 4, 1> &q_meas)
{
    auto [p, v, q, b_a, b_g] = _unpack();
    (void)v; (void)b_a; (void)b_g;

    // 杆臂补偿
    Eigen::Matrix3d Rcam = quat_to_rot(q);
    Eigen::Vector3d cam_expected = p - Rcam * params.p_imu_in_cam;
    Eigen::Vector3d dp = p_meas - cam_expected;
    // 右误差约定（body 系）：dq = q_nom⁻¹ ⊗ q_meas
    Eigen::Matrix<double, 4, 1> dq_v =
        quat_mul(quat_inv(q), q_meas);
    if (dq_v(0) < 0.0) dq_v *= -1.0;  // 短路径检查
    Eigen::Vector3d dtheta = 2.0 * dq_v.segment<3>(1);

    Eigen::Matrix<double, 6, 1> z;
    z.segment<3>(0) = dp;
    z.segment<3>(3) = dtheta;

    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

    // 基于残差自适应调整观测噪声
    double pos_residual_norm = dp.norm();
    double rot_residual_norm = dtheta.norm();

    double pos_noise_adaptive = params.cam_pos_noise
        * std::max(1.0, pos_residual_norm
                         / (params.cam_pos_noise * 3.0));
    double rot_noise_adaptive = params.cam_rot_noise
        * std::max(1.0, rot_residual_norm
                         / (params.cam_rot_noise * 3.0));
    pos_noise_adaptive = std::min(pos_noise_adaptive,
                                  params.cam_pos_noise * 5.0);
    rot_noise_adaptive = std::min(rot_noise_adaptive,
                                  params.cam_rot_noise * 5.0);

    Eigen::Matrix<double, 6, 6> R_mat =
        Eigen::Matrix<double, 6, 6>::Zero();
    // 支持 Z 轴解耦
    if (params.use_cam_z_noise_decoupling)
    {
        R_mat(0, 0) = params.cam_pos_noise * params.cam_pos_noise;
        R_mat(1, 1) = params.cam_pos_noise * params.cam_pos_noise;
        R_mat(2, 2) = params.cam_pos_z_noise * params.cam_pos_z_noise;
    }
    else
    {
        double pn2 = pos_noise_adaptive * pos_noise_adaptive;
        for (int i = 0; i < 3; ++i) R_mat(i, i) = pn2;
    }
    double rn2 = rot_noise_adaptive * rot_noise_adaptive;
    for (int i = 0; i < 3; ++i) R_mat(i + 3, i + 3) = rn2;

    Eigen::Matrix<double, 6, 6> S = H * P * H.transpose() + R_mat;
    S += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix<double, 6, 6> S_inv = S.inverse();
    double chi2 = z.transpose() * S_inv * z;
    last_chi2_ = chi2;
    total_updates_++;

    if (chi2_reject_enabled_ && chi2 > chi2_threshold_95_)
    {
        rejected_updates_++;
        P += 0.1 * Eigen::Matrix<double, 15, 15>::Identity()
             * (chi2 - chi2_threshold_95_) / chi2_threshold_95_;
        return {false, false};
    }

    Eigen::Matrix<double, 15, 6> K = P * H.transpose() * S_inv;
    Eigen::Matrix<double, 15, 1> dx = K * z;

    Eigen::Matrix<double, 15, 15> I_mat =
        Eigen::Matrix<double, 15, 15>::Identity();
    P = (I_mat - K * H) * P * (I_mat - K * H).transpose()
        + K * R_mat * K.transpose();

    _inject(dx);
    return {true, true};
}

// ---- Radar 高度更新（基础版）----
inline
std::pair<double, std::optional<double>>
ESKF_VIO::update_radar_height(double radar_height)
{
    auto [p, v, q, b_a, b_g] = _unpack();
    (void)v; (void)q; (void)b_a; (void)b_g;

    double residual = radar_height - p(2);

    Eigen::Matrix<double, 1, 15> H = Eigen::Matrix<double, 1, 15>::Zero();
    H(0, 2) = 1.0;

    double R_val = params.radar_alt_noise
                   * params.radar_alt_noise;

    Eigen::Matrix<double, 15, 1> PHt = P * H.transpose();
    double S = H(0, 2) * PHt(2) + R_val;

    if (S <= 0.0 || !std::isfinite(S))
    {
        return {-1.0, std::nullopt};
    }

    double nis = residual * residual / S;

    const double chi2_95 = 3.841;
    if (nis > chi2_95)
    {
        return {nis, std::nullopt};
    }

    double S_eff = H(0, 2) * PHt(2) + R_val;
    if (S_eff <= 0.0 || !std::isfinite(S_eff))
    {
        return {nis, residual};
    }

    Eigen::Matrix<double, 15, 1> K = PHt / S_eff;
    Eigen::Matrix<double, 15, 1> dx = K * residual;

    Eigen::Matrix<double, 15, 15> G_reset = _inject(dx);

    Eigen::Matrix<double, 15, 15> I =
        Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix<double, 15, 15> I_KH = I - K * H;
    P = I_KH * P * I_KH.transpose() + K * K.transpose() * R_val;
    P = G_reset * P * G_reset.transpose();
    P = 0.5 * (P + P.transpose());

    return {nis, residual};
}

// ---- Radar 高度更新（完整版，对标 Python update_altitude）----
inline
std::tuple<double, double, double>
ESKF_VIO::update_altitude(double z_meas,
                          std::optional<double> noise_std_override)
{
    auto [p, v, q, b_a, b_g] = _unpack();
    (void)v; (void)q; (void)b_a; (void)b_g;

    // 2026-08-12：高度沿估计重力方向投影（W' 系倾斜时 z 分量 ≠ 垂直距离）
    // n_hat = -ĝ/|ĝ|（垂直向上），与 fusion_thread NIS 校验同约定（h = p·n_hat，
    // H_z 行 = n_hatᵀ）；无估计（norm<0.1）时回退 [0,0,1] = 原行为，零回归风险
    Eigen::Vector3d n_hat = (params.gravity.norm() < 0.1)
                                ? Eigen::Vector3d(0.0, 0.0, 1.0)
                                : (-params.gravity.normalized());

    double z_pred = p.dot(n_hat);

    // 观测矩阵 H_z: 1×15（位置行 = n_hatᵀ）
    Eigen::Matrix<double, 1, 15> H_z =
        Eigen::Matrix<double, 1, 15>::Zero();
    H_z.segment<3>(0) = n_hat.transpose();

    double noise_std = noise_std_override.has_value()
                         ? noise_std_override.value()
                         : params.radar_alt_noise;
    double R_z = noise_std * noise_std;

    Eigen::Matrix<double, 15, 1> PHt = P * H_z.transpose();
    double S = n_hat.dot(P.block<3, 3>(0, 0) * n_hat) + R_z;  // H_z·P·H_zᵀ
    S += 1e-12;

    double innovation = z_meas - z_pred;
    double nis = innovation * innovation / S;

    // 标量卡尔曼增益
    Eigen::Matrix<double, 15, 1> K_z = PHt / S;

    double dx_scalar = K_z.head<3>().dot(n_hat) * innovation;  // 仅用于验证
    (void)dx_scalar;

    Eigen::Matrix<double, 15, 1> dx = K_z * innovation;

    // Joseph 形式协方差更新
    Eigen::Matrix<double, 15, 15> I_mat =
        Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix<double, 15, 15> P_old = P;
    Eigen::Matrix<double, 15, 15> I_KH = I_mat - K_z * H_z;
    P = I_KH * P_old * I_KH.transpose()
        + K_z * K_z.transpose() * R_z;
    P = 0.5 * (P + P.transpose());
    P += 1e-12 * Eigen::Matrix<double, 15, 15>::Identity();

    _inject(dx);

    return {nis, innovation, S};
}

// ---- V17 参数访问器 ----
inline
bool ESKF_VIO::adaptive_noise_enabled() const
{
    return adaptive_noise_enabled_;
}

inline
void ESKF_VIO::set_adaptive_noise_enabled(bool enabled)
{
    adaptive_noise_enabled_ = enabled;
}

inline
double ESKF_VIO::adaptive_noise_alpha() const
{
    return adaptive_noise_alpha_;
}

inline
void ESKF_VIO::set_adaptive_noise_alpha(double alpha)
{
    adaptive_noise_alpha_ = clamp(alpha, 0.0, 1.0);
}

inline
int ESKF_VIO::adaptive_noise_window() const
{
    return adaptive_noise_window_;
}

inline
void ESKF_VIO::set_adaptive_noise_window(int window_size)
{
    adaptive_noise_window_ = std::max(10, window_size);
}

inline
bool ESKF_VIO::chi2_reject_enabled() const
{
    return chi2_reject_enabled_;
}

inline
void ESKF_VIO::set_chi2_reject_enabled(bool enabled)
{
    chi2_reject_enabled_ = enabled;
}

inline
double ESKF_VIO::chi2_threshold_95() const
{
    return chi2_threshold_95_;
}

inline
void ESKF_VIO::set_chi2_threshold_95(double threshold)
{
    chi2_threshold_95_ = threshold;
}

inline
int ESKF_VIO::total_updates() const
{
    return total_updates_;
}

inline
int ESKF_VIO::rejected_updates() const
{
    return rejected_updates_;
}

inline
double ESKF_VIO::last_chi2() const
{
    return last_chi2_;
}

// ============================================================
// 内部方法实现
// ============================================================

inline
bool ESKF_VIO::_predict_nominal(
    const std::optional<Eigen::Vector3d> &acc_opt,
    const std::optional<Eigen::Vector3d> &gyro_opt,
    double dt)
{
    // IMU 数据无效时仅做简单传播
    if (!acc_opt.has_value() || !gyro_opt.has_value())
    {
        auto [p, v, q, b_a, b_g] = _unpack();
        p = p + v * dt;
        b_a *= std::exp(-params.p_acc * dt);
        b_g *= std::exp(-params.p_gyro * dt);
        _pack(p, v, q, b_a, b_g);

        Eigen::Matrix<double, 15, 15> F =
            Eigen::Matrix<double, 15, 15>::Identity();
        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        P = F * P * F.transpose() + Q;
        return false;
    }

    Eigen::Vector3d acc  = *acc_opt;
    Eigen::Vector3d gyro = *gyro_opt;

    auto [p, v, q, b_a, b_g] = _unpack();

    Eigen::Vector3d acc_u  = acc  - b_a;
    Eigen::Vector3d gyro_u = gyro - b_g;

    // IMU → Camera 外参变换：加计与陀螺一致地转到相机系
    // （姿态状态 q 是相机系→世界系，两路旋转必须用同一外参，否则姿态传播错误）
    Eigen::Vector3d acc_cam  = params.R_imu_cam * acc_u;
    Eigen::Vector3d gyro_cam = params.R_imu_cam * gyro_u;

    Eigen::Matrix3d Rmat = quat_to_rot(q);
    // 使用在线估计的重力
    Eigen::Vector3d a_world = Rmat * acc_cam + params.gravity;

    // 名义态传播
    p = p + v * dt + 0.5 * a_world * dt * dt;
    v = v + a_world * dt;

    Eigen::Matrix<double, 4, 1> dq = quat_exp(gyro_cam * dt);
    q = quat_mul(q, dq);
    q.normalize();

    b_a *= std::exp(-params.p_acc * dt);
    b_g *= std::exp(-params.p_gyro * dt);

    _pack(p, v, q, b_a, b_g);

    // ---- 误差动力学矩阵 A, G ----
    Eigen::Matrix<double, 15, 15> A =
        Eigen::Matrix<double, 15, 15>::Zero();
    Eigen::Matrix<double, 15, 12> G =
        Eigen::Matrix<double, 15, 12>::Zero();

    Rmat = quat_to_rot(q);

    A.block<3, 3>(0, 3)  = Eigen::Matrix3d::Identity();
    A.block<3, 3>(3, 6)  = -Rmat * skew(acc_cam);
    A.block<3, 3>(3, 9)  = -Rmat * params.R_imu_cam;
    A.block<3, 3>(6, 6)  = -skew(gyro_cam);
    A.block<3, 3>(6, 12) = -params.R_imu_cam;   // ∂ω_cam/∂b_g = −R
    A.block<3, 3>(9, 9)  = -params.p_acc
                            * Eigen::Matrix3d::Identity();
    A.block<3, 3>(12, 12) = -params.p_gyro
                             * Eigen::Matrix3d::Identity();

    G.block<3, 3>(3, 0)  = Rmat;
    G.block<3, 3>(6, 3)  = Eigen::Matrix3d::Identity();
    G.block<3, 3>(9, 6)  = Eigen::Matrix3d::Identity();
    G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

    // 连续时间噪声协方差
    Eigen::Matrix<double, 12, 12> Qc =
        Eigen::Matrix<double, 12, 12>::Zero();
    for (int i = 0; i < 3; ++i)
        Qc(i, i) = params.sigma_acc * params.sigma_acc;
    for (int i = 0; i < 3; ++i)
        Qc(i + 3, i + 3) = params.sigma_gyro * params.sigma_gyro;
    for (int i = 0; i < 3; ++i)
        Qc(i + 6, i + 6) = params.sigma_acc_bias
                           * params.sigma_acc_bias;
    for (int i = 0; i < 3; ++i)
        Qc(i + 9, i + 9) = params.sigma_gyro_bias
                           * params.sigma_gyro_bias;

    // 使用前向 Euler 离散化 (与 Python 一致):
    //   F_d = I + A·dt
    //   Q_d = G·Qc·G^T·dt + pos_rw
    Eigen::Matrix<double, 15, 15> F_d =
        Eigen::Matrix<double, 15, 15>::Identity() + A * dt;

    Eigen::Matrix<double, 15, 15> Q_d =
        G * Qc * G.transpose() * dt;

    // 位置随机游走：补偿前向 Euler 离散化误差
    Q_d.block<3, 3>(0, 0) += Eigen::Matrix3d::Identity()
                              * (params.sigma_pos_rw
                                 * params.sigma_pos_rw) * dt;

    P = F_d * P * F_d.transpose() + Q_d;
    P = 0.5 * (P + P.transpose());
    return true;
}

inline
Eigen::Matrix<double, 15, 15>
ESKF_VIO::_inject(const Eigen::Matrix<double, 15, 1> &dx)
{
    Eigen::Vector3d dp    = dx.segment<3>(0);
    Eigen::Vector3d dv    = dx.segment<3>(3);
    Eigen::Vector3d dtheta = dx.segment<3>(6);
    Eigen::Vector3d dba   = dx.segment<3>(9);
    Eigen::Vector3d dbg   = dx.segment<3>(12);

    auto [p, v, q, b_a, b_g] = _unpack();

    p    += dp;
    v    += dv;
    q     = quat_mul(q, quat_exp(dtheta));
    q    /= q.norm();
    b_a  += dba;
    b_g  += dbg;

    _pack(p, v, q, b_a, b_g);

    // Reset 雅可比
    Eigen::Matrix<double, 15, 15> G_reset =
        Eigen::Matrix<double, 15, 15>::Identity();
    G_reset.block<3, 3>(6, 6) -= 0.5 * skew(dtheta);
    return G_reset;
}

inline
std::tuple<Eigen::Vector3d, Eigen::Vector3d,
           Eigen::Matrix<double, 4, 1>,
           Eigen::Vector3d, Eigen::Vector3d>
ESKF_VIO::_unpack() const
{
    Eigen::Vector3d p   = x.segment<3>(0);
    Eigen::Vector3d v   = x.segment<3>(3);
    Eigen::Matrix<double, 4, 1> q = x.segment<4>(6);
    Eigen::Vector3d b_a = x.segment<3>(10);
    Eigen::Vector3d b_g = x.segment<3>(13);
    return {p, v, q, b_a, b_g};
}

inline
void ESKF_VIO::_pack(const Eigen::Vector3d &p,
                     const Eigen::Vector3d &v,
                     const Eigen::Matrix<double, 4, 1> &q,
                     const Eigen::Vector3d &b_a,
                     const Eigen::Vector3d &b_g)
{
    x.segment<3>(0)  = p;
    x.segment<3>(3)  = v;
    x.segment<4>(6)  = q;
    x.segment<3>(10) = b_a;
    x.segment<3>(13) = b_g;
}

} // namespace eskf

#endif // ESKF_VIO_HPP
