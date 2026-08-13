// =============================================================
// 示例: eskf_vio 单头文件库 — 完整融合流程演示
// 场景: 模拟一个垂直运动的载体（世界系 z 轴向上）
//   - 前 4s: 恒加速下落 (a_z = -1 m/s²)
//   - 4~6s: 减速 (a_z = +0.5 m/s²)
//   - 之后: 悬停
// 三种观测:
//   - IMU: 100 Hz 加计/陀螺（含噪声）
//   - 相机: 10 Hz 位姿（含噪声；周期注入姿态翻转帧演示 FDI）
//   - 雷达: 20 Hz 高度（含噪声；周期注入跳变帧演示检验器）
// 编译运行:
//   g++ -std=c++17 -I<eigen3路径> -I.. examples/demo.cpp -o demo && ./demo
// =============================================================

#include "eskf_vio.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#ifdef _WIN32
#define NOMINMAX   // 禁止 windows.h 定义 min/max 宏，避免与 std::min/std::max 冲突
#include <windows.h>
#endif

using namespace eskf;

// ---- 真值运动: 返回世界系加速度 a_world (m/s²) ----
static double truth_accel_z(double t)
{
    if (t < 4.0) return -1.0;
    if (t < 6.0) return +0.5;
    return 0.0;
}

// 数值积分真值轨迹 (从 t=0, p=(0,0,10), v=0 起步)
static void truth_trajectory(double t, Eigen::Vector3d &p, Eigen::Vector3d &v)
{
    p.setZero(); v.setZero();
    p(2) = 10.0;
    const double dt = 1e-4;
    for (double tt = 0.0; tt < t; tt += dt)
    {
        double h = std::min(dt, t - tt);
        double az = truth_accel_z(tt);
        v(2) += az * h;
        p(2) += v(2) * h;
    }
}

// 标准正态随机数
static std::mt19937 rng(42);
static double gauss(double sigma)
{
    std::normal_distribution<double> n(0.0, 1.0);
    return n(rng) * sigma;
}

int main()
{
#ifdef _WIN32
    // Windows 控制台默认代码页(437/936)不是 UTF-8，强制切换后中文输出才能正常显示
    SetConsoleOutputCP(CP_UTF8);
#endif

    // ---- 初始化 ESKF ----
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(16);
    x0.segment<3>(0) = Eigen::Vector3d(0.0, 0.0, 10.0);   // 位置 (真值起点)
    x0(6) = 1.0;                                          // 姿态: 单位四元数

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(15, 15) * 0.1;

    ESKFParams params;
    // 无 Z 轴解耦（雷达直接约束 z，简化场景）
    params.use_cam_z_noise_decoupling = false;
    params.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);

    ESKF_VIO eskf(x0, P0, params);

    // ---- 在线重力估计器 & 雷达检验器 ----
    GravityEstimator ge(60);                 // 60 帧滑窗
    RadarAltimeter radar(0.30, 7.879, 30.0, 20, 20, 10);

    // ---- 时间参数 ----
    const double dt_imu  = 0.01;             // IMU 100 Hz
    const double dt_cam  = 0.10;             // 相机 10 Hz
    const double dt_rad  = 0.05;             // 雷达 20 Hz
    const double T_END   = 10.0;             // 仿真 10 s

    double t_cam_last = -1e9, t_rad_last = -1e9;
    std::vector<std::pair<double, Eigen::Vector3d>> imu_steps;  // (dt, 比力世界系)

    // 统计
    int n_hybrid = 0, n_rot_skipped = 0, n_ignore = 0, n_radar_ok = 0, n_radar_rej = 0;

    for (double t = 0; t < T_END; t += dt_imu)
    {
        // ---- 真值 ----
        Eigen::Vector3d p_true, v_true;
        truth_trajectory(t, p_true, v_true);
        double az = truth_accel_z(t);

        // ---- IMU 测量 (相机系) ----
        // 比力(世界系) s = a_world - g；姿态单位阵 → 相机系 = 世界系
        Eigen::Vector3d s_world(0.0, 0.0, az + 9.81);
        Eigen::Vector3d acc_meas = s_world + Eigen::Vector3d(gauss(0.1), gauss(0.1), gauss(0.1));
        Eigen::Vector3d gyro_meas(gauss(0.005), gauss(0.005), gauss(0.005));

        // 记录给重力估计器的 IMU 步 (比力, 世界系)
        imu_steps.emplace_back(dt_imu, s_world);

        // ---- 预测 ----
        eskf.predict_adaptive(acc_meas, gyro_meas, dt_imu);

        // ---- 相机观测 (10 Hz) ----
        if (t - t_cam_last >= dt_cam - 1e-9)
        {
            t_cam_last = t;

            Eigen::Vector3d p_cam = p_true + Eigen::Vector3d(gauss(0.05), gauss(0.05), gauss(0.05));
            Eigen::Matrix<double, 4, 1> q_cam;
            q_cam << 1.0, 0.0, 0.0, 0.0;

            // 每隔 ~1s 注入一帧姿态翻转 (演示 hybrid 姿态门限: 位置仍更新)
            bool inject_flip = (std::abs(t - 2.0) < 0.01) || (std::abs(t - 5.0) < 0.01);
            if (inject_flip) q_cam(1) = 1.0;   // 180° 翻转

            auto [nis, z_opt, dq_opt, action] =
                eskf.update_camera_pose_hybrid(p_cam, q_cam);
            (void)nis; (void)z_opt; (void)dq_opt;
            if      (action == "hybrid")       n_hybrid++;
            else if (action == "rot_skipped")  n_rot_skipped++;
            else if (action == "ignore")       n_ignore++;

            // 重力估计: 滑窗样本 = 相机位姿 + 帧间 IMU 比力步
            ge.add_sample(t, p_cam, imu_steps);
            imu_steps.clear();
            auto [g_opt, info] = ge.update(1e-3, 50, 5, true, t);
            (void)info;
            if (g_opt.has_value())
                eskf.params.gravity = *g_opt;  // 写回滤波器 (在线重力)
        }

        // ---- 雷达观测 (20 Hz) ----
        if (t - t_rad_last >= dt_rad - 1e-9)
        {
            t_rad_last = t;

            // 每隔 ~1.5s 注入一次跳变 (演示检验器拒绝)
            double z_meas = p_true(2) + gauss(0.30);
            bool inject_jump = (std::abs(t - 1.0) < 0.01) || (std::abs(t - 3.5) < 0.01);
            if (inject_jump) z_meas += 50.0;

            // 预测高度 & 预测协方差 (同 fusion_thread 用法)
            Eigen::Vector3d n_hat = (eskf.params.gravity.norm() < 0.1)
                                        ? Eigen::Vector3d(0, 0, 1)
                                        : (-eskf.params.gravity.normalized());
            double z_pred  = eskf.x.segment<3>(0).dot(n_hat);
            double P_z_pred = n_hat.dot(eskf.P.block<3, 3>(0, 0) * n_hat);

            auto [ok, nis, info] = radar.validate(z_meas, z_pred, P_z_pred, t);
            (void)nis;
            if (ok && !info.radar_failed)
            {
                n_radar_ok++;
                eskf.update_altitude(z_meas, info.effective_noise_std);
            }
            else
            {
                n_radar_rej++;
            }
        }

        // ---- 周期打印 ----
        if (std::abs(t - std::round(t)) < dt_imu / 2)
        {
            Eigen::Vector3d p_est = eskf.x.segment<3>(0);
            Eigen::Vector3d v_est = eskf.x.segment<3>(3);
            std::printf(
                "t=%5.1fs  p_est=(%7.3f,%7.3f,%7.3f)  p_true=(%7.3f,%7.3f,%7.3f)  "
                "v_z=%6.3f  g_est=(%6.3f,%6.3f,%6.3f)  converged=%d\n",
                t, p_est(0), p_est(1), p_est(2), p_true(0), p_true(1), p_true(2),
                v_est(2),
                ge.get_gravity()(0), ge.get_gravity()(1), ge.get_gravity()(2),
                ge.is_converged(2.0, 0.5) ? 1 : 0);
        }
    }

    // ---- 汇总 ----
    Eigen::Vector3d p_true_f, v_true_f;
    truth_trajectory(T_END, p_true_f, v_true_f);
    Eigen::Vector3d p_est_f = eskf.x.segment<3>(0);

    std::printf("\n==== 汇总 ====\n");
    std::printf("相机更新: hybrid=%d  rot_skipped=%d  ignore=%d\n",
                n_hybrid, n_rot_skipped, n_ignore);
    std::printf("雷达: 接受=%d  拒绝=%d  (注入 2 个跳变应被拒)\n",
                n_radar_ok, n_radar_rej);
    std::printf("终端位置误差: %.3f m (期望 < 0.5 m)\n", (p_est_f - p_true_f).norm());
    std::printf("重力估计: (%6.3f, %6.3f, %6.3f)  vs 真值 (0,0,-9.81)\n",
                ge.get_gravity()(0), ge.get_gravity()(1), ge.get_gravity()(2));

    auto rs = radar.get_statistics();
    std::printf("雷达统计: total=%d accepted=%d (%.1f%%) reject_jump=%d reject_nis=%d\n",
                rs.total_calls, rs.accepted, rs.accept_rate * 100.0,
                rs.reject_jump, rs.reject_nis);

    bool ok = (p_est_f - p_true_f).norm() < 0.5
              && (ge.get_gravity() - Eigen::Vector3d(0, 0, -9.81)).norm() < 0.05
              && rs.reject_jump >= 2;
    std::printf("\n%s\n", ok ? "[PASS] 融合收敛, 重力估计正确, 雷达拒跳变正常"
                             : "[FAIL] 见上方输出排查");
    return ok ? 0 : 1;
}
