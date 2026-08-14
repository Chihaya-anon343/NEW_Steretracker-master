#pragma once

/**
 * @file FusionTypes.hpp
 * @brief 多源融合样本/观测类型 (ESKF 适配层输入数据类型)
 *
 * Module: fusion
 * Function: 定义 IMU / 雷达 / 相机三种传感器的带时间戳样本与观测结构体,
 *           作为 EskfFusionManager 输入接口的数据载体。
 *
 * 时间戳约定:
 *   - 所有时间戳为 double 秒 (与 EskfFusionManager 输入接口一致)
 *   - CameraObservation 显式区分 t_exposure (t0 曝光时刻) 与
 *     t_arrival (t1 送达时刻), 两者之差即"延迟测量" (方案B 反向传播输入);
 *     无延迟语义时 (旧接口) t_exposure == t_arrival
 *
 * 坐标/单位约定:
 *   - ImuSample.acc     : SI 单位 (m/s²), IMU 本体坐标系 (比力, 含重力补偿前提)
 *   - ImuSample.gyro    : SI 单位 (rad/s), IMU 本体坐标系
 *   - RadarSample.height: 米 (m), 世界系 Z 轴高度
 *   - CameraObservation : R_tpl_cam = 模板→相机旋转, t_cam_mm 单位 mm
 *     (与 PnP 解算输出一致, EskfFusionManager 内部 ÷1000 转 m)
 *
 * Dependencies: Eigen
 */

#include <Eigen/Dense>

namespace gpnp {
namespace fusion {

/// IMU 样本 (加速度 + 角速度, SI 单位, IMU 本体坐标系)
struct ImuSample {
    double t = 0.0;                       ///< 时间戳 (s)
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();   ///< 加速度 (m/s², 比力)
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  ///< 角速度 (rad/s)
};

/// 雷达高度样本 (m, 世界系 Z 轴方向)
struct RadarSample {
    double t = 0.0;                       ///< 时间戳 (s)
    double height = 0.0;                  ///< 高度 (m)
};

/// 相机位姿观测 (PnP 输出, 含曝光/送达双时间戳 → 延迟测量)
struct CameraObservation {
    double t_exposure = 0.0;              ///< t0 曝光时刻 (s)
    double t_arrival  = 0.0;              ///< t1 送达时刻 (s), t1 - t0 = 延迟
    Eigen::Matrix3d R_tpl_cam = Eigen::Matrix3d::Identity();  ///< 模板→相机旋转
    Eigen::Vector3d t_cam_mm = Eigen::Vector3d::Zero();       ///< 平移 (mm)
    bool valid = false;                   ///< 该帧视觉是否成功 (false → 仅惯性传播)
};

} // namespace fusion
} // namespace gpnp
