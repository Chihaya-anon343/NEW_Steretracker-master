#pragma once

/**
 * @file SensorTypes.hpp
 * @brief 多传感器输入系统的扩展数据类型。
 *
 * 补充 Types.hpp 中已有的双目视觉数据结构，
 * 添加 IMU、高度计、统一传感器数据包等类型。
 */

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>

namespace gpnp {
namespace input {

// ============================================================================
// IMU 数据
// ============================================================================

/// 单帧 IMU 测量值（加速度 + 角速度 + 姿态角）。
struct ImuData {
    double t = 0.0;               ///< 时间戳（秒）
    Eigen::Vector3d acc;          ///< 加速度 (m/s²)
    Eigen::Vector3d gyro;         ///< 角速度 (rad/s)
    Eigen::Vector3d angle;        ///< 姿态角 (度)
};

// ============================================================================
// 高度计数据
// ============================================================================

/// 单帧高度计/雷达测量值。
struct AltimeterData {
    double t = 0.0;               ///< 时间戳（秒）
    double height = 0.0;          ///< 高度 (米)
};

// ============================================================================
// 统一传感器数据包
// ============================================================================

/// 以相机帧为基准对齐后的统一传感器数据包。
/// 由 InputProvider::getNextPacket() 组装。
struct SensorPacket {
    int64_t timestamp_us = 0;     ///< 统一时间戳（微秒），以相机帧时间为基准

    cv::Mat left_image;           ///< 左相机图像
    cv::Mat right_image;          ///< 右相机图像

    std::optional<ImuData> imu;        ///< 对齐后的 IMU 数据（若启用）
    std::optional<double> height;      ///< 对齐后的高度数据（若启用），单位 米

    bool valid = false;           ///< 数据包是否有效
};

} // namespace input
} // namespace gpnp
