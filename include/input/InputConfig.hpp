#pragma once

/**
 * @file InputConfig.hpp
 * @brief 输入系统的配置结构体。
 *
 * 从 tracker_config.json 的 "input_system" 节反序列化。
 */

#include <string>

namespace gpnp {
namespace input {

// ============================================================================
// 图像源类型枚举
// ============================================================================

enum class ImageSourceType {
    File,       ///< 静态文件对（当前行为）
    Directory,  ///< 目录中的双目编号图像序列 (left_*/right_*)
    Sequence,   ///< 目录中的单目图像序列 (frame_0000.jpg, frame_0001.jpg...)
    Camera      ///< 实时摄像头（未来）
};

// ============================================================================
// 图像输入配置
// ============================================================================

struct ImageInputConfig {
    ImageSourceType type = ImageSourceType::File;

    /// type == File: 左/右图像路径
    std::string left_path;
    std::string right_path;

    /// type == Directory: 包含 left_*.png / right_*.png 的目录
    std::string directory_path;

    /// type == Directory: 文件名模式（默认 "left" / "right"）
    std::string left_pattern  = "left";
    std::string right_pattern = "right";

    /// type == Sequence: 文件名前缀（如 "frame" 匹配 frame_0000.jpg）
    std::string sequence_pattern = "frame";

    /// type == Camera: 设备路径（如 "/dev/video0;video1"）
    std::string camera_devices;

    /// 帧率限制（Camera 模式），0 表示不限制
    double target_fps = 0.0;
};

// ============================================================================
// IMU 输入配置
// ============================================================================

struct ImuInputConfig {
    bool enabled = false;

    /// 串口设备路径，如 "/dev/ttyUSB0"
    std::string port;

    /// 波特率，如 921600
    int baud_rate = 921600;

    /// 协议类型: "custom" = 自定义二进制协议（需在 SerialImuSource 中实现）
    std::string protocol = "custom";
};

// ============================================================================
// 高度计输入配置
// ============================================================================

struct AltimeterInputConfig {
    bool enabled = false;

    /// CAN 接口名称，如 "can0"
    std::string can_interface;

    /// 高度计类型: "can" = SocketCAN
    std::string type = "can";
};

// ============================================================================
// 输入系统总配置
// ============================================================================

struct InputSystemConfig {
    ImageInputConfig image;
    ImuInputConfig imu;
    AltimeterInputConfig altimeter;

    /// 若为 true，InputProvider 内部使用 RingBuffer + 采集线程（Camera 模式时自动启用）
    bool use_threaded_capture = false;
};

} // namespace input
} // namespace gpnp
