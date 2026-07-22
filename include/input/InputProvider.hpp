#pragma once

/**
 * @file InputProvider.hpp
 * @brief 输入系统统一协调器。
 *
 * 职责：
 *   1. 根据 InputSystemConfig 创建图像源 + 可选的 IMU / 高度计源
 *   2. 提供 getNextPacket() → 返回以相机帧为基准对齐的 SensorPacket
 *   3. 管理 TimeSyncUnit 进行多传感器时间对齐（Phase 2+）
 *
 * 使用示例：
 * @code
 *   InputProvider provider;
 *   if (provider.initialize(config)) {
 *       SensorPacket packet;
 *       while (provider.getNextPacket(packet)) {
 *           tracker.process(packet.left_image, packet.right_image, ...);
 *       }
 *   }
 * @endcode
 */

#include "input/InputConfig.hpp"
#include "input/SensorTypes.hpp"

#include <memory>
#include <string>

namespace gpnp {
namespace input {

// 前置声明（Phase 2 的传感器源）
class IImuSource;
class IAltimeterSource;
class TimeSyncUnit;

class InputProvider {
public:
    InputProvider();
    ~InputProvider();

    InputProvider(const InputProvider&) = delete;
    InputProvider& operator=(const InputProvider&) = delete;

    // ========================================================================
    // 初始化
    // ========================================================================

    /// 从配置初始化整个输入系统。
    /// @return 成功返回 true。
    bool initialize(const InputSystemConfig& config);

    // ========================================================================
    // 帧获取
    // ========================================================================

    /// 获取下一帧数据包。
    ///
    /// 内部流程：
    ///   1. 从图像源读取下一帧（左+右图）
    ///   2. 若有 IMU 源 → TimeSyncUnit 线性插值到图像时间戳
    ///   3. 若有高度计源 → TimeSyncUnit 指数加权融合到图像时间戳
    ///   4. 组装 SensorPacket 并返回
    ///
    /// @param[out] packet  对齐后的统一传感器数据包。
    /// @return 成功返回 true；到达末尾或错误返回 false。
    bool getNextPacket(SensorPacket& packet);

    // ========================================================================
    // 状态查询
    // ========================================================================

    /// 是否已成功初始化。
    bool isOpen() const;

    /// 总帧数（-1 表示未知）。
    int totalFrames() const;

    /// 当前帧序号（从 0 开始）。
    int currentFrame() const;

    /// 重置到第一帧（仅对 File/Directory 源有效）。
    bool reset();

    /// 获取内部配置的只读引用。
    const InputSystemConfig& config() const { return config_; }

private:
    /// 创建图像源（根据 ImageInputConfig::type 选择具体实现）。
    bool createImageSource();

    /// 创建 IMU 源（若启用）。Phase 2 实现。
    bool createImuSource();

    /// 创建高度计源（若启用）。Phase 2 实现。
    bool createAltimeterSource();

    InputSystemConfig config_;
    std::unique_ptr<class IStereoImageSource> image_source_;

    // Phase 2: 传感器源 + 时间同步（当前 Phase 1 使用 opaque pointer 避免不完整类型析构）
    void* imu_source_ = nullptr;         ///< IImuSource*, Phase 2 改为 unique_ptr
    void* altimeter_source_ = nullptr;   ///< IAltimeterSource*, Phase 2 改为 unique_ptr
    // TimeSyncUnit* 在 Phase 2 中从 extracted_input_system 引入
    void* time_sync_unit_ = nullptr;  // opaque pointer, Phase 2 中用 unique_ptr

    int current_frame_ = 0;
};

} // namespace input
} // namespace gpnp
