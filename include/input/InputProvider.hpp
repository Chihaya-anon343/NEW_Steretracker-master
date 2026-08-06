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
#include "input/RingBuffer.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

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
    /// @param[out] packet    对齐后的统一传感器数据包。
    /// @param timeout_ms     线程化模式的等待超时 (毫秒)。-1 = 无限阻塞;
    ///                       有限超时无帧时返回 false (调用方应检查
    ///                       isCaptureStopped() 区分"源结束"与"超时")。
    /// @return 成功返回 true；到达末尾/采集停止/超时返回 false。
    bool getNextPacket(SensorPacket& packet, int timeout_ms = -1);

    /// 线程化采集是否已停止（源结束 / 摄像头断开 / 显式 shutdown）。
    /// 同步模式恒为 true——getNextPacket 返回 false 即源结束。
    bool isCaptureStopped() const;

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

    // ========================================================================
    // 线程化采集 (Phase 3.1)
    // ========================================================================

    /// 停止采集线程并等待其退出（join）。
    /// 析构时自动调用；主循环结束后手动调用以获得稳定统计。
    void shutdown();

    /// 输入运行统计（线程化采集模式; 同步模式恒为 0）。
    struct InputStats {
        int64_t captured = 0;   ///< 采集线程从图像源取得的帧数
        int64_t consumed = 0;   ///< 消费者成功取出的帧数
        int64_t dropped = 0;    ///< 未被消费的帧数 (缓冲溢出 + take-latest 跳帧)
    };
    InputStats stats() const;

private:
    /// 采集线程主循环: 图像源 nextFrame → 推入环形缓冲。
    /// 仅线程化采集模式运行; 摄像头断开/读取失败时置 capture_failed_ 并退出。
    void captureLoop();

    /// 创建图像源（根据 ImageInputConfig::type 选择具体实现）。
    bool createImageSource();

    /// 创建 IMU 源（若启用）。Phase 2 实现。
    bool createImuSource();

    /// 创建高度计源（若启用）。Phase 2 实现。
    bool createAltimeterSource();

    InputSystemConfig config_;
    std::unique_ptr<class IStereoImageSource> image_source_;

    // ---- Phase 3.1: 线程化采集状态 ----
    bool threaded_capture_ = false;
    std::thread capture_thread_;
    std::atomic<bool> capture_running_{false};
    std::atomic<bool> capture_failed_{false};
    mutable std::mutex queue_cv_mtx_;          ///< 与 queue_cv_ 配合的互斥量
    std::condition_variable queue_cv_;         ///< 消费者阻塞等待新帧
    std::unique_ptr<RingBuffer<SensorPacket>> frame_ring_;  ///< 帧缓冲 (take-latest)
    std::atomic<int64_t> captured_frames_{0};  ///< 采集线程产帧数
    std::atomic<int64_t> consumed_frames_{0};  ///< 消费者取帧数

    // Phase 2: 传感器源 + 时间同步（当前 Phase 1 使用 opaque pointer 避免不完整类型析构）
    void* imu_source_ = nullptr;         ///< IImuSource*, Phase 2 改为 unique_ptr
    void* altimeter_source_ = nullptr;   ///< IAltimeterSource*, Phase 2 改为 unique_ptr
    // TimeSyncUnit* 在 Phase 2 中从 extracted_input_system 引入
    void* time_sync_unit_ = nullptr;  // opaque pointer, Phase 2 中用 unique_ptr

    int current_frame_ = 0;
};

} // namespace input
} // namespace gpnp
