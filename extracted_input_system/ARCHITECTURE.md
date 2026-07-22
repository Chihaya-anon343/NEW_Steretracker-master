# 多传感器输入系统架构文档

> 提取自 NUAA 视觉导航与定位系统 (VNAL)，面向新项目的传感器输入系统设计参考。

---

## 1. 核心设计思想

这套输入系统解决的是**多传感器系统中一个普遍问题**：

> 如何同时从多种异构传感器（相机、IMU、雷达/RTK）采集数据，并以统一的时间基准输出同步数据包？

关键设计决策：

| 决策 | 做法 | 理由 |
|------|------|------|
| **每传感器独立线程** | 相机、IMU、RTK 各一个采集线程 | 各传感器数据到达速率不同（相机 25Hz / IMU 数百Hz / RTK 不定），单线程轮询会相互阻塞 |
| **环形缓冲区** | `std::deque` + 容量上限，满时丢弃最老数据 | 防止消费者慢于生产者时内存无限增长 |
| **以相机为主时钟** | 其他传感器数据向相机时间戳对齐 | 相机是视觉导航系统的主传感器，帧率稳定且是定位的起点 |
| **IMU 线性插值** | 不是简单的"最近邻"，而是前后两帧线性插值 | IMU 频率远高于相机，插值能更精确还原相机曝光时刻的真实惯性状态 |
| **雷达指数加权融合** | 不是取最新值，而是时间窗口内所有值按时间距离加权 | 雷达数据可能抖动，加权融合能平滑噪声 |
| **生产者-消费者解耦** | 采集线程只写缓冲，主线程只读 | 避免采集延迟影响主循环节拍 |

---

## 2. 整体架构图

```
                          ┌──────────────────────┐
                          │     配置/参数层        │
                          │ 设备路径、波特率、     │
                          │ 帧率、缓冲容量等       │
                          └──────────┬───────────┘
                                     │ 注入
    ┌────────────┬────────────┬──────┴──────┬────────────┐
    │            │            │             │            │
    ▼            ▼            ▼             ▼            ▼
┌────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────────────┐
│ Camera │ │   IMU   │ │  Radar   │ │   协议解析器(可插拔)   │
│ 线程   │ │  线程   │ │  线程    │ │ IImuParser            │
│        │ │         │ │          │ │ IRadarParser          │
│ V4L2   │ │ 串口    │ │ SocketCAN│ │ (项目特定实现)         │
└───┬────┘ └────┬────┘ └────┬─────┘ └──────────┬───────────┘
    │           │           │                   │
    ▼           ▼           ▼                   │
┌────────┐ ┌────────┐ ┌──────────┐              │
│Camera  │ │ IMU    │ │ Radar    │              │
│Buffer  │ │ Buffer │ │ Buffer   │              │
│[8帧]   │ │ [512条]│ │ [64条]   │              │
└───┬────┘ └───┬────┘ └────┬─────┘              │
    │          │           │                    │
    │          ▼           ▼                    │
    │   ┌─────────────────────────┐             │
    │   │    TimeSyncUnit         │◄────────────┘
    │   │  • IMU线性插值          │
    │   │  • 雷达指数加权融合     │
    │   └───────────┬─────────────┘
    │               │
    └───────┬───────┘
            ▼
    ┌──────────────────┐
    │ GetSyncedData()   │  ← 主循环调用
    │ 以相机时间戳为基准 │
    │ 组装同步数据包     │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  GenericSensorPacket │ → 消费者（检测/融合模块）
    └──────────────────┘
```

---

## 3. 可复用组件（本目录提供）

### 3.1 CanSocket — Linux SocketCAN C++ RAII 封装

**文件**: `can_socket/can_socket.hpp` + `can_socket/can_socket.cpp`  
**依赖**: 仅 Linux 内核头文件（`<linux/can.h>`, `<linux/can/raw.h>`）+ POSIX socket  
**命名空间**: `nanhang::`

**提供的能力**:

```cpp
#include "can_socket.hpp"

// 1. 连接到 CAN 接口
nanhang::CanSocket can;
can.canConnect("can0");          // 自动 socket() → ioctl(SIOCGIFINDEX) → bind()

// 2. 发送 CAN 帧
uint8_t data[8] = {0xB8, 0x00, 0x00, 0x00, 0x08, 0x8D, 0x00, 0x00};
nanhang::CanFrame cmd(0x200, data, 8);
can.sendFrame(cmd);

// 3. 接收 CAN 帧（带超时）
nanhang::CanFrame rx;
if (can.receiveFrame(rx, 100)) {   // 100ms 超时
    // rx.id, rx.dlc, rx.data[]
}

// 4. 错误处理
const char* err = can.getLastError();  // "Receive timeout" 等

// 5. 自动资源管理
can.disconnect();  // 析构时自动调用
```

**为什么需要这个封装？** Linux 内核只提供 C 语言的 `socket(PF_CAN, SOCK_RAW, CAN_RAW)` 系统调用，你需要手动处理 `struct sockaddr_can`、`struct ifreq`、`ioctl(SIOCGIFINDEX)`、`bind()`、`struct can_frame` 转换等底层细节。这个封装把这些变成一行 `canConnect("can0")`。

**移植注意**:
- 纯 Linux 模块，不适用 Windows/macOS
- 需要内核编译了 SocketCAN 支持 (`CONFIG_CAN=y`, `CONFIG_CAN_RAW=y`)
- 使用前需配置 CAN 接口: `sudo ip link set can0 type can bitrate 500000 && sudo ip link set up can0`

---

### 3.2 TimeSyncUnit — 多传感器时间同步

**文件**: `time_alignment/timeAlignment.hpp` + `time_alignment/timeAlignment.cpp`  
**依赖**: Eigen3（仅 `Eigen::Vector3d`）

**提供的能力**:

```cpp
#include "timeAlignment.hpp"

TimeSyncUnit sync(/*radar_alpha=*/50.0, /*radar_max_dt=*/0.2);

// 每当收到 IMU 数据时喂入
sync.addIMU(t_seconds, accel_vector, gyro_vector);

// 每当收到雷达数据时喂入
sync.addRadar(t_seconds, height_meters);

// —— 时间对齐查询 ——

// 1. 获取 t_cam 时刻的 IMU 线性插值
IMUSample imu;
if (sync.getInterpolatedIMU(t_cam, imu)) {
    // imu.acc, imu.gyro 是精确插值结果
}

// 2. 获取 t_cam 时刻的雷达高度（指数加权融合）
double height;
if (sync.getSyncedRadarHeight(t_cam, height)) {
    // height 是时间加权融合结果
}

// 3. 消费 t_cam 之前的所有 IMU 数据（用于 ESKF 预测）
std::vector<IMUSample> imu_history = sync.getIMUUntil(t_cam);
// 返回后这些数据从内部缓冲中移除
```

**算法细节**:

```
IMU 插值（线性）:
  已知: (t₁, acc₁, gyro₁) 和 (t₂, acc₂, gyro₂)，且 t₁ < t_cam < t₂
  α = (t_cam - t₁) / (t₂ - t₁)
  acc_cam  = acc₁  + α × (acc₂  - acc₁)
  gyro_cam = gyro₁ + α × (gyro₂ - gyro₁)

雷达融合（指数加权）:
  对缓冲中所有满足 |t - t_cam| < radar_max_dt 的样本:
  w_i = exp(-radar_alpha × |t_cam - t_i|)
  height = Σ(w_i × h_i) / Σ(w_i)
```

**内部缓冲管理**:
- IMU 缓冲保留 2 秒窗口，超出自动清理
- 雷达缓冲保留 `2 × radar_max_dt` 窗口
- 都是 `std::deque`，旧数据在前，新数据在后

---

## 4. 适配新项目的步骤

### 第 1 步：确定你的传感器清单

列出新项目用到的所有传感器，明确每个传感器的：

| 属性 | 示例 |
|------|------|
| 硬件接口 | V4L2 / 串口 / SocketCAN / SPI / I2C / UDP / 文件 |
| 数据速率 | 25Hz / 200Hz / 事件驱动 / 不定 |
| 协议格式 | 标准协议 / 自定义二进制帧 / ASCII 文本 |
| 输出数据类型 | 图像 / 加速度+角速度 / 高度 / 经纬度 |

### 第 2 步：定义通用传感器数据包

替换项目特定的 `SensorData`，设计你自己的：

```cpp
struct SensorPacket {
    int64_t timestamp_us;   // 统一时间基准（建议微秒）

    // 相机（可选）
    std::optional<cv::Mat>  camera_frame;

    // IMU（可选）
    std::optional<Eigen::Vector3d> accel;   // m/s²
    std::optional<Eigen::Vector3d> gyro;    // rad/s

    // 高度（可选）
    std::optional<double>   height;         // m

    // 扩展：GPS、磁力计、气压计等...
    std::optional<double>   latitude, longitude;
    std::optional<Eigen::Vector3d> mag;
};
```

### 第 3 步：实现协议解析器（新项目最费时的部分）

这是**每个项目必须自己写的部分**。为每个传感器写一个 parser：

```cpp
class MyImuParser {
public:
    // 喂入原始字节，返回解析出的完整帧数
    int feed(const uint8_t* raw_data, int len) {
        // 1. 追加到内部缓冲
        buffer_.insert(buffer_.end(), raw_data, raw_data + len);

        int parsed = 0;
        // 2. 搜索帧头
        while (buffer_.size() >= FRAME_LEN) {
            if (buffer_[0] == 0x55 && buffer_[1] == 0xAA) {  // 你的帧头
                // 3. 校验和
                if (checksum_ok()) {
                    // 4. 解析字段并输出
                    ImuPacket pkt = parse_frame();
                    output_queue_.push(pkt);
                    buffer_.erase(buffer_.begin(), buffer_.begin() + FRAME_LEN);
                    parsed++;
                } else {
                    buffer_.pop_front();  // 校验失败，滑1字节
                }
            } else {
                buffer_.pop_front();  // 未找到帧头，滑1字节
            }
        }
        return parsed;
    }

    // 消费者调用：取出已解析的数据
    bool getNext(ImuPacket& out) {
        if (output_queue_.empty()) return false;
        out = output_queue_.front();
        output_queue_.pop();
        return true;
    }

private:
    std::deque<uint8_t> buffer_;
    std::queue<ImuPacket> output_queue_;
    static constexpr int FRAME_LEN = 57;  // 你的协议帧长度
};
```

### 第 4 步：搭建采集线程 + 环形缓冲（模式固定，代码模板化）

```cpp
template<typename DataType>
class RingBuffer {
public:
    explicit RingBuffer(size_t cap) : capacity_(cap) {}

    void push(DataType&& item) {
        std::unique_lock lock(mutex_);
        buffer_.push_back(std::move(item));
        if (buffer_.size() > capacity_) {
            buffer_.pop_front();
            dropped_++;
        }
    }

    bool getLatest(DataType& out) {
        std::shared_lock lock(mutex_);
        if (buffer_.empty()) return false;
        out = buffer_.back();
        return true;
    }

    size_t size() const {
        std::shared_lock lock(mutex_);
        return buffer_.size();
    }

private:
    std::deque<DataType> buffer_;
    size_t capacity_;
    size_t dropped_ = 0;
    mutable std::shared_mutex mutex_;
};

// 相机采集线程模板
void cameraThread(RingBuffer<CameraFrame>& cam_buf, 
                  const std::string& device,
                  std::atomic<bool>& running) {
    cv::VideoCapture cap(device, cv::CAP_V4L2);
    while (running) {
        cv::Mat frame;
        cap >> frame;
        if (!frame.empty()) {
            cam_buf.push({frame, getTimestampUs()});
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40)); // 25fps
    }
}

// IMU 采集线程模板
void imuThread(RingBuffer<ImuPacket>& imu_buf,
               MyImuParser& parser,
               const std::string& serial_port,
               std::atomic<bool>& running) {
    int fd = openSerial(serial_port, B921600);
    uint8_t raw[512];
    while (running) {
        int n = read(fd, raw, sizeof(raw));
        if (n > 0) {
            int parsed = parser.feed(raw, n);
            ImuPacket pkt;
            while (parser.getNext(pkt)) {
                imu_buf.push(std::move(pkt));
            }
        }
    }
}
```

### 第 5 步：实现同步输出接口

```cpp
bool getSyncedPacket(SensorPacket& out) {
    // 1. 取最新相机帧作为时间基准
    CameraFrame cam;
    if (!cam_buf_.getLatest(cam)) return false;

    int64_t t_ref = cam.timestamp_us;

    // 2. IMU：线性插值到 t_ref
    out.accel = sync_.interpolateAccel(t_ref);
    out.gyro  = sync_.interpolateGyro(t_ref);

    // 3. 雷达：指数加权融合到 t_ref
    out.height = sync_.fuseRadarHeight(t_ref);

    // 4. 组装
    out.timestamp_us = t_ref;
    out.camera_frame = cam.frame;
    return true;
}
```

### 第 6 步：在主循环中消费

```cpp
int main() {
    // 启动采集线程
    std::atomic<bool> running{true};
    std::thread cam_th(cameraThread, std::ref(cam_buf), "/dev/video0", std::ref(running));
    std::thread imu_th(imuThread, std::ref(imu_buf), std::ref(parser), "/dev/ttyS0", std::ref(running));
    // ...

    // 主循环
    SensorPacket pkt;
    while (running) {
        if (getSyncedPacket(pkt)) {
            processDownstream(pkt);  // 交给下游模块
        }
    }
}
```

---

## 5. 关键设计模式总结

### 模式 1：环形缓冲 + 读写锁

```
                   std::shared_mutex
                        │
    生产者(写) ── unique_lock ──→ deque ←── shared_lock ── 消费者(读)
                                                        （多个并发读）
```

- 生产者用 `unique_lock`（独占），消费者用 `shared_lock`（共享）
- `std::deque` 支持 O(1) 头尾操作
- 容量上限防止内存泄漏，满时 `pop_front()` 丢弃最老数据
- 维护 `dropped_` 计数器用于监控

### 模式 2：相机主时钟

选择帧率最稳定、对定位最重要的传感器作为时间基准。在视觉导航系统中这是相机。在纯惯性系统中可能是 IMU，在 GNSS/INS 组合中可能是 GPS PPS 脉冲。

### 模式 3：协议解析器与采集循环分离

**错误的做法**（本项目当前的方式）：
```
采集循环 = 串口读取 + 协议解析 + 写入缓冲   ← 三者耦合
```

**正确的做法**（可复用的方式）：
```
采集循环 = 串口读取 → 喂入 Parser
Parser   = 协议解析 → 输出数据包
                           ↓
                       写入缓冲
```

分离后，换一个传感器只需替换 Parser，采集循环不变。

### 模式 4：消费者拉取而非生产者推送

采集线程只负责把数据放入环形缓冲，**不主动**通知消费者。主循环按自己的节拍**拉取**数据。这避免了：
- 采集速度波动影响主循环
- 需要复杂的事件/条件变量通知机制
- 多消费者时的分发问题

---

## 6. 本目录文件清单

```
extracted_input_system/
├── ARCHITECTURE.md              ← 本文档（架构说明 + 适配指南）
├── CMakeLists.txt               ← 构建配置
├── can_socket/                  ← 可复用模块 1：SocketCAN 封装
│   ├── can_socket.hpp
│   └── can_socket.cpp
└── time_alignment/              ← 可复用模块 2：时间同步
    ├── timeAlignment.hpp
    └── timeAlignment.cpp
```

### 在你的新项目中使用

```cmake
# 方式一：直接添加子目录
add_subdirectory(extracted_input_system)
target_link_libraries(your_app can_socket time_alignment)

# 方式二：仅拷贝需要的文件
# 拷贝 can_socket/ 或 time_alignment/ 到你的项目中
# CanSocket 仅需 Linux + pthread
# TimeSyncUnit 仅需 Eigen3
```

---

## 7. 经验教训 / 注意事项

1. **不要在采集线程中做重活**。采集线程的唯一职责是读硬件 → 解析 → 写入缓冲。任何计算密集型操作（图像预处理、滤波）应该放在下游消费者中。

2. **协议解析的字节对齐陷阱**。自定义二进制协议中，不同传感器的字节序可能不同。本项目 IMU 协议使用大端序，CAN 数据是小端序。确保 parser 中有明确的大小端转换。

3. **时间戳来源要统一**。所有传感器的时间戳应来自同一时钟源（如 `CLOCK_MONOTONIC`），或者有一个明确的基准时钟。不要混用系统时间和硬件时间戳而不做对齐。

4. **缓冲容量需要根据实际场景调优**。太大会增加内存和延迟，太小会导致丢帧。建议：
   - 相机缓冲：2-3 倍帧间隔的数据量（8 帧 @25fps ≈ 320ms 窗口）
   - IMU 缓冲：覆盖相机帧间隔的 10-20 倍（512 条 @200Hz ≈ 2.5s 窗口）
   - 雷达缓冲：覆盖 2-3 倍最大观测间隔

5. **IMU 插值在边界条件下会失败**。当 `t_cam` 超出缓冲范围时（所有数据都比它旧或都比它新），插值返回 false。要有降级策略（最近邻回退）。

6. **串口数据可能不完整到达**。`read()` 不一定返回完整的一帧。本项目的处理方式是累积到内部缓冲，逐字节滑动搜索帧头。这是一种稳健的做法。

7. **单例进程锁**。本项目使用 PID 文件锁确保单实例。对于嵌入式系统这是好习惯，但不是输入系统的必要部分。
