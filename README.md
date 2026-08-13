# Steretracker — GPNP 双目视觉跟踪器

基于 **YOLO 检测 → 五状态分级 → 策略选择 → 特征提取 → 位姿解算** 流水线的 C++17 双目/单目视觉定位系统，面向无人机视觉定位场景。

**技术栈**: C++17 · OpenCV 4.x · Eigen 3.x · ONNX Runtime

---

## 0. 系统需求概述

本项目的系统需求规格定义于 [sysml/sysrequire.puml](sysml/sysrequire.puml)，采用 SysML 需求图建模。顶层需求分解为七个子系统：

| 需求编号     | 子系统        | 对应章节                                      |
| ------------ | ------------- | --------------------------------------------- |
| SYS-REQ-100  | 图像输入系统  | [§2 运行模式](#2-运行模式-run-mode) / [§3 图像输入系统](#3-图像输入系统-inputprovider) |
| SYS-REQ-200  | 目标检测与 ROI 生成 | [§4 目标检测与五状态分级](#4-目标检测与五状态分级) |
| SYS-REQ-300  | 五状态目标分级与特征提取 | [§5 五状态策略](#5-五状态策略) |
| SYS-REQ-400  | 位姿解算系统  | [§6 位姿解算](#6-位姿解算) |
| SYS-REQ-500  | 可视化与输出  | [§7 可视化与输出](#7-可视化与输出) |
| SYS-REQ-600  | 配置与运行模式 | [§2 运行模式](#2-运行模式-run-mode) / [§10 配置文件](#10-配置文件) |
| SYS-REQ-700  | 构建与部署    | [§1 快速开始](#1-快速开始) / [§12 构建与运行](#12-构建与运行) / [§13 项目目录结构](#13-项目目录结构) |

### 整体流程

```
                          ┌─────────────────┐
                          │ tracker_config.json │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  mode ?                          │
                    └──────┬──────────────┬───────────┘
                    normal │              │ debug
                           ▼              ▼
                    ┌──────────────┐  ┌──────────────────────┐
                    │ InputProvider │  │ cv::imread + 手动/YOLO│
                    │(File/Dir/Seq/ │  │ verbose_console=true  │
                    │     Camera)   │  │                      │
                    └──────┬───────┘  └──────────┬───────────┘
                           │                     │
                           └──────┬──────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │      逐 帧 循 环             │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  mono_mode ?                     │
                    └──────┬──────────────┬───────────┘
                      true │              │ false
                           ▼              ▼
                    ┌──────────────────────┐  ┌──────────────────────┐
                    │  MonoTracker::process│  │StereoTracker::process│
                    │  (EPnP, 无立体)      │  │(GPnP + 光流 + MAD)   │
                    └──────────┬───────────┘  └──────────┬───────────┘
                               │                         │
                               └──────┬──────────────────┘
                                      ▼
                    ┌──────────────────────────────────────┐
                    │ 五状态分级 (双眼共享判定)             │
                    │ State 1远 → 2中 → 3中近 → 4近 → 5极近 │
                    │ 退化链: AKAZE → BC → TinyTarget 终止  │
                    │ Dual-ROI 独立路径 / 无检测 skip       │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 输出位姿 [R|t] + 日志 + 可视化 │
                    └─────────────────────────────┘
```

---

## 1. 快速开始

```bash
# 编译
git clone https://github.com/Chihaya-anon343/NEW_Steretracker.git
cd NEW_Steretracker
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 运行 (默认配置 config/tracker_config.json)
./Steretracker

# 指定配置文件
./Steretracker path/to/config.json
```

**依赖**: C++17 · OpenCV 4.x · Eigen 3.x · ONNX Runtime

---

## 2. 运行模式 (Run Mode)

系统通过顶层 `mode` 字段（`"normal"` 或 `"debug"`）控制运行行为：

|          | Normal                              | Debug                     |
| -------- | ----------------------------------- | ------------------------- |
| 图像源   | `InputProvider` (input_system 节: 实时 `camera` 或离线 `file`/`directory`/`sequence`) | `cv::imread` (input 节: 离线固定图像) |
| ROI      | YOLO 检测                           | 手动 ROI 或 YOLO          |
| 终端输出 | 每帧一行简介 (`[Frame N] Strategy n=... t=...`) | 详细统计                  |
| 可视化   | 仅三维坐标轴叠加 (`mono_f{N}.png`)  | 各策略完整中间面板        |
| 日志文件 | 可选 — `output.log_file: true` 时输出 `tracking_log.txt`（完整处理过程 + 配置摘要） | — |

两种模式均支持 `mono_mode: true/false` 切换双目与单目。

---

## 3. 图像输入系统 (InputProvider)

> 对应 SYS-REQ-100 系列

> **输入分类**：输入按数据来源明确分为两大类：
> - **实时相机输入** — `input_system.image.type: "camera"`（`CameraSource`，Phase 3 已实现，支持线程化采集，见 [§3.5 实时摄像头模式](#35-实时摄像头模式cameraSource)）
> - **离线图像输入** — `input_system.image.type: "file" | "directory" | "sequence"`（静态文件对 / 双目编号序列 / 单目序列）；Debug 模式绕过 InputProvider，直接 `cv::imread` 读取固定图像，同样属于离线输入

### 3.1 架构

```
┌──────────────────────────────────────────────────────────────┐
│                       InputProvider                           │
│  统一协调器：创建图像源 → 组装 SensorPacket                      │
├──────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐  ┌────────────────────┐              │
│  │ IStereoImageSource  │  │ TimeSyncUnit       │ (Phase 2)   │
│  │ (图像源抽象接口)     │  │ (IMU 插值 + 融合)   │              │
│  └────────┬───────────┘  └────────────────────┘              │
│           │                                                   │
│     ┌─────┼─────────────┬──────────────┐                     │
│     ▼     ▼             ▼              ▼                      │
│  ┌────┐ ┌──────────┐ ┌──────────┐ ┌────────┐                │
│  │File│ │Directory │ │Sequence  │ │Camera  │                │
│  │文件│ │双目编号序│ │单目序列  │ │实时摄像│ (Phase 3 ✅)    │
│  │对  │ │列        │ │          │ │头      │                │
│  └────┘ └──────────┘ └──────────┘ └────────┘                │
│                                                               │
│  输出: SensorPacket { left, right, timestamp, IMU, 高度 }     │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 四种图像源模式

| 分类         | type            | 类                        | 用途                                   |
| ------------ | --------------- | ------------------------- | -------------------------------------- |
| 离线图像输入 | `"file"`      | `FileStereoSource`      | 静态双目文件对                         |
| 离线图像输入 | `"directory"` | `DirectoryStereoSource` | 双目编号图像序列                       |
| 离线图像输入 | `"sequence"`  | `SequenceSource`        | 单目图像序列                           |
| 实时相机输入 | `"camera"`   | `CameraSource`          | **实时摄像头** (Phase 3, 单目, USB/内置 webcam) |

> **CameraSource 要点**:
> - OpenCV `VideoCapture` 打开: `"0"` → 设备索引; `/dev/video0` → 设备路径; 支持 `"0;1"` 多设备格式 (当前取第一个)
> - 阻塞读帧 (`cap_->grab()`), 天然按摄像头帧率节流; `target_fps` 额外 sleep 限速; 时间戳在 grab 返回后立即打戳 (接近曝光时刻)
> - 单目语义: 右图 = 左图副本; `totalFrames() = -1` (帧数未知)
> - 打开后 `grab()` 预热 5 帧 (等自动曝光/白平衡收敛)
> - **线程化采集 (Phase 3.1)**: Camera 类型自动启用采集线程 + RingBuffer (take-latest 策略), 读帧与 YOLO/提取解耦 — 处理慢时跳过旧帧而非累积延迟 (见 [§3.5 实时摄像头模式](#35-实时摄像头模式cameraSource))

### 3.3 SensorPacket 统一数据包

```cpp
struct SensorPacket {
    int64_t timestamp_us;
    cv::Mat left_image, right_image;
    std::optional<ImuPacket> imu;      // Phase 2 预留
    std::optional<HeightPacket> height;// Phase 2 预留
    bool valid = false;
};
```

### 3.4 Phase 规划

| Phase   | 状态      | 内容                                     |
| ------- | --------- | ---------------------------------------- |
| Phase 1 | ✅ 已完成 | 图像源抽象 + RingBuffer |
| Phase 2 | 占位      | IMU + 高度计 + TimeSyncUnit (`extracted_input_system/` 提供 CanSocket/TimeSyncUnit 参考实现, 未接入) |
| Phase 3 | ✅ 已完成 | 实时摄像头源 (CameraSource, 同步阻塞读帧) |
| Phase 3.x | ✅ 已完成 | 线程化采集 (采集线程 + RingBuffer take-latest) + FPS/丢帧统计 + Ctrl+C 优雅退出 |

### 3.5 实时摄像头模式 (CameraSource)

实时模式配置示例见 [config/tracker_config_webcam.json](config/tracker_config_webcam.json)：

```jsonc
"input_system": {
    "max_frames": 0,                       // 0 = 无限循环
    "image": { "type": "camera", "camera_devices": "0", "target_fps": 10 }
},
"mono_mode": true                          // 单摄像头必须单目
```

| 要点 | 说明 |
|------|------|
| 运行方式 | `./build/Steretracker config/tracker_config_webcam.json` |
| 帧循环 | `main.cpp` `while (getNextPacket() && frame < max_frames && !Ctrl+C)` 逐帧 YOLO → 策略 → 位姿 |
| 采集模型 | **线程化 (Phase 3.1)**: 采集线程按相机节拍产帧入环形缓冲; 主循环 take-latest 取最新帧 — 处理慢时跳帧而非累积延迟, 丢帧数可统计 |
| 结束方式 | 摄像头断开 → `getNextPacket()` false 退出; **Ctrl+C → 优雅收尾**: 停止采集 → 打印汇总 (处理帧数/位姿成功率/实际 FPS/采集-消费-丢弃统计) → 日志落盘 |
| 日志 | `log_file: true` 时退出后写 `output/sequence/tracking_log.txt` (含配置摘要 + 逐帧详情 + 运行汇总) |
| 相机内参 | ⚠️ webcam 配置为 640×480 占位值, **必须棋盘格标定后替换** fx/fy/cx/cy |
| 当前局限 | 仅单目 (USB 双目/RTSP 未支持); `target_fps` 依赖摄像头驱动对 CAP_PROP_FPS 的支持 |

### 3.6 配置

参见 [§10 配置文件](#10-配置文件) 中的 `input_system` 节。

---

## 4. 目标检测与五状态分级

> 对应 SYS-REQ-200 系列

### 4.1 YOLO ONNX 检测

**模块**: `YoloDetector` + `YoloRoiProvider` + `RoiGenerator`

基于 ONNX Runtime 运行 YOLO 推理（`best.onnx`），输出目标类别 (class 0 → 整体 / class 1 → 中心区域) 与边界框。

检测接口因模式不同：

- **双目**: `yolo.detect(L, R)` — 对左右图**各推理一次**（并行，`std::async`），双侧均有检测时生成相互独立的 ROI 并立体配对；仅单侧检测时另一侧 invalid（main.cpp 依 `stereo_mono_fallback` 决定单目降级或跳过）
- **单目**: `yolo.detectMono(L)` — 仅左图推理

> YOLO 未检测到任何目标时 **直接跳过当前帧**，不触发策略退化链。退化仅在特征提取失败时发生。

### 4.2 五状态决策树

`RoiGenerator::generateGroup()` 根据检测结果和面积完成五状态判定，双目与单目共享同一逻辑：

```
                     YOLO 检测
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    有 class0        无 class0       无 class0
    有/无 class1    有 class1       无 class1
        │               │               │
   按面积分级      State 5 极近     SKIP 帧
        │         (class1回退)
        │
  ┌─────┼──────────┬──────────┐
  │     │          │          │
 ≤800  801~40000  40001~      ≥490000
  │     │       <490000       + 有 class1
  │     │          │          │
State1  State2   State3     State4
  远      中      中近        近
TinyT.  BC      AKAZE     Dual-ROI
```

### 4.3 五状态总览表

| State | 名称 | 条件 | 策略 | 双目 Pose | 单目 Pose |
|-------|------|------|------|-----------|-----------|
| **1 远** | Distant | 有 class0, 面积 ≤ tiny_max | TinyTarget | `cv::solvePnP(ITERATIVE)` | `MonoPnP(EPnP)` |
| **2 中** | Medium | 有 class0, 801~40000 | BinaryCorner | `InitialPnP→GPnP(warm)` | `MonoPnP(EPnP)` |
| **3 中近** | Medium-Close | 有 class0, 40001~489999 或 ≥490000 无 class1 | AKAZE 单ROI | `InitialPnP→GPnP(warm)` | `MonoPnP(EPnP)` |
| **4 近** | Close | 有 class0 + class1, ≥490000 | Dual-ROI | `InitialPnP→GPnP(warm)` | `MonoPnP(EPnP)` |
| **5 极近** | Very-Close | 无 class0 + 有 class1, ≥class1_min_area | 按面积重分类 | 按重分类 | 按重分类 |

> ⚠️ **阈值说明**: `tiny_max_area` / `akaze_min_area` / `dual_trigger_area` 为早期测试遗留值，需在实际场景中重新标定。当前仅起占位作用，不影响系统架构。

### 4.4 单 ROI 生成

- 按面积筛选 class 0 最大边界框
- 按 `roi_expand_ratio` 外扩 ROI
- 确保 ROI 不小于 `roi_min_size`
- **忽略 class 1 检测结果**

### 4.5 双 ROI 生成

**触发条件**: class 0 面积 > `dual_trigger_area` (占位值: 490000 px²)，且存在 class 1

- **primary (class 0)**: 目标整体边缘 → BinaryCorner 提取
- **secondary (class 1)**: 目标中心纹理 → AKAZE 提取

不满足时降级为单 ROI (class 0)。

### 4.6 State 5 — 近距离 class1 回退 (CloseRange)

> 配置: `strategies.close_range`

当 class 0 丢失但 class 1 存在且面积 ≥ `class1_min_area` 时，使用 class 1 生成 ROI 并按面积重分类走标准 State 1~4 策略链。

配置项：
| 配置 | 说明 |
|------|------|
| `close_range.enabled` | 是否启用回退 |
| `close_range.class1_min_area` | class 1 最小面积阈值 |
| `close_range.roi_expand_ratio` | ROI 外扩比例 |
| `close_range.min_expand_pixels` | 最小外扩像素 |
| `close_range.akaze_min_area` | class1-only 时的 AKAZE 策略阈值 (0=使用默认) |
| `close_range.tiny_max_area` | class1-only 时的 TinyTarget 策略阈值 (0=使用默认) |

> `close_range.akaze_min_area` 和 `close_range.tiny_max_area` 允许 class1-only 场景使用与 class0 不同的面积阈值来选择策略链。

---

## 5. 五状态策略

> 对应 SYS-REQ-300 系列

### 5.1 State 1 远 — TinyTarget

**适用场景**: 远距离微小矩形目标（如 4 角点标定板）

| 维度 | 双目 | 单目 |
|------|------|------|
| **提取方法** | `extract()` (左右双路) | `extractMono()` (仅左图) |
| **流程** | Otsu→模板匹配(IoU)→超分(×4)→连通域评分→minAreaRect→角度对齐 | 同双目但仅左图 |
| **左右匹配** | 模板匹配 L↔R | — |
| **视差滤波** | — | — |
| **Pose** | `cv::solvePnP(ITERATIVE)` | `MonoPnPSolver(EPnP)` |
| **Warm-start** | ❌ | ❌ |
| **退化** | 终止(无后备) | 终止(无后备) |

**立体流水线** (双目):

```
Otsu二值化 → 模板匹配(IoU) → 超分辨率×4 → GaussianBlur → Otsu → 形态学
→ 连通域评分(面积/矩形度/中心距/长宽比) → minAreaRect → cornerSubPix亚像素
→ 角度对齐(4象限量化) → 视差计算 → cv::solvePnP(ITERATIVE)
```

### 5.2 State 2 中 — BinaryCorner

**适用场景**: 中等尺寸多边形目标（如 10 角点标识板）

| 维度 | 双目 | 单目 |
|------|------|------|
| **提取方法** | `extract()` | `extractMono()` |
| **流程** | Otsu→最大连通域→模板匹配(IoU)→旋转回正→approxPolyDP→重排序 | 同双目仅左图 |
| **左右匹配** | 模板匹配 L↔R | — |
| **Pose** | `InitialPnP→GPnP(warm-start)` | `MonoPnP(EPnP)` |
| **Warm-start** | ✅ 帧间缓存 | ❌ |
| **退化** | → TinyTarget | → TinyTarget |

**首帧初值** (双目):

1. 若 `use_initial_pnp=true`，运行 `InitialPnPSolver` (RANSAC 300 iter + ITERATIVE 精化)，成功 → GPNP warm-start
2. 失败或 `use_initial_pnp=false` → 视差估算深度 `depth = f·b/median_disp`，clamp [50, 5000]

### 5.3 State 3 中近 — AKAZE_GPNP

**适用场景**: 大尺寸/纹理丰富目标

| 维度 | 双目 | 单目 |
|------|------|------|
| **提取方法** | `extract()` (resize×scale → AKAZE → 光流 → 投影 → 模板匹配) | `extractMono()` (resize×scale → AKAZE → 模板匹配) |
| **光流** | ✅ LK L→R + FB校验(<1.0px) + MAD视差滤波 | ❌ |
| **立体投影** | ✅ 视差→深度→右图重投影 | ❌ |
| **MAD滤波** | ✅ (±3σ) | ❌ |
| **模板匹配** | Ratio Test(0.75)→Cross-Check→Homography RANSAC(5.0px) | 同双目 |
| **Pose** | `InitialPnP→GPnP(warm-start)` | `MonoPnP(EPnP)` |
| **Warm-start** | ✅ 帧间缓存 | ❌ |
| **退化** | → BinaryCorner → TinyTarget | → BinaryCorner → TinyTarget |

> **State 3 中 class 1 的处理**: 即使 YOLO 同时检测到 class 0 和 class 1，若面积未达到 `dual_trigger_area`，仍走 AKAZE 单 ROI (class 0)，忽略 class 1。只有面积 ≥ `dual_trigger_area` 且有 class 1 时，才进入 State 4。

### 5.4 State 4 近 — Dual-ROI 混合

**适用场景**: 大目标同时可见边缘 (class 0) 与中心 (class 1)

| 维度 | 双目 | 单目 |
|------|------|------|
| **提取方法** | BC `extract()` : AK `extract()` 并行 | BC `extractMono()` : AK `extractMono()` 并行 |
| **左右匹配** | BC: 模板匹配 L↔R / AK: LK光流 L→R | — |
| **MAD滤波** | AK 侧 | — |
| **合并** | BC角点 + AK关键点 → 统一 2D/3D 对应 | BC角点 + AK关键点 → 统一 2D/3D |
| **Pose** | `InitialPnP→GPnP(warm-start)` | `MonoPnP(EPnP)` |
| **Warm-start** | ✅ 帧间缓存 | ❌ |
| **退化** | 独立路径，不参与退化链 | 独立路径，不参与退化链 |
| **可视化** | 5 面板 | 3 面板 |

**合并阶段的 3D 对应**:

- **BC 贡献**: `pts_left_match[i]` ↔ `dual_bc_tmpl_pts3d_[i]`
- **AK 贡献**: `pts_left_match[i]` ↔ `ak_pts3d[trainIdx]`

### 5.5 State 5 极近 — CloseRange 回退

当 class 0 因遮挡/出画面而丢失，但 class 1 面积 ≥ `class1_min_area`，用 class 1 生成 ROI → 按面积重分类 → 走 State 1~4 对应策略链。

### 5.6 退化后备链

当主策略特征提取失败时，自动退化到低一级精度：

```
AKAZE_GPNP  ──失败──→  BinaryCorner  ──失败──→  TinyTarget  (终止)
```

| 说明 |
|------|
| Dual-ROI (State 4) 为独立路径，不进入退化链 |
| YOLO 无检测时直接 skip 帧，不走退化链 |
| 退化链由 `configureStrategyChain(int roi_area)` 自动完成，对外不可见 |

---

## 6. 位姿解算

> 对应 SYS-REQ-400 系列

### 6.1 GPnPSolver (双目通用)

基于 **Eigen Levenberg-Marquardt** 非线性优化，最小化双目交叉射线残差：

```
残差 = cross(P_3d - origin, direction_left) + cross(P_3d - t_rl, direction_right)
```

优化参数: `[qx, qy, qz, qw, tx, ty, tz]` (7 维)，支持帧间 warm-start。

### 6.2 InitialPnPSolver (双目首帧)

RANSAC PnP (300 iter, 8.0px → 0.99 confidence) → ITERATIVE 精化。为 GPnPSolver 提供高质量初值。

### 6.3 MonoPnPSolver (单目通用)

OpenCV EPnP RANSAC → ITERATIVE 精化，仅重投影约束，每帧独立无缓存。

**ITERATIVE 发散回退**：精化前保存 RANSAC EPnP 结果；若精化后 `|t|` 超出 `[10, 100000]` mm，自动回退到 RANSAC 初值并输出 `[MonoPnP] ITERATIVE 发散，回退到 RANSAC EPnP 结果`。此机制解决极小 ROI（~30px²）下深度估计病态导致的精化发散问题。

### 6.4 位姿有效性校验

所有求解器输出须通过:

1. `t.z > 0` — 相机在目标前方
2. `10 < |t| < 100000` mm — 深度合理（MonoPnP）；双目为 `[10, 20000]`
3. R, t 各分量有限 (无 NaN/Inf)

### 6.5 求解器对比

| 特性 | GPnPSolver | InitialPnPSolver | MonoPnPSolver |
|------|-----------|-----------------|---------------|
| 算法 | Eigen LM | OpenCV RANSAC+ITERATIVE | OpenCV EPnP+ITERATIVE |
| 约束 | 重投影+立体射线 | 重投影 | 重投影 |
| 参数空间 | 7维 [q,t]，warm-start | 无初值 | 无初值 |
| 帧间缓存 | ✅ 上帧位姿 | ❌ | ❌ |
| 适用模式 | 双目 | 双目首帧 | 单目全策略 |

### 6.6 ESKF 多源信息融合 (可选)

> 对应融合需求;默认关闭 (`eskf.enabled=false`),启用后**融合位姿为主输出**。

基于 `eskf/eskf_vio.hpp` (header-only 误差状态卡尔曼滤波器) 的适配层,将 **PnP 位姿 (相机观测) + IMU (预测) + 雷达高度 (Z 轴观测)** 融合输出平滑位姿:

```
数据流 (normal 模式, 以相机帧为节拍):
  相机帧 t_cam
    ├─ 合成/硬件 IMU 样本 (t ≤ t_cam) ──逐样本 predict 积分传播──▶ 状态
    ├─ 合成/硬件 雷达样本 (t ≤ t_cam) ──validate + update_altitude──▶ 状态
    └─ PnP 位姿 (成功时) ──update_camera_pose_hybrid (内置 FDI 拒坏帧)──▶ 状态
  输出: 融合 p/v/q (m, m/s, 四元数) → 终端每帧一行 + 日志
```

| 要点 | 说明 |
|------|------|
| 对齐方案 | **排干式 (drain-to-camera-time)**: 以相机帧为基准, 排干所有 `t ≤ t_cam` 的 IMU 样本逐样本积分 (保留高频信息), 雷达逐样本检验后更新 |
| 世界系 | 模板系, Z 轴向上 (重力 (0,0,-9.81), 雷达高度沿 Z); 可用 `R_template_world` 修正 |
| 单位 | PnP `t(mm)` → 内部 SI (m); 相机位姿 `p = -Rᵀ·t/1000`, `R_cam_w = Rᵀ` |
| 初始化 | 首个有效相机位姿 lazy init (置名义态, 不经过更新); 视觉丢失 > `max_cam_gap_s` 自动重置 |
| 视觉丢失 | YOLO miss / PnP 失败帧 → 状态由 IMU 继续传播 (ESKF 核心价值) |
| 数据源 | `input_system.imu/altimeter.type: "simulated"` 合成源 (离线链路验证); 硬件源 (Phase 2) 以同样样本流模式接入 |
| 运行 | `./build/Steretracker config/tracker_config_eskf.json` (示例配置) |
| 输出 | `[Frame N] ESKF p=[x,y,z]m v=[..] q=[w,x,y,z] | PnP t(mm)=[...]`; 汇总含 ESKF 统计 (FDI 忽略/姿态跳过/雷达接受率) |

适配层接口: `fusion::EskfFusionManager` (include/fusion/EskfFusionManager.hpp) — `feedImu` / `feedRadar` / `feedCameraPose` / `propagateTo`,输出 `position()/velocity()/rotation()/quaternion()/stats()`。

**延迟测量反向传播 (方案B 核心)**: 相机位姿是 t0 曝光时刻拍的、t1 才送达。`feedCameraPose(CameraObservation{t_exposure, t_arrival, ...})` 在延迟 > 1ms 时回退到 ≤ t0 的最近状态快照、重放 IMU 到 t0、在 t0 应用相机更新、再重放到当前;延迟超窗 (`backprop_window_s`) 时用协方差膨胀兜底 (`latency_fallback=inflate`) 或直接丢弃 (`reject`)。

**退化监控**: `getLatestState() → FusionState{position, velocity, quaternion, cov_trace, quality}`;`quality ∈ Normal/Degraded/Stale`,相机丢失后仍输出惯导结果并用协方差迹标记可信度。详见 CLAUDE.md §6.9。

> **线程化 (Phase 4, 已实现)**: `eskf.threaded: true` 时 `fusion.start()` 启动内部融合工作线程 — feedImu/feedRadar/feedCameraPose 异步入队, 融合线程独立以 IMU 节拍预测、相机/雷达异步更新, 相机缺席时状态继续传播; 输出经 `getLatestState()` 线程安全读取。默认 `false` 保持同步排干 (向后兼容)。

---

## 7. 可视化与输出

> 对应 SYS-REQ-500 系列

### 7.1 输出目录命名

| 模式 | 规则 | 示例 |
|------|------|------|
| Normal | `output/<输入源目录名>/` | `output/output_frames/` |
| Debug | `output/<图像文件名(去除' - '后缀)>/` | `output/im0/` |

若 `visualize=true`，自动创建目录。

### 7.2 Normal 模式

- **终端**: 每帧一行 `[Frame N] Strategy n=X r=[rx,ry,rz] t=[tx,ty,tz]`
- **可视化**: 仅三维坐标轴叠加图 `mono_f{N}.png`（使用实际帧号 N）
- **日志**: 若 `output.log_file: true`，输出 `tracking_log.txt`（包含配置摘要 + 完整处理过程，形如 debug 终端输出）

### 7.3 Debug 模式

- **终端**: 详细统计 (特征点/匹配/投影/视差中位数/GPNP状态/耗时)
- **可视化**: 各策略完整中间面板（文件名使用实际帧号）

| 策略 | 面板数 | 内容 |
|------|--------|------|
| AKAZE (双目) | 4 | 特征/立体/模板/坐标轴 |
| BinaryCorner (双目) | 5 | 二值/轴系/模板/立体/重投影 |
| TinyTarget (双目) | 标准 | solvePnP 输出 |
| Dual-ROI (双目) | 5 | 双 ROI 合并 |
| BinaryCorner (单目) | 6 | 二值/直立/轴系/模板/重投影 + overview |
| TinyTarget (单目) | 3 | 轴系/重投影 + overview |
| AKAZE (单目) | 3 | 轴系/匹配点 + overview |

---

## 8. 核心数据结构

定义于 `include/common/Types.hpp`：

| 结构体 | 用途 |
|--------|------|
| `StereoCameraParams` | 相机内外参 (K, R_rl, t_rl, focal_length, baseline) |
| `TrackerConfig` | 跟踪器配置 (面积阈值、LK参数、双ROI缩放等) |
| `PipelineResult` | 单帧输出: 特征/光流/投影/匹配/位姿/计时/ROI偏移 |
| `RoiRect` | ROI 矩形 (x, y, width, height) |
| `RoiGroup` | 双ROI组 (primary + secondary + is_dual 标志) |
| `PoseEstimate` | 位姿估计 (R, t, success, num_points) |

输入系统数据结构 (`include/input/`)：

| 结构体 | 用途 |
|--------|------|
| `InputSystemConfig` | 输入系统顶层配置 |
| `SensorPacket` | 统一帧数据包 |

---

## 9. 单目模式 (Mono Mode)

> 对应 SYS-REQ-130

通过 `mono_mode: true` 启用。`MonoTracker` 继承自 `TrackerBase`（与 `StereoTracker` 共享基类）。

```
TrackerBase
├── MonoTracker (mono_mode=true)
│   └── process() → mono_pnp_(MonoPnPSolver)
│
└── StereoTracker (默认)
    └── process() → gpnp_solver_ + initial_pnp_ + mad_filter_
```

| 维度 | 双目 | 单目 |
|------|------|------|
| 输入图像 | 左图 + 右图 | 仅左图 |
| 光流追踪 | ✅ LK L→R + FB | ❌ |
| 立体投影 | ✅ 视差→深度 | ❌ |
| MAD 滤波 | ✅ | ❌ |
| Pose 求解器 | GPnP + InitialPnP | MonoPnP (EPnP) |
| Warm-start | ✅ | ❌ |
| 五状态判定 | 相同 | 相同（共享 RoiGenerator） |

---

## 10. 配置文件

> 对应 SYS-REQ-600 系列

配置文件位于 `config/tracker_config.json`，关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mode` | `"normal"` | 运行模式: `"normal"` / `"debug"` |
| `mono_mode` | `false` | 启用单目模式 (单摄像头实时必须 true) |
| `input_system.max_frames` | 0 | 最大帧数, ≤0 = 无限 |
| `input_system.image.type` | `"file"` | 图像源: `"file"` / `"directory"` / `"sequence"` / `"camera"` |
| `input_system.image.camera_devices` | `"0"` | 摄像头设备: 索引 `"0"` 或路径 `/dev/video0` |
| `input_system.image.target_fps` | 0 | 实时目标帧率, 0 = 不限制 |
| `output.visualize` | `true` | 是否保存可视化图像 |
| `output.log_file` | `false` | Normal 模式下是否输出 TXT 日志文件 |
| `eskf.enabled` | `false` | 启用 ESKF 多源融合 (normal 模式; debug 自动禁用) |
| `eskf.noise.*` | — | ESKF 噪声参数 (sigma_acc/gyro/bias, cam_pos/rot_noise, radar_alt_noise...) |
| `eskf.gravity` | `[0,0,-9.81]` | 世界系重力向量 (世界系=模板系, Z 向上) |
| `eskf.R_imu_cam` | 单位阵 | IMU→相机安装外参旋转 |
| `eskf.p_imu_in_cam` | 零 | IMU 杆臂 (m) |
| `eskf.R_template_world` | 单位阵 | 模板系→世界系修正旋转 |
| `eskf.init_std.*` | — | 首个相机位姿 lazy init 的标准差 (P0) |
| `eskf.backprop_window_s` | `0.2` | 反向传播回退窗口 (s) |
| `eskf.state_hist_hz` | `100` | 状态快照记录频率 (Hz) |
| `eskf.latency_fallback` | `"inflate"` | 延迟超窗兜底: `"inflate"`(协方差膨胀) / `"reject"`(丢弃) |
| `eskf.max_output_age_s` | `0.5` | 相机更新间隔超限 → DEGRADED |
| `eskf.threaded` | `false` | 启用内部融合工作线程 (异步消费; 默认同步排干) |
| `input_system.imu.type` | `"custom"` | `"simulated"` = 合成 IMU 源 (rate_hz/sigma_acc/sigma_gyro/bias) |
| `input_system.altimeter.type` | `"can"` | `"simulated"` = 合成雷达源 (rate_hz/noise_m/inject_jump_every_s) |
| `strategies.tiny_max_area` | 800 | State 1/2 分界 (占位值) |
| `strategies.akaze_min_area` | 40001 | State 2/3 分界 (占位值) |
| `strategies.dual_trigger_area` | 490000 | State 3/4 分界 (占位值) |
| `strategies.close_range.enabled` | — | 启用 State 5 class1 回退 |
| `strategies.close_range.class1_min_area` | — | class 1 最小面积 |
| `strategies.close_range.akaze_min_area` | 0 | class1-only 专用 AKAZE 阈值 |
| `strategies.close_range.tiny_max_area` | 0 | class1-only 专用 TinyTarget 阈值 |

> ⚠️ 三个面积阈值为占位值，需在实际场景中标定后修正。

### 模式对照速查表

| | Normal 单目 | Normal 双目 | Debug 单目 | Debug 双目 |
|----|-----|-----|-----|-----|
| `mode` | normal | normal | debug | debug |
| `mono_mode` | true | false | true | false |
| `input_system` | **必填** | **必填** | 不需要 | 不需要 |
| `input` | 不需要 | 不需要 | **必填** | **必填** |
| ROI 来源 | YOLO | YOLO | 手动/YOLO | 手动/YOLO |
| 终端输出 | 简洁 | 简洁 | 详细 | 详细 |

---

## 11. 工厂函数

> 对应 SYS-REQ-630

| 函数 | 用途 | 校验 |
|------|------|------|
| `makeTrackerConfig(...)` | 构造跟踪器配置 | `0<scale≤1`, `gpnp≥3`, 面积阈值>0 |
| `makeYoloConfig(...)` | 构造 YOLO 配置 | 路径非空, conf∈(0,1] |
| `makeStereoCameraParams(K,R,t)` | 构造相机参数 | K标准内参, R det=1.0 |

---

## 12. 构建与运行

> 对应 SYS-REQ-700

### 依赖

| 库 | 用途 |
|----|------|
| OpenCV 4.x | AKAZE、光流、PnP、图像处理 |
| Eigen 3.x | 线性代数、GPNP LM 优化 |
| ONNX Runtime | YOLO 模型推理 |

### 编译

```bash
git clone https://github.com/Chihaya-anon343/NEW_Steretracker.git
cd NEW_Steretracker
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 运行

```bash
./build/Steretracker                    # 默认配置
./build/Steretracker path/to/config.json # 指定配置

# 实时摄像头 (单目 webcam, 见 §3.5)
./build/Steretracker config/tracker_config_webcam.json
# 先验证摄像头可用性
python scripts/camera_capture.py --preview

# ESKF 多源融合 (合成 IMU/雷达, 见 §6.6)
./build/Steretracker config/tracker_config_eskf.json
```

---

## 13. 项目目录结构

```
Steretracker/
├── README.md
├── CMakeLists.txt
├── main.cpp                    # 程序入口 (YOLO ROI + InputProvider + 模式分发)
├── best.onnx                   # YOLO ONNX 模型
│
├── config/
│   ├── tracker_config.json
│   └── tracker_config_webcam.json   # 实时摄像头配置 (type=camera + mono_mode)
│
├── data/                       # 测试图像与模板
│
├── include/
│   ├── common/                 # Types.hpp, Config.hpp, GeometryUtils.hpp
│   ├── detection/              # YoloDetector, YoloRoiProvider, RoiGenerator
│   ├── feature/                # FeatureExtractor, AkazeGpnp, BinaryCorner, TinyTarget
│   ├── fusion/                 # EskfFusionManager (ESKF 多源融合适配层)
│   ├── input/                  # InputProvider, IStereoImageSource, CameraSource, RingBuffer, SimulatedSensors
│   ├── matching/               # TemplateMatcher
│   ├── pose/                   # InitialPnPSolver, GPnPSolver, MonoPnPSolver
│   ├── stereo/                 # StereoProjector
│   ├── tracker/                # TrackerBase, MonoTracker, StereoTracker
│   ├── visualization/          # Visualizer
│   └── utils/                  # PoseUtils
│
├── eskf/                       # ESKF 多源融合库 (header-only, 第三方)
│   └── eskf_vio.hpp            # ESKF_VIO + GravityEstimator + RadarAltimeter
│
├── src/                        # 对应 .cpp 实现
├── scripts/camera_capture.py   # 摄像头预览/抓拍辅助脚本
├── sysml/                      # SysML 需求模型
│   ├── sysrequire.puml         # 系统需求规格 (SYS-REQ)
│   ├── softwarerequire.puml
│   └── flow.puml
└── extracted_input_system/     # 独立输入系统模块 (CanSocket/TimeSyncUnit, Phase 2 参考)