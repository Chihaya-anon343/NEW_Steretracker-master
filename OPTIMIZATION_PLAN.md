# Steretracker 优化方案（2026-08-06）

> **说明**：本文档基于对 README.md、CLAUDE.md 及关键源码（main.cpp、YoloRoiProvider、InputProvider、CameraSource、StereoTracker、YoloDetector）的逐行核查编写。
> 文档中的所有"现状"均以**代码事实**为准，README/CLAUDE.md 中的过时描述已在 §4 列出修正项。

---

## 1. 现状梳理

### 1.1 整体流程

```
配置解析 (mode / mono_mode)
  → 输入源: InputProvider(File/Dir/Seq/Camera) | Debug: cv::imread
  → 逐帧循环:
      YOLO 检测 (左右各一次推理, 串行) → 五状态分级 (面积阈值)
        → State 1 TT / 2 BC / 3 AKAZE / 4 Dual-ROI / 5 class1回退 / 无检测 skip
      → 特征提取 → 模板匹配 (AKAZE三阶段 / BC IoU / TT 超分)
      → PnP: 单目 MonoPnP(EPnP, 无缓存) | 双目 InitialPnP→GPnP(warm-start)
      → 退化链: AKAZE→BC→TT (失败时串行降级)
  → 输出: 可视化 + 日志 (同步落盘)
```

### 1.2 线程模型（代码证据）

| 线程 | 位置 | 职责 | 同步机制 |
|------|------|------|---------|
| 采集线程 ×1 | InputProvider.cpp:144-168 `captureLoop()` | 图像源读帧 → RingBuffer take-latest | mutex + condition_variable + atomic；Camera 自动启用，其他源 `use_threaded_capture` 显式启用 |
| 主线程 ×1 | main.cpp:359 `processFrame()` | YOLO(左右两次推理, std::async 并行) → tracker.process → 可视化/日志，全程同步 | 无（被采集队列节流） |
| Dual-ROI 临时线程 ×2 | StereoTracker.cpp:1053/1057、MonoTracker.cpp:189/192 | BC 提取 ‖ AK 提取（**项目唯一应用级并行点**） | `std::async` + future，每帧临时创建 |
| 库内部线程 | YoloDetector.hpp:117 `SetIntraOpNumThreads`(intra_op_threads=4)、OpenCV 内部 | 单次 YOLO 推理内部、图像算法内部 | 不归应用管理 |

**结论**：现状为"1 采集线程 + 1 处理线程"的两级结构；处理侧（YOLO→提取→PnP）纯串行。

### 1.3 讨论修正记录（认知纠偏）

| # | 此前的认知 | 代码核查结果 | 证据 |
|---|-----------|-------------|------|
| 1 | "双目只对左图推理，左右共享检测结果"（README §4.1 描述） | **左右各做一次完整 YOLO 推理，串行执行** | YoloRoiProvider.cpp:47-49 两次 `detector_->detect()` |
| 2 | "左右图检测是并行的" | **从未并行**：YoloRoiProvider 全历史无 `std::async`（`git log -S` 为空）；README 说法为过时文档 | git 历史 + YoloRoiProvider.cpp |
| 3 | "项目有并行先例" | 并行先例是 **Dual-ROI 的 BC‖AK 特征提取**（std::async），非 YOLO 检测 | StereoTracker.cpp:1053/1057、MonoTracker.cpp:189/192（commit 1aa6b69 引入） |
| 4 | "右图特征点靠光流" | 正确——特征提取阶段 AKAZE 左图关键点 → OpticalFlowTracker LK L→R 找右图对应 + 立体投影验证；与 YOLO 阶段是**两个独立阶段** | StereoTracker extract() 流水线 |
| 5 | "三级流水是下一步" | 有结构性冲突（GPnP warm-start 依赖上帧位姿、take-latest 丢帧语义），**建议先测量 + 低成本并行**，见 §2 中优先级 | 见 §2.2-8 |

---

## 2. 优化方案表

### 2.1 高优先级 — 实时性/正确性直接收益

| # | 优化项 | 现状（代码证据） | 方案 | 收益 | 风险/工作量 |
|---|--------|-----------------|------|------|-----------|
| H1 | **左右 YOLO 推理并行** | YoloRoiProvider.cpp:47-49 串行两次 detect() | ✅ **已完成 (2026-08-06)**：`std::async` 并行左右推理 + 异常回退串行（YoloRoiProvider.cpp:47-62） | YOLO 耗时近乎减半（CPU 下受 intra_op 争抢影响略小于 2×） | 低。已确认 YoloDetector::detect() 线程安全（共享成员只读 + session Run 并发安全） |
| H2 | **YOLO 检测降频 + ROI 预测** | 每帧全图推理 ×2；采集已线程化但处理侧仍逐帧全跑 | 每 N 帧检测一次，中间帧用上帧位姿/光流预测 ROI（无人机场景目标帧间位移有界） | 实时帧率显著提升，是 webcam 模式最直接收益 | 中。需 ROI 预测失效回退逻辑（预测失败→下帧补检测） |
| H3 | **单目 warm-start + 帧间追踪** | MonoPnPSolver 每帧独立无缓存（§6.3）；单目无光流/投影/MAD，退化面窄 | 帧间 LK 追踪关键点 + 上帧位姿注入 MonoPnP 初值；或 R,t 时域平滑（EMA/Kalman） | 消除单目位姿帧间抖动——webcam 实时模式是唯一已实现实时路径，收益直接 | 中。改动集中在 Tracker 层，可单测 |
| H4 | **面积阈值标定** | tiny_max_area/akaze_min_area/dual_trigger_area 为占位值（README 自标注 ⚠️）；webcam 配置 tiny_max_area=400 与文档默认 800 不一致 | 录制带真值数据集 → 标定工具/脚本自动搜索最优阈值 | 五状态分级是策略系统地基，阈值错则状态切换错乱级联下游 | 中。需要数据集，属流程而非纯代码 |
| H5 | **单侧检测丢帧行为** | main.cpp:413 `stereo_mono_fallback` 默认 false → 右图被遮挡/漏检时整帧放弃 | 评估默认开启；或左→右 ROI 几何推导（基线+深度范围）作回退 | 遮挡场景成功率提升 | 低 |

### 2.2 中优先级 — 稳定性/吞吐

| # | 优化项 | 现状 | 方案 | 收益 | 风险/工作量 |
|---|--------|------|------|------|-----------|
| M1 | **策略切换防抖 (hysteresis)** | RoiGenerator 每帧按面积直接判状态，阈值边界抖动导致 TT↔BC/BC↔AKAZE 频繁切换 | 切换需跨阈值后保持 N 帧；或上下行用不同阈值 | 位姿输出稳定，避免策略来回横跳 | 低，可单测（test_roi_generator 已有五状态用例基础） |
| M2 | **退化链时间一致性先验** | 每帧从主策略开始探测，失败才降级，最坏 3 倍耗时 | 上帧成功策略优先直接尝试，降级探测延后 | 避免"已知会失败"的主策略白跑 | 中。需处理策略状态缓存失效 |
| M3 | **YOLO 提前一拍 (double-buffer)** | 处理侧串行：YOLO → 提取 → PnP | 处理帧 N 提取/PnP 时，std::async 预跑帧 N+1 的 YOLO（一个 pending 槽位） | 等效"准三级流水"主干效果，帧率提升 | 中。关键点：ROI 必须与帧严格配对，不得错位 |
| M4 | **可视化/日志异步落盘** | 每帧 imwrite 同步 IO；日志本已一次性批量落盘（main.cpp 末尾），无需改 | ✅ **已完成 (2026-08-06)**：AsyncImageSaver 后台线程写盘（include/utils/AsyncImageSaver.hpp，header-only 无 CMake 改动），32 处 imwrite 调用点替换，帧循环结束 flush | 低帧率场景占比可观 | 低。深拷贝入队保证线程安全，拷贝成本远低于 PNG 编码 |
| M5 | **README §4.1 文档修正** | §4.1 写"对左图推理，左右 ROI 共享检测结果"，与代码矛盾 | 改为"左右各推理一次，生成独立 ROI 后立体配对" | 消除文档误导（本方案的认知偏差源头） | 零 |

### 2.3 低优先级 — 工程化/可测性

| # | 优化项 | 现状 | 方案 | 收益 | 风险/工作量 |
|---|--------|------|------|------|-----------|
| L1 | **CameraSource 可测化改造** | CameraSource.cpp:40 `cap_` 在 open() 内硬编码 `make_unique<cv::VideoCapture>()`，无注入点 → 帧语义（clone/时间戳/限速）无法脱离硬件测试 | open() 增加 capture 工厂参数（默认真实实现），测试注入假 VideoCapture | 帧语义测试可在 CI 确定性运行 | 中 |
| L2 | **CameraSource 自动化测试** | test_input_system.cpp 14 用例覆盖 File/Dir/Seq + 线程化采集，Camera 零用例 | 第一层（确定性，无需硬件）：open 失败路径、未 open 时 nextFrame、shutdown 安全；第二层（可选 SKIP）：真实设备帧语义 | 补全唯一无测试的输入源 | 低（第一层）/ 硬件（第二层） |
| L3 | **配置-文档漂移对齐** | CLAUDE.md §8.2 示例字段与真实 tracker_config.json 不一致（real_w vs template_real_width_mm、pixel_to_meter_scale vs _class0/1、template_path vs template_dir）；两份 config 重复维护 | 文档示例引用真实配置；或引入默认配置合并机制 | 减少维护漂移，防止示例误导 | 低 |
| L4 | **代码注释修正** | InputConfig.hpp:23 `Camera ///< 实时摄像头（未来）`，实际 Phase 3 已实现 | 改为"已实现 (Phase 3)" | 消除过时注释 | 零 |
| L5 | **tests/README 计数修正** | §4.5 标题"13 场景"与表格 14 行不符 | 统一为 14 | 文档一致性 | 零 |
| L6 | **Phase 2 传感器接入** | SensorPacket.imu/height 恒空，IMU/高度计占位 | 按无人机需求优先级决定是否推进 | — | 大 |
| L7 | **USB 双目 / RTSP 支持** | §3.5 明确局限：仅单目 | 硬件方案确定后参考线程化采集模式扩展 | 双目实时场景 | 大 |

---

## 3. 明确不推荐 / 缓做的方案

| 方案 | 不推荐原因 |
|------|-----------|
| **完整三级流水（采集‖YOLO‖提取+PnP）** | ① GPnP warm-start 依赖"上帧位姿"，流水线打破处理顺序=帧序，需帧号对齐+结果重排，波及 Tracker 状态机；② take-latest 丢帧语义与多级管道交互复杂；③ 收益不确定——应先测量定位瓶颈（PipelineResult.total_time_ms + timing 字段已存在）。**M3（YOLO 提前一拍）是更优的中间方案** |
| **右图独立推理"补检测"**（替代并行） | 与 H1 冲突：双图推理已是现状，优化方向是并行化与降频，而非再加成本 |
| **真实摄像头设为必跑测试** | 无摄像头环境（CI/服务器）必红，引入 flaky；应沿用 fixtures 缺失 SKIP 的现有模式 |

---

## 4. 建议推进顺序

1. **零成本修正**：M5（README §4.1）、L4（InputConfig.hpp 注释）、L5（tests/README 计数）——✅ **已完成 (2026-08-06)**；
2. **快速见效（Tracker 层，可单测）**：M1 策略防抖、H3 单目 warm-start；
3. **处理侧并行**：H1 左右 YOLO 并行 ✅ → M3 YOLO 提前一拍（先补阶段计时日志测量验证瓶颈）；
4. **实时性深化**：H2 检测降频 + ROI 预测、H5 单侧丢帧回退；
5. **工程化**：L1/L2 CameraSource 可测化 + 测试、L3 配置文档对齐；
6. **数据驱动**：H4 阈值标定（需先建真值数据集）。

---

*生成日期: 2026-08-06 · 依据: README.md / CLAUDE.md / main.cpp / src/* 源码逐行核查*
