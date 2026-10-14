# Dual-ROI 光流接入与验证结果

2026-10-12 已完成接入。Dual-ROI 正常路径使用一次左 YOLO、两组左 BC 和右图光流；两组共享单帧金字塔，支持联合 GPnP、class1 BC 退化及仅视差输出。主配置与示例已开启 `dual_roi_flow`，使用方式和接口说明见 `docs/dual_roi_stereo_flow.md`。

## 回归测试

Docker Release 构建主程序通过。普通 BC 光流 **23/23**、Dual-ROI 专项 **13/13** 通过，共 **36 项**。

专项覆盖：板系模型与普通 BC 尺度隔离；class0＋class1 联合位姿；亚像素垂直偏移；遮挡后保留原角点及模板编号；class1 失败保留 class0；巨型 primary 短路；tier1-only 和缺少 class1 标注；模型数量不匹配；旧前端延迟回退只计一帧一条日志；仅视差 TT 无 PnP；TT 左目位姿须显式允许；右图只检测到 class1；原五参数 ROI 接口；金字塔复用及跨帧重置。

位姿测试使用独立生成的 400 mm 板、已知相机/60 mm 基线和 900 mm 深度。刻意将普通 BC 的 class0 尺度设为 13 mm/px，仍能通过板系联合位姿测试，说明新 Dual-ROI 主流程使用明确的板模型。测试检查深度与重投影的一致性，不作为真实设备的位姿精度报告。

接入时发现并修复：

- 旧 class1 链误用尚未求位姿的 `success` 标志，造成 BC/TT 提取后被误判失败。
- class1 BC 点数与模板不一致时可能按前缀对应，新增严格数量检查。
- 旧前端可能从无纹理右 ROI 得到矩形角点并返回优化位姿。新光流路径的旧前端回退增加图像、原右观测及重投影检查，被拒后恢复原缓存。

## 近距离图像端到端验证

输入左图 `data/big/img_1.png`（798×786），独立生成右图：水平平移 -80 px、垂直 +1.774 px，保持画布大小、双线性插值、边界复制。原图未修改。该数据只验证对应点，不用于物理位姿判断。

专用配置 `output/dual_roi_flow_experiment/real_flow.json`：debug、仅视差、纯光流，Dual 触发阈值 1000；关闭 primary 短路以检查两组点。主配置的 200 万阈值和 0.98 短路比例未改变。

| 项目 | 结果 |
|---|---:|
| debug 两次处理 | 2/2 通过 |
| class0 合格点 | 每次 10 |
| class1 合格点 | 每次 10 |
| 单帧新建左右金字塔 | 1 对 |
| YOLO 次数 | 左 2、右 0 |
| PnP 耗时 | 0 |
| 第一帧导出点视差 MAE | 0.014075 px |
| 第一帧最大绝对视差误差 | 0.0384 px |

`real_flow_short.json` 保留 0.98 primary 短路比例：class0 跳过、class1 每次 10 对，2/2 通过，ROI 搜索 552×328，无全图重试，左 YOLO 2、右 0，PnP 跳过。

检查了两组对应图、逐点状态、左目 TT 回退的特征/重投影图，以及合成位姿重投影图。失败和仅视差模式不画位姿投影；TT 左目回退明确标识 `LeftPnpFallback`，不画未验证的右匹配。

## 前端性能对照

使用同一对近距离人工图，CPU YOLO、OpenCV 4.5.4（12 线程）、Release。每条路径预热 10 次，然后各测量 30 次，交替先后顺序。图像常驻内存，结果文件在计时后写入，关闭可视化。

比较内容为两组 ROI 的前端：旧路径运行左右 YOLO、两组左右 BC；新路径运行左 YOLO、两组左 BC、共享金字塔及光流。测试工具固定处理两个 ROI，不包含 primary 短路、PnP、P0 标注关联、运动门控、融合或磁盘解码。由此评估前端开销，不能解释成完整系统位姿处理的加速比例。

| 指标 | 双侧 YOLO＋BC | 左 YOLO＋BC＋光流 |
|---|---:|---:|
| YOLO 平均耗时 | 170.300 ms | 112.312 ms |
| 灰度/特征/匹配平均耗时 | 14.512 ms | 18.582 ms |
| 前端平均耗时 | **184.811 ms** | **130.895 ms** |
| 前端 P95 | 211.291 ms | 185.684 ms |
| 足够配对点/通过匹配检查的运行次数 | 30/30 | 30/30 |
| 每次保留点 | 20 | 20 |
| 保留点视差 MAE | 0.1000 px | **0.01409 px** |
| 最大绝对视差误差 | 1.0000 px | **0.03839 px** |

前端平均耗时减少 **29.17%**。收益主要来自省去右 YOLO；灰度/特征/匹配阶段反而增加约 **28.05%**（约 4.07 ms），不能把前端收益全部归因于光流本身更快。

测量阶段合计左 YOLO 60 次、右 30 次，分别对应两条路径各 30 次左推理、旧路径额外 30 次右推理。新路径总共新建 30 对金字塔、无全图重试。旧路径的成功口径是提取足够配对点，新路径还经过 FB、NCC、极线及帧级质量检查；两者不等同于位姿成功率。

单张人工图对的微基准不代表真实连续双目、GPU 环境或完整 FPS。真实遮挡、光度差异、大视差及物理位姿精度仍需标定双目数据验证。

## 原有 60 对普通 BC 数据回归

使用主配置的目录输入、200 万 Dual 阈值和仅视差模式，关闭可视化重新处理 `data/stereo_crop`：

- **60/60** 帧通过，保留 **598** 对点。
- 与上次极线带优化版本的 **60 个 CSV 逐字节相同**。
- 视差 MAE **0.009003 px**，最大绝对误差 **0.0363 px**。
- 左 YOLO 60 次、右 0 次，PnP 耗时为零。

## 配置与产物

主配置的 `dual_roi_flow=true` 不改变面积分级。当前大部分 `stereo_crop` 图仍走普通 BC；检查 Dual 可运行以下独立配置：

```bash
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/GPNP output/dual_roi_flow_experiment/real_flow.json
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/GPNP output/dual_roi_flow_experiment/real_flow_short.json
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/benchmark_dual_roi_frontend output/dual_roi_flow_experiment/real_flow.json 30 output/dual_roi_flow_experiment/frontend
```

若右人工图尚未生成，基准命令末尾加 `prepare`，将在该配置指定的实验输出路径生成右图。当前已保留生成结果；不会改动 `data/big` 或 `data/stereo_crop`。

`output/dual_roi_flow_experiment` 包含：

- `integration/`、`integration_short/`：端到端 CSV、两区域可视化和状态图。
- `synthetic/`：板系联合位姿、仅视差和 TT 左目回退的示例图。
- `frontend/samples.csv`、`points.csv`、`summary.json`：性能与逐点误差。
- `ordinary_regression/`、`ordinary_comparison.csv`、`verification.json`：原 60 对图回归结果。
- `GPNP_legacy`、`GPNP_flow`、`config_before/after.json`、`provenance.json`：修改前后程序与配置、输入及程序哈希。

算法核心在 `src/tracker/StereoTrackerDualFlow.cpp`；共用金字塔在 `StereoFlowFrame`，由 `src/matching/StereoFlowMatcher.cpp` 实现。未引入跨帧视差缓存。
