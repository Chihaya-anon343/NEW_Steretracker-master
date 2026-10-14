# 普通 BC 双目前端性能对比

2026-10-12 在项目 Docker CPU 环境实测，当前人工图对上的光流前端总耗时比原路径减少约 **28.3%**，前端处理能力约为原路径的 **1.39 倍**。收益来自减少一次 YOLO；当前左 BC + 全图光流的特征与匹配阶段比双侧 BC 更慢。

## 条件与范围

- 输入：当前配置中的 `data/big/cj01_image_0032.jpg`（左）和 `data/big/cj01_image_0032 - 副本.jpg`（右），1699×1744，约 +45 px 人工视差。
- Docker `st_copy_build`；OpenCV 4.5.4；CPU ONNX Runtime；模型 `yolov8n_p2.onnx`；可见硬件并发数 12，OpenCV 默认线程数 12、YOLO intra-op 线程数 4。
- GCC 11，独立 Release 构建，`-O3 -DNDEBUG`。模型和模板预加载，图像常驻内存。
- 原路径调用现有 `YoloRoiProvider::detect`，保留左右 YOLO 并行执行，然后左右独立 BC 提取并配对。
- 新路径调用现有左图检测、BC 单目提取与 StereoFlowMatcher，二维 LK 使用同坐标初值，保留 FB/NCC/极线/视差/去重/覆盖检查及按需粗搜索，不用 PnP 初值。
- 两条路径都关闭 PnP、可视化和文件输出。计时包含 YOLO、灰度转换、ROI 裁剪与特征/匹配；不包含模型加载、磁盘读写、融合与完整 Tracker 状态管理。
- 两个独立测量批次，每批每条路径先预热 10 次，再测量 100 次；每轮交替先后顺序。以下合并 run2、run3，各路径 200 个样本。run1 为初步测量，其旧路径成功统计误用了位姿成功标志，未纳入最终统计；修正仅涉及基准的成功统计，未改动算法与计时范围。

## 合并结果

| 阶段 | 原路径均值 | 光流路径均值 | 差值 |
|---|---:|---:|---:|
| YOLO | 185.94 ms | 118.82 ms | −67.11 ms |
| 灰度、裁剪、BC 与匹配 | 15.25 ms | 25.48 ms | +10.23 ms |
| 总前端耗时 | **201.19 ms** | **144.31 ms** | **−56.88 ms** |

| 总前端统计 | 原路径 | 光流路径 |
|---|---:|---:|
| 中位数 | 187.25 ms | 136.96 ms |
| P95 | 287.96 ms | 211.22 ms |
| 均值对应串行前端处理能力 | 4.97 帧/秒 | 6.93 帧/秒 |

两批总耗时均值分别为 215.61→154.08 ms、186.77→134.54 ms，降幅分别为 28.54%、27.97%。机器负载带来绝对耗时变化，但这两批的相对收益接近。

原路径 200/200 次提取出足够的配对点，光流路径 200/200 次通过匹配检查，均为 10 对；视差中位数分别为 45.000 和约 45.002 px。每帧原路径左右各一次 YOLO，新路径仅左一次。成功条件按各路径自身实现判定，不表示两者在真实场景中精度相等。

## 结论与后续

对当前人工图对和 CPU 配置，新路径有明确的前端耗时收益，约 28.3% 的耗时下降对应约 39.4% 的串行前端吞吐增加。不能将这些数字直接当作完整系统 FPS、GPU 收益或真实序列的成功率。

光流采用全图金字塔，而原 BC 主要处理目标 ROI，这可能增加匹配阶段开销；本次未进一步分析内部子阶段，不能据此确定唯一瓶颈。后续可以在保持视差搜索范围的前提下评估右侧搜索区域裁剪，但应作为独立改动重新测量。

## 复现与原始结果

基准源码：`tests/benchmark_stereo_frontend.cpp`，目标 `benchmark_stereo_frontend`。

```bash
bash scripts/docker-toolchain.sh cmake -S . -B build-codex-bc-bench -DCMAKE_BUILD_TYPE=Release -DGPNP_BUILD_TESTS=ON
bash scripts/docker-toolchain.sh cmake --build build-codex-bc-bench --target benchmark_stereo_frontend -j2
bash scripts/docker-toolchain.sh ./build-codex-bc-bench/tests/benchmark_stereo_frontend config/tracker_config.json 100 output/bc_flow_benchmark/run
```

原始逐次测量及分批报告位于 `output/bc_flow_benchmark/run2/`、`run3/` 的 `samples.csv` 和 `report.md`。所有预热样本均未计入统计。
