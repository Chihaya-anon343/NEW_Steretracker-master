# stereo_crop 双目 BC 批量测试

2026-10-12 对当前 `data/stereo_crop` 的 **30 对图像全部完成测试**。当前 BC + ROI 光流在极线容差为 3 px 时全部通过；默认 1.5 px 容差不足以容纳这批数据约 1.774 px 的人工垂直平移。

## 数据与条件

- 当前 manifest 为 30 对，源目录 `data/上升image_00_间隔50`，并非此前的负视差版本；当前生成脚本和实测均为 **正视差 20–26 px**，本次没有交换左右输入。
- 右图另有固定 +1.774 px 垂直平移，photometric=false。水平误差按 manifest 的逐帧视差计算；垂直误差参考 manifest 的已舍入偏移值，图像仿射插值可能产生少量差异。
- 运行项目实际 GPNP 主程序、YOLO 模型及 BC 策略，保持原 BC 参数、时序策略和质量检查；每帧均提取 10 个左角点。
- 同步 Directory 输入，禁用线程采集，确保 30 对按顺序逐帧处理，没有 take-latest 跳帧。每次左 YOLO 共 30 次，右 YOLO 为 0 次。
- 保留仅视差模式，InitialPnP/GPnP 全部跳过，位姿解算耗时为 0。日志中“位姿成功 0”符合该模式语义，不表示视差匹配失败。
- 独立测试配置与输出存于 `output/stereo_crop_bc_test`；原图、生成脚本和主 JSON 配置均未改动。主程序仅补充正常模式逐帧匹配耗时、ROI 尺寸与回退日志。

## 测试组

| 组别 | 极线容差 | 金字塔 | 可视化 | 匹配成功 |
|---|---:|---|---|---:|
| default_1p5 | 1.5 px | ROI | 开 | **0/30** |
| tolerance_3 | 3 px | ROI | 开 | **30/30** |
| full_3 | 3 px | 全图 | 关 | **30/30** |
| roi_3_perf | 3 px | ROI | 关 | **30/30** |

第一组与第二组仅调整极线容差，说明默认拒绝来自与该人工数据不相容的容差设置。default_1p5 的所有帧均无有效输出，并重试全图；重试不能解决相同的阈值限制。

## 精度与配对结果

以 ROI、3 px 容差结果计：

| 指标 | 结果 |
|---|---:|
| 成功帧数 | **30/30** |
| 接受角点对总数 | **299** |
| 每帧接受点对 | 29 帧 10 对；第 5 帧 9 对 |
| 视差平均绝对误差 MAE | **0.00908 px** |
| 视差 RMSE | **0.00996 px** |
| 最大绝对视差误差 | **0.0363 px** |
| 平均观测垂直偏移 | 1.78186 px |
| 垂直偏移 MAE | 0.01087 px |
| 最大垂直偏移误差 | 0.110 px |
| 全图回退 | **0/30** |

第 5 帧过滤了原角点编号 6，ROI 与全图结果一致，其余 9 点继续通过数量和覆盖检查。第 20 帧、原角点 4 的水平误差最大（−0.0363 px）；同帧角点 8 的垂直误差最大（−0.110 px）。这些误差来自接受点，未以帧中位数掩盖单点误差。

全图与 ROI 在 **30/30 帧中接受完全相同的原角点编号**，均为 299 对。导出的右点坐标最大差异约 **0.001 px**，包括 CSV 数值舍入影响。两条路径精度指标接近，ROI 裁剪未造成接受点损失。

## 耗时观察

关闭可视化的两次单遍 Debug 集成运行：

| 匹配耗时 | 全图 | ROI |
|---|---:|---:|
| 平均 | 13.48 ms | 2.97 ms |
| 中位数 | 13.08 ms | 1.74 ms |

这批图中的目标逐渐变小，ROI 搜索区域也从约 1128×968 降至 696×304，处理像素减少明显。上述数值包含首次帧和运行负载波动；本次没有预热、多轮交替或 Release 基准，因此不能将它作为稳定加速比例。主程序实际 FPS 还包含 PNG 输入、CSV 输出以及启用可视化时的编码写盘，不等于纯算法吞吐。

## 使用与复现

对该人工数据，使用测试配置中的 `stereo_frontend.max_epipolar_error_px=3.0`。主配置仍保留当前默认值，未自动扩大真实双目容差；真实设备的设置应结合标定和校正误差确定。

```bash
bash scripts/docker-toolchain.sh ./build-codex-bc-flow/GPNP output/stereo_crop_bc_test/tolerance_3.json
```

该配置已使用 `mode=normal`、`image.type=directory`、`directory_path=data/stereo_crop`，左右前缀保持 left/right。其余三组可用对应的 JSON 路径重复运行。

分析脚本 `scripts/analyze_stereo_crop_bc_test.ps1` 从逐帧日志、CSV 与 manifest 快照重新计算指标，并验证每帧点数。输出：

- `output/stereo_crop_bc_test/summary.json`：汇总精度、耗时和回退统计。
- `frames.csv`：四组共 120 条逐帧记录。
- `points.csv`：逐点坐标、视差与误差。
- `full_vs_roi.csv`：30 帧的角点编号及右点差异核对。
- `tolerance_3/stereo_f1.png` 至 `stereo_f30.png`：整段序列的左右配对可视化。
- 各组 `tracking_log.txt` 和 `disparity_fN.csv`：主程序原始输出；`manifest_snapshot.json` 保存此次数据真值。

可在 PowerShell 运行 `./scripts/analyze_stereo_crop_bc_test.ps1` 重算统计。本次验证的是人工双目数据的 BC 提取与左右光流配对，不评价真实位姿精度或物理深度尺度。
