# SSH 远端 150 对双目视差实验分析

分析日期：2026-10-14。原始资料为 [remote_output](C:/D/Codes/NEW_Steretracker-master_COPY/remote_output) 中的 150 个 CSV 和一份 [tracking_log.txt](C:/D/Codes/NEW_Steretracker-master_COPY/remote_output/tracking_log.txt:866)。数据集与此前 150 对图相同，依据用户说明，以当前本地 [manifest.json](C:/D/Codes/NEW_Steretracker-master_COPY/data/stereo_crop/manifest.json) 作为逐帧视差与垂直偏移真值。

## 结论

单轮 150/150 帧通过，保留 1558/1574 点；视差 MAE=0.008538525 px、RMSE=0.010321618 px、P95=0.0175 px、最大绝对误差=0.0496 px。全部保留点的水平误差不超过 0.05 px。

匹配均值为 3.341725 ms，P95 为 7.134460 ms，程序汇总 FPS 为 2.848773。远端没有同机旧版对照和多轮结果，不能据此计算算法加速或重复稳定性。仅视差运行没有进行任何 PnP，位姿成功数 0 不表示求解失败。

## 平台与版本

| 项目 | 内容 |
|---|---|
| CPU／线程设置 | |
| GPU／推理执行设备 | |
| 操作系统／容器环境 | |
| 构建模式／编译器 | |
| OpenCV／ONNX Runtime | |
| 远端程序版本／SHA256 | |

按用户要求留空，后续填写。日志记录配置路径为 config/tracker_config.json，但没有当时的完整配置快照。已确认 normal、双目、directory、LK、仅视差、左 YOLO 150 次和右 YOLO 0 次；没有依据将执行设备指定为 CPU 或 CUDA。

## 校验与口径

- 数据帧从 1 到 150，逐帧 CSV、StereoFrontend 记录和汇总表无缺失、额外帧或重复帧。
- CSV 行数与日志 pairs/nMatch 相同，原左点编号唯一，成功标志、策略、区域点数与日志相符。
- 精度使用成功帧保留点，逐点加权；耗时使用单轮 150 个 match_ms。P95 使用 nearest-rank。
- 水平误差使用 disparity_px 减 manifest 真值；纵向误差使用导出坐标的 right_y-left_y 减 dy_px。左右坐标独立舍入，不能重新相减替代 disparity_px。
- 导入一轮结果，没有预热或重复实验记录。日志两处 YOLO 次数相同，只是同一计数的重复打印，未相加为 300 次。
- 真值 manifest 与本地优化实验保存快照逐字节一致。远端未附图片哈希，未将本地图片哈希冒充远端输入校验。
- 原始导入资料在分析前后哈希一致；[provenance.json](C:/D/Codes/NEW_Steretracker-master_COPY/output/remote_stereo_150_analysis/provenance.json) 记录分析时文件指纹，不能代表远端运行前的程序／配置指纹。

## 策略、精度与点保留

| 策略 | 帧通过 | 保留／原始点 | MAE px | RMSE px | P95 px | 最大误差 px |
|---|---:|---:|---:|---:|---:|---:|
| Dual-ROI | 23/23 | 446/460 | 0.007044843 | 0.008513139 | 0.0154 | 0.0203 |
| BC | 101/101 | 1008/1010 | 0.009550099 | 0.011307011 | 0.0193 | 0.0496 |
| TT | 26/26 | 104/104 | 0.005139635 | 0.006785363 | 0.01318 | 0.0176 |
| 全体 | 150/150 | 1558/1574 | 0.008538525 | 0.010321618 | 0.0175 | 0.0496 |

总体保留率 98.9835%；垂直误差 MAE=0.012685494 px，最大 0.152 px。策略序列为 Dual 1–9、BC 10、Dual 11–24、BC 25–124、TT 125–150。Dual 保留 class0 229 点、class1 217 点。不同策略对应不同输入尺度，组间误差和耗时不能直接充当同输入策略消融。

最大误差点为第 92 帧，源图 frame_000910.png，BC 原编号 4；真值 40 px，观测 39.9504 px，误差 -0.0496 px，纵向误差约 -0.132 px。

![远端实验总览](C:/D/Codes/NEW_Steretracker-master_COPY/docs/assets/algorithm_v1_4/remote_150_overview.png)

## 时间与过滤

| 策略 | 样本 | 匹配均值 ms | 中位数 ms | P95 ms | 最大 ms | 平面检查均值 ms |
|---|---:|---:|---:|---:|---:|---:|
| Dual-ROI | 23 | 6.290512 | 5.942120 | 8.985080 | 9.557620 | 0.182824 |
| BC | 101 | 3.077432 | 2.560540 | 6.171080 | 7.237710 | 0.104174 |
| TT | 26 | 1.759860 | 1.709605 | 2.157750 | 2.209380 | 0.001099 |
| 全体 | 150 | 3.341725 | 2.420545 | 7.134460 | 9.557620 | 0.098367 |

运行汇总：YOLO 平均 209.588361 ms，跟踪平均 9.910587 ms，特征提取字段平均 4.892834 ms，PnP 0 ms。特征提取字段包含匹配，是跟踪的细分项；平面时间也是匹配内部计时，不能叠加或解释为开关过滤的净增耗时。

FPS 对应的平均处理间隔约 351.028 ms，不是逐帧延迟 P95。YOLO 在 YOLO＋跟踪的记录时间中占 95.48%，未测得整个循环的逐项时间分解，不能把差额直接归因于 SSH。

23 帧 Dual 的两组均记录 disparity_plane，101 帧 BC 记录 disparity_plane；26 帧 TT 为 insufficient_redundancy。累计平面剔除 1 点，发生在第 29 帧。其原始点 10 个，保留 9 个，原编号 9 不在 CSV，最大保留误差 0.0130 px，帧仍通过。

共未保留 16 点：Dual 14 点、BC 2 点。其中只确认 1 点由平面过滤剔除，剩余 15 点没有逐点拒绝原因资料。帧 16、17、18、19、21、22、23、24、29、42 的保留点少于原始点。退化、全图重试、旧前端回退均为 0，金字塔对数 150，粗搜索尝试 141。

![远端保留点误差分布](C:/D/Codes/NEW_Steretracker-master_COPY/docs/assets/algorithm_v1_4/remote_error_distribution.png)

## 本地优化结果核对

与 output/solver_geometry_optimization/after/run1 比较，1558 个对应身份一致，无新增／缺失。1497 行全字段相同，61 行数值不同，类别、原编号、区域、模板、模型、策略及成功标志无差异。最大差异为左坐标 0 px、右 x 0.001 px、右 y 0.010 px、disparity_px 0.0002 px。未确定差异来源，不将跨平台比较称为远端重复稳定性。

本地原版第 29 帧原角点 9 曾有 -0.6371 px 误差；本次远端未保留它，日志确认平面剔除 1 点。被拒点的 FB、NCC 与平面残差没有远端记录，因此未移用本地诊断值。完整历史对照见 [优化报告](C:/D/Codes/NEW_Steretracker-master_COPY/docs/solver_geometry_optimization_validation.md)。

## 复现与产物

```text
python scripts/analyze_remote_stereo_results.py
```

默认使用 remote_output 与 data/stereo_crop/manifest.json，在 output/remote_stereo_150_analysis 写出 frame_metrics.csv、point_metrics.csv、strategy_summary.csv、summary.json 和 provenance.json。脚本使用 Python 标准库统计，ReportLab 绘图、PDFium 生成 Markdown 用 PNG，Windows 绘图字体为微软雅黑。图表同时保留 PNG、SVG 和 PDF。

本次资料可以复现离线统计。完整远端执行的重建仍需配置、程序、依赖与平台信息。当前数据是人工图像对，没有物理尺度和位姿真值；无失败／遮挡场景，不能由本轮推断失效恢复或真实 GPnP 准确度。
