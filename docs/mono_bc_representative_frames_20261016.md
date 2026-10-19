# 单目 BC：代表帧可视化与性能开销

本页只展示已经冻结的实验数据；可视化不重新解算位姿。文件名帧号从 0 开始，完整运行 CSV 的 frame 为文件名帧号＋1。

## 如何阅读

每张对比图上排：标注 ROI 下独立 BC 提取，青色圆圈为 GT，红色叉为观测角点，黄色线为位置误差。下排：原序列完整 YOLO＋历史状态＋PnP 的记录结果，坐标轴 X 红、Y 绿、Z 蓝，长度均为 100mm，白色虚线为 GT。轴超出裁剪范围时只显示可见部分。两排属于不同实验，不能把独立 BC 拒绝直接当作完整追踪失败。

选帧覆盖清晰、倾斜、同几何退化、旋转、大 ROI、超范围倾角、新增拒绝、残余深度跳变，并加入一例旧方法局部大误差。它们是解释用样例，不是随机样本，整体结论仍以全量实验为准。

![九帧总览](../output/mono_normalized_20261016/representatives/overview.png)

## 性能开销

已有独立交错计时：61 个共同成功 ROI、每组 732 次调用。原 BC 中位数/P95 为 4.016/56.006ms，新精化方案为 2.136/18.910ms。排除了图像读取、YOLO、PnP及可视化。

新增资源探针：8 个相同且共同成功的 ROI，每种方法单独启动进程，预热后重复 50 轮、每组 400 次。CPU 时间不含读取/预热；峰值 RSS 是整个进程的高水位，包含 OpenCV、读取缓存及预热，不能解读为算法独占内存。

| 方法 | CPU ms/次 | 测量区间 wall ms/次 | 单核利用率 | 整进程峰值 RSS（MiB） |
|---|---:|---:|---:|---:|
| current | 18.114 | 18.116 | 100.0% | 87.4 |
| normalized | 4.924 | 4.924 | 100.0% | 66.9 |
| refined | 5.421 | 5.421 | 100.0% | 67.1 |

单核利用率高表示该区间主要在计算；它不是整台电脑 CPU 占用百分比。资源探针的样本与全量计时样本不同，不能混合计算加速比；未测 GPU 利用率和完整系统 FPS。

![耗时、阶段及资源](../output/mono_normalized_20261016/representatives/performance.png)

新精化路径的阶段耗时（原独立计时的全部 816 次 class0 调用）：

| 阶段 | 中位数 ms | P95 ms |
|---|---:|---:|
| 粗归一化及四方向 | 0.284 | 1.184 |
| 标准空间角点拟合 | 0.158 | 0.333 |
| 原图边线精化 | 0.299 | 1.901 |
| H / 形状验证 | 0.422 | 2.501 |

阶段分位数不能直接相加；完整 BC 还包含 Otsu、连通域、填孔、形态学及结果封装。

## 帧索引

| 场景 / 文件帧号 | 倾角 | 角点 RMSE：旧→新（px） | BC 计时中位数：旧→新（ms） | 完整流程策略：旧→新 |
|---|---:|---:|---:|---|
| reference / 000045 | 7.3° | 8.05 → 0.45 | 3.31 → 2.15 | DualRoi_BCJ → DualRoi_BCJ |
| reference / 000055 | 5.7° | 14.37 → 0.61 | 未在原计时样本中 | DualRoi_BC → DualRoi_BCJ |
| oblique_clean / 000075 | 22.7° | 2.68 → 0.49 | 2.44 → 1.53 | DualRoi_BCJ → DualRoi_BCJ |
| oblique_degraded / 000075 | 22.7° | 6.21 → 0.41 | 2.49 → 1.59 | TinyTarget → TinyTarget |
| fast_rotation / 000135 | 27.7° | 1.40 → 0.66 | 21.98 → 6.07 | DualRoi_BC → DualRoi_BC |
| oblique_clean / 000165 | 41.5° | 2.31 → 1.56 | 68.55 → 19.55 | DualRoi_C1BC → DualRoi_C1BC |
| grazing / 000060 | 69.2° | 3.33 → 2.37 | 15.90 → 2.96 | BinaryCorner → BinaryCorner |
| reference / 000015 | 10.7° | 1.12 → 拒绝 | 1.16 → 0.57 | BinaryCorner → BinaryCorner |
| oblique_degraded / 000149 | 29.5° | 2.29 → 1.29 | 未在原计时样本中 | TinyTarget → TinyTarget |

## reference / frame_000045.png：常规清晰目标

[原始帧](../tests/data/mono_robustness_20261015/reference/frame_000045.png)；倾角 7.3°，class0 跨度 270px。

![常规清晰目标](../output/mono_normalized_20261016/representatives/cards/reference_f000045.png)

- current：DualRoi_BCJ；Z=1997.18mm，GT=1990.65mm；法向误差 1.90°；深度相对误差 0.33%。
- refined：DualRoi_BCJ；Z=1989.45mm，GT=1990.65mm；法向误差 1.89°；深度相对误差 0.06%。

## reference / frame_000055.png：旧方法局部大误差样例

[原始帧](../tests/data/mono_robustness_20261015/reference/frame_000055.png)；倾角 5.7°，class0 跨度 521px。

![旧方法局部大误差样例](../output/mono_normalized_20261016/representatives/cards/reference_f000055.png)

- current：DualRoi_BC；Z=1112.39mm，GT=1111.66mm；法向误差 1.02°；深度相对误差 0.07%。
- refined：DualRoi_BCJ；Z=1111.32mm，GT=1111.66mm；法向误差 0.45°；深度相对误差 0.03%。

## oblique_clean / frame_000075.png：应用范围内倾斜

[原始帧](../tests/data/mono_robustness_20261015/oblique_clean/frame_000075.png)；倾角 22.7°，class0 跨度 225px。

![应用范围内倾斜](../output/mono_normalized_20261016/representatives/cards/oblique_clean_f000075.png)

- current：DualRoi_BCJ；Z=2736.69mm，GT=2717.26mm；法向误差 4.79°；深度相对误差 0.72%。
- refined：DualRoi_BCJ；Z=2720.21mm，GT=2717.26mm；法向误差 4.02°；深度相对误差 0.11%。

## oblique_degraded / frame_000075.png：与上一帧同几何的退化图

[原始帧](../tests/data/mono_robustness_20261015/oblique_degraded/frame_000075.png)；倾角 22.7°，class0 跨度 225px。

![与上一帧同几何的退化图](../output/mono_normalized_20261016/representatives/cards/oblique_degraded_f000075.png)

- current：完整位姿失败。
- refined：完整位姿失败。

2026-10-19 复核：完整流程选中的 YOLO 框在背景上，触发 TT 四点，图中上排精确的 BC 十点没有进入该 PnP。改用标注 ROI 且保留 30°先验后，BC 十点 PnP 成功，重投影误差约 0.289px。详见 [失败原因核查](mono_pnp_failure_diagnosis_20261019.md)。

## fast_rotation / frame_000135.png：快速面内旋转

[原始帧](../tests/data/mono_robustness_20261015/fast_rotation/frame_000135.png)；倾角 27.7°，class0 跨度 594px。

![快速面内旋转](../output/mono_normalized_20261016/representatives/cards/fast_rotation_f000135.png)

- current：DualRoi_BC；Z=740.77mm，GT=738.23mm；法向误差 0.40°；深度相对误差 0.34%。
- refined：DualRoi_BC；Z=738.65mm，GT=738.23mm；法向误差 0.20°；深度相对误差 0.06%。

## oblique_clean / frame_000165.png：大ROI性能压力，超30度

[原始帧](../tests/data/mono_robustness_20261015/oblique_clean/frame_000165.png)；倾角 41.5°，class0 跨度 1276px。超过当前 PnP 的 30°先验，仅作压力样例。

![大ROI性能压力，超30度](../output/mono_normalized_20261016/representatives/cards/oblique_clean_f000165.png)

- current：DualRoi_C1BC；Z=494.14mm，GT=390.23mm；法向误差 11.54°；深度相对误差 26.63%。
- refined：DualRoi_C1BC；Z=494.14mm，GT=390.23mm；法向误差 11.54°；深度相对误差 26.63%。

## grazing / frame_000060.png：大倾角压力，超30度

[原始帧](../tests/data/mono_robustness_20261015/grazing/frame_000060.png)；倾角 69.2°，class0 跨度 228px。超过当前 PnP 的 30°先验，仅作压力样例。

![大倾角压力，超30度](../output/mono_normalized_20261016/representatives/cards/grazing_f000060.png)

- current：完整位姿失败。
- refined：完整位姿失败。

2026-10-19 对照确认：保持 30°先验时拒绝；仅在诊断中关闭先验后，BC 十点求解成功，估计倾角 69.16°。这是超工作范围的约束拒绝。详见 [失败原因核查](mono_pnp_failure_diagnosis_20261019.md)。

## reference / frame_000015.png：新BC拒绝边界样例

[原始帧](../tests/data/mono_robustness_20261015/reference/frame_000015.png)；倾角 10.7°，class0 跨度 50px。

![新BC拒绝边界样例](../output/mono_normalized_20261016/representatives/cards/reference_f000015.png)

- current：BinaryCorner；Z=11679.36mm，GT=11434.16mm；法向误差 5.95°；深度相对误差 2.14%。
- refined：BinaryCorner；Z=11482.46mm，GT=11434.16mm；法向误差 5.78°；深度相对误差 0.42%。

## oblique_degraded / frame_000149.png：残余深度跳变样例

[原始帧](../tests/data/mono_robustness_20261015/oblique_degraded/frame_000149.png)；倾角 29.5°，class0 跨度 828px。

![残余深度跳变样例](../output/mono_normalized_20261016/representatives/cards/oblique_degraded_f000149.png)

- current：TinyTarget；Z=23079.94mm，GT=551.75mm；法向误差 30.16°；深度相对误差 4083.02%。
- refined：TinyTarget；Z=23079.94mm，GT=551.75mm；法向误差 30.16°；深度相对误差 4083.02%。

这是保留的失败案例：完整流程退化到 TT 后仍输出严重错误的深度。新 BC 没有解决该后备路径问题，不能据其他样例宣称所有深度跳变已消除。

## C++ 实际归一化阶段

下面逐行对应上述九帧，三列依次为前景、四方向粗归一化、H 验证后的归一化结果。拒绝帧缺少后续图像时明确留空。所有图均由实际 C++ 调试输出产生，并核对了重放角点与原冻结 CSV 完全一致。

![归一化阶段](../output/mono_normalized_20261016/representatives/normalization_stages.png)

完整统计与配置见 [主实验报告](mono_bc_normalized_20261016.md)。
