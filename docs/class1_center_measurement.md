# class1 相对 class0 的中心偏移计算

日期：2026-10-20。

结果：`class1_center_in_class0_mm = [-3.1501, 27.3201]`，已填入主配置。
即按正向模板观察，class1 中心在 class0 中心左侧约 3.15 mm、下方约 27.32 mm。

## 数据与中心定义

- 图像：`data/clean_target2.png`，PNG 尺寸 753×756 px。
- 角点：`scripts/class0_points.txt`、`scripts/class1_points.txt`，各 10 个完整人工标注角点。
- 物理尺度：`strategies.board_template.template_real_width_mm = 400`、`template_real_height_mm = 400`。
- 中心定义：与 `include/common/TemplateGeometry.hpp` 的 `placeTemplateInClass0()` 一致，取每组完整角点的 XY 包围框中点；不使用角点均值、图像中心或 YOLO 检测框中心。

| 点集 | X 最小 / 最大 (px) | Y 最小 / 最大 (px) | 包围框中心 (px) |
|---|---|---|---|
| class0 | 89.26 / 654.70 | 94.13 / 660.57 | (371.980, 377.350) |
| class1 | 347.36 / 384.74 | 410.50 / 447.47 | (366.050, 428.985) |

像素中心差为 `C1 - C0 = (-5.930, 51.635) px`。

## 换算

采用当前 MonoTracker / StereoTracker 构造 Dual 模型时的映射：

```text
X_mm = x_px / image_width_px  * template_real_width_mm  - template_real_width_mm / 2
Y_mm = y_px / image_height_px * template_real_height_mm - template_real_height_mm / 2
```

两中心相减时平移项抵消：

```text
ΔX = (366.050 - 371.980) × 400 / 753 = -3.150066401 mm
ΔY = (428.985 - 377.350) × 400 / 756 = 27.320105820 mm
```

正方向为模板右方 +X、下方 +Y，因此配置直接填写上述差值，无须反号。
这里的 400 mm 对应完整模板图像范围，不是白色 class0 图案的包围框宽度。
依据同一换算，class0 标注跨度约 300.37×299.70 mm，class1 约 19.86×19.56 mm。

## 精度与适用范围

这是基于现有人工标注和配置物理尺度的图像估计，不是新的实物测量。
小数保留四位用于记录计算结果，不代表实测达到该精度。
未对正视图额外做透视或畸变校正，与当前板模板建模方式保持一致。

普通 BC 的独立缩放参数和 TT 的 295 / 17 mm 边长与板图模型存在尺寸差异；本次不改变它们。
本偏移修正中心位置，不能消除不同策略间的尺度差异。
若后续确认完整图像范围不对应 400×400 mm，应按实际尺度重新计算。

已复核 PNG 尺寸、标注数量、包围框中心、换算公式和更新后 JSON 的可解析性；本次仅更新配置与计算记录，未运行跟踪或新增位姿精度实验。
