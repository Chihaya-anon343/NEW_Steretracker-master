"""Inside Docker: build an experiment against freshly built tracker objects."""
from pathlib import Path
import shlex
import subprocess
import csv
import hashlib
import json
import math

root = Path(__file__).resolve().parents[1]
build = root / 'build-codex-bc-bench/tests'
output = root / 'output/class_center_alignment_20261020'
output.mkdir(parents=True, exist_ok=True)
subprocess.run(['cmake', '--build', str(root / 'build-codex-bc-bench'), '--target', 'test_dual_roi_flow', '-j', '4'], check=True)
flags = (build / 'CMakeFiles/test_dual_roi_flow.dir/flags.make').read_text()
options = []
for line in flags.splitlines():
    if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
        options.extend(shlex.split(line.split('=', 1)[1]))
obj = output / 'experiment.o'
subprocess.run(['/usr/bin/c++', *options, '-c', str(root / 'scripts/experiment_class_centers.cpp'), '-o', str(obj)], check=True)
link = shlex.split((build / 'CMakeFiles/test_dual_roi_flow.dir/link.txt').read_text())
link = [str(obj) if x.endswith('/unit/test_dual_roi_flow.cpp.o') else x for x in link]
link[link.index('-o') + 1] = str(output / 'experiment')
subprocess.run(link, cwd=build, check=True)
subprocess.run([str(output / 'experiment')], cwd=root, check=True)

rows = list(csv.DictReader((output / 'poses.csv').open()))
actual = [r for r in rows if r['stage'] == 'tracker']
control = [r for r in rows if r['stage'] == 'ideal_control']
assert all(r['success'] == '1' and r['model_frame'] == 'class0' for r in rows)
def translation(row):
    return [float(row[k]) for k in ('tx_mm', 'ty_mm', 'tz_mm')]
delta = [b-a for a, b in zip(translation(actual[0]), translation(actual[1]))]
gap = math.sqrt(sum(d*d for d in delta))
pixel_gap = math.hypot(*(float(actual[1][k])-float(actual[0][k]) for k in ('origin_u_px', 'origin_v_px')))
dot = sum(float(actual[0][f'r{i}{j}'])*float(actual[1][f'r{i}{j}']) for i in range(3) for j in range(3))
angle = math.degrees(math.acos(max(-1, min(1, (dot-1)/2))))
control_gap = math.dist(translation(control[0]), translation(control[1]))
spans = []
for cls in (0, 1):
    points = list(csv.DictReader((output / f'class{cls}/observations.csv').open()))
    spans.append([max(float(p[k]) for p in points)-min(float(p[k]) for p in points) for k in ('x_mm', 'y_mm')])
cfg = json.loads((root / 'config/tracker_config.json').read_text())
(output / 'config_snapshot.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n')
files = ['config/tracker_config.json', 'data/clean_target2.png', 'scripts/class0_points.txt', 'scripts/class1_points.txt',
         'scripts/experiment_class_centers.cpp', 'scripts/run_class_center_experiment.py',
         'src/tracker/MonoTracker.cpp', 'src/pose/MonoPnPSolver.cpp', 'include/common/TemplateGeometry.hpp']
files += [str(p.relative_to(root)) for p in (root / cfg['strategies']['binary_corner']['template_dir']).iterdir() if p.is_file()]
(output / 'provenance.json').write_text(json.dumps({
    'sha256': {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
    'protocol': 'same source PNG, same configured K, manual class-specific ROI; independent MonoTracker cold starts; temporal disabled; no YOLO or joint solve',
    'offset_mm': cfg['template_geometry']['class1_center_in_class0_mm'],
    'delta_t_mm': delta, 'distance_mm': gap, 'pixel_gap': pixel_gap,
    'rotation_difference_deg': angle, 'ideal_control_gap_mm': control_gap,
    'model_spans_mm': spans,
}, ensure_ascii=False, indent=2)+'\n')
table = '\n'.join(f"| class{r['class_id']} | {r['strategy']} | {r['n_points']} | {float(r['tx_mm']):.3f} | {float(r['ty_mm']):.3f} | {float(r['tz_mm']):.3f} | {float(r['mean_reprojection_px']):.3f} |" for r in actual)
(output / 'report.md').write_text(f'''# 同一靶标 class0 / class1 原点一致性实验

日期：2026-10-20。结论：当前配置下，两组均成功，但最终三维原点不重合。

## 两组实验

两组使用完全相同的 `data/clean_target2.png`（753×756），相机内参取主配置。
组一只提供 class0 ROI，组二只提供 class1 ROI（`is_class1=true`）。ROI 从对应人工角点包围框外扩，class0 外扩10px，class1外扩5px；运行时角点由真实提取器计算。
每组重新创建 MonoTracker，关闭 temporal，无缓存或 seed；不运行 YOLO，不混合两类角点。
两组都使用偏移 `[-3.1501, 27.3201] mm`，输出系均为 class0。
BC 参数、角点身份验证配置、TT 后备参数、TT 验证和倾角先验取主配置；BC 成功，因此两组均未进入 TT。

| 组别 | 策略 | 对应点数 | tX (mm) | tY (mm) | tZ (mm) | 全部对应点平均重投影误差 (px) |
|---|---|---:|---:|---:|---:|---:|
{table}

- class1 − class0：`[{delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f}] mm`。
- 三维原点间距：**{gap:.3f} mm**。
- 图像中原点距离：**{pixel_gap:.3f} px**。
- 旋转差：**{angle:.3f}°**。

位姿满足 `P_camera = R * P_class0 + t`；t 即 class0 原点在相机坐标中的位置。
不设人为验收阈值；上述差异足以说明本次输出不一致。图上看似接近不能证明三维位置一致。

## 可视化

![两组独立解算](comparison.png)

![同图叠加，class0 实线、class1 虚线](overlay.png)

![相机坐标中的原点和三维轴投影](camera_space.png)

红色 +X、绿色 +Y、蓝色朝上法向 -Z；所有轴长度55mm。黄色圆圈是输出原点。
相机坐标图为 X-Z、Y-Z 正交投影，两面板均为1.6像素/mm，可看到明显深度差。
各组目录保留跟踪器原生中间结果和实际二维/三维对应点 `observations.csv`。

## 解释与边界

实际 BC 模型跨度分别为 class0 `{spans[0][0]:.3f}×{spans[0][1]:.3f} mm`，class1 `{spans[1][0]:.3f}×{spans[1][1]:.3f} mm`。
400×400mm 板图映射下标注跨度则分别约为300.37×299.70mm、19.86×19.56mm。
现有模型尤其 class1 尺寸与板图尺度不同，按物体大小推断的深度会受影响；class1只有约37px跨度，角点像素误差也更显著。
因此，中心偏移统一原点定义不等于两组图像解算必然给出相同位姿；本次未改尺寸或算法来促使结果重合。
这里只能指出这些影响因素，不能凭本次两次解算精确拆分各因素贡献。

额外理想投影对照使用同一板图物理模型、已知 `t=[25,-35,900] mm`、绕Y轴0.14rad再与绕X轴-0.08rad组合的姿态。
两组独立调用项目 MonoPnPSolver，原点差 `{control_gap:.6f} mm`，支持公共原点放置公式正确。
该对照不经过图像特征提取，不能代替上述真实前端结果。

这张图是参考正视图，尚未确认其裁剪/缩放与当前相机内参一致，因此绝对毫米位姿不能视为新的实物测量。
本次验证的是相同输入与相同内参下两个当前处理分支的输出一致性；实物精度还需同一相机拍摄的标定图像确认。

## 复现与数据

Windows通过项目 Docker 工具链运行：

```text
"C:/Program Files/Git/bin/bash.exe" scripts/docker-toolchain.sh python3 scripts/run_class_center_experiment.py
```

脚本先更新共享跟踪器对象，再构建并运行实验。主配置和算法不修改。
数值见 `poses.csv`，配置快照见 `config_snapshot.json`，输入/源码/模板哈希与差值见 `provenance.json`。
''', encoding='utf-8')
print(json.dumps({'distance_mm': gap, 'pixel_gap': pixel_gap, 'rotation_difference_deg': angle, 'ideal_control_gap_mm': control_gap}, indent=2))
