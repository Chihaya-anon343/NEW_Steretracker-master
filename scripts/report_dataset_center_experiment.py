"""Summarize all selected frames, retaining unsuccessful arms if any."""
from pathlib import Path
import csv
import json
import math
import statistics

root=Path(__file__).resolve().parents[1]
out=root/'output/dataset_center_alignment_20261020'
data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
lines=[]
with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f)
    writer.writerow(['sample','source','class0_success','class1_success','class0_strategy','class1_strategy','class0_z_mm','class1_z_mm','delta_z_mm','distance_mm','image_gap_px','rotation_difference_deg'])
    for s in data['samples']:
        a,b=s['class0'],s['class1']
        if a['success'] and b['success']:
            dot=sum(x*y for x,y in zip(a['R'],b['R']))
            angle=math.degrees(math.acos(max(-1,min(1,(dot-1)/2))))
            s['rotation_difference_deg']=angle
            writer.writerow([s['id'],s['path'],a['success'],b['success'],a['strategy'],b['strategy'],a['t_mm'][2],b['t_mm'][2],s['delta_z_mm'],s['distance_mm'],s['image_gap_px'],angle])
            lines.append(f"| {s['id']} | {Path(s['path']).name} | {a['t_mm'][2]:.2f} | {b['t_mm'][2]:.2f} | {s['distance_mm']:.2f} | {s['image_gap_px']:.2f} | {angle:.2f} |")
        else:
            writer.writerow([s['id'],s['path'],a['success'],b['success'],a['strategy'],b['strategy']])
            lines.append(f"| {s['id']} | {Path(s['path']).name} | 未形成双路有效位姿 | — | — | — | — |")
gaps=[s['distance_mm'] for s in data['samples'] if 'distance_mm' in s]
data['statistics']={'paired_success':len(gaps),'total':len(data['samples']),'min_mm':min(gaps),'max_mm':max(gaps),'mean_mm':statistics.mean(gaps),'median_mm':statistics.median(gaps)}
(out/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
details=[]
for s in data['samples']:
    details.append(f"### {s['id']} — {s['path']}\n\n![两组独立解算]({s['id']}/comparison.png)\n\n![同图叠加局部]({s['id']}/target_detail.png)\n")
report=f'''# 数据集随机样本：class0 / class1 三维轴一致性

日期：2026-10-20。共选6张图、12次独立单目解算。结果：{len(gaps)}/6张两路均成功，但本次三维原点均存在明显差异。

## 抽样与方法

候选来自 `data/big`、`data/mid`、`data/small`、`data/rotated` 以及 `data/stereo_crop/left_*.png`。
按文件内容SHA256去重后共{data['deduplicated_pool_size']}张，固定随机种子20261020打乱。
按随机顺序检测，扫描{data['scanned']}张，保留前6张满足条件的图：YOLO同时检出class0和class1、class1至少95%位于class0内、class0框距画幅边界至少2px、class1框短边至少4px。
不按解算成功与否筛选，不人工挑选位姿最好的样本。抽样顺序见 `randomized_pool.txt`，筛选记录见 `selection_audit.tsv`。
上述完整性是检测框条件，另已人工检查6张可视化均可见完整的外图案及中心小图案。

每张图使用原分辨率与主配置内参，ROI取实际YOLO检测框；一组只传class0，另一组只传class1并设置is_class1。
每次新建MonoTracker、关闭temporal并固定求解器随机种子；无热启动、无共享位姿、不走Dual联合解。
独立BC/TT策略链及尺寸保持当前配置；12次均由BinaryCorner成功输出，均使用10个对应点，model_frame均为class0。
中心偏移均为 `[-3.1501,27.3201] mm`。主配置与生产算法均未修改。

本次5张来自实拍序列裁剪生成的stereo_crop左图，1张来自data/big；没有把裁剪视差当作真实双目距离，本实验也未用右图。

## 数值

| 样本 | 源图 | class0 Z (mm) | class1 Z (mm) | 三维原点距离 (mm) | 图像原点距离 (px) | 旋转差 (°) |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

三维原点距离最小{min(gaps):.2f}mm、最大{max(gaps):.2f}mm、均值{statistics.mean(gaps):.2f}mm、中位数{statistics.median(gaps):.2f}mm。
三维距离计算为 `||t_class1 - t_class0||`，因为两者都输出class0原点在同一相机坐标系的位置。
每路完整XYZ、旋转矩阵、原点像素、重投影误差、检测置信度与ROI见 `results.json`；整理表见 `comparison.csv`。

## 可视化

![6张随机样本的同图轴叠加](gallery.png)

class0为实线、class1为虚线，红+X、绿+Y、蓝-Z（朝上法向）；轴长均55mm。黄色检测框对应class0，青色对应class1。
图像为展示而缩放/裁剪，位姿计算始终使用原图与原始坐标。

![相机X-Z平面的原点与三维轴](camera_space_gallery.png)

相机坐标图每个面板按本样本范围自动缩放，同一面板X、Z方向采用相同mm/像素比例，不能比较不同面板的屏幕长度。

## 结论与边界

当前配置下，6张图上的独立class0和class1分支均未给出相同三维原点。
其中class1的深度都大于class0；较小的重投影误差不能证明其毫米尺度或姿态正确。
既有BC/TT与板图尺寸不一致、中心图案分辨率低、不同靶板实物尺寸/中心布局及角点/姿态误差都可能影响结果；本次未做因素消融，不能据此精确分摊原因。
尤其stereo_crop中的板图与clean_target2不是同一张参考照片，当前偏移和尺寸是否适用其实体靶标尚未实测确认，不能把6张全部视为同一物理尺寸的板。
这证明当前配置在这些样本上的输出一致性存在问题，不代表已测得真实世界定位误差；无外部三维真值，且各图裁剪/缩放与相机标定关系未全部确认。
抽样受YOLO能检出两类目标这一条件限制，并不是整个数据集的成功率统计。

## 单图结果

{chr(10).join(details)}

复现入口：经项目Docker工具链运行 `python3 scripts/run_dataset_center_experiment.py`。
可视化汇总由 `scripts/render_dataset_center_results.cpp` 生成，报告由 `scripts/report_dataset_center_experiment.py` 生成。
配置快照见 `config_snapshot.json`，输入与程序哈希、统计见 `summary.json`。
'''
(out/'report.md').write_text(report,encoding='utf-8')
print(json.dumps(data['statistics'],indent=2))
for s in data['samples']:
    print(s['id'],s['path'],round(s.get('distance_mm',-1),2))
