# -*- coding: utf-8 -*-
"""Report saved measurements; no tracking or parameter tuning."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("output/mono_priority_20261019")
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def stat(values):return np.median(values),np.percentile(values,95),np.max(values)

def main():
    pose=read(ROOT/"pose_summary.json");corner=read(ROOT/"corner_summary.json");timing=read(ROOT/"timing_summary.json")
    pipeline=read(ROOT/"pipeline_timing.json");frames=read(ROOT/"pose_frames.json")
    figures=ROOT/"figures";figures.mkdir(exist_ok=True)
    modes=("baseline","roi","quality","partial");labels=("Baseline","ROI","Quality","Partial")
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    active=[next(s for s in pose if s["mode"]==m and s["scope"]=="tilt_le30")for m in modes]
    for ax,key,title in zip(axes,("success","normal_jumps","depth_jumps"),("Valid pose outputs / 455","Normal-direction jumps","Depth jumps")):
        values=[r[key]for r in active];bars=ax.bar(labels,values,color=["#777777","#359c9c","#d8963b","#4c78a8"])
        ax.set_title(title)
        for b,v in zip(bars,values):ax.annotate(str(v),(b.get_x()+b.get_width()/2,b.get_height()),ha="center",va="bottom")
        ax.set_ylim(0,max(values)*1.18+1)
    fig.savefig(figures/"pipeline_quality.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    c=[next(s for s in corner if s["mode"]==m and s["scope"]=="tilt_le30")for m in ("baseline","quality","partial")]
    axes[0].bar(["Baseline","Quality","Partial"],[s["success"]for s in c]);axes[0].set(title="BC outputs / 411 fixed GT ROIs",ylim=(380,411))
    axes[1].bar(["Baseline","Quality","Partial"],[s["point_error_px"]["p95"]for s in c]);axes[1].set(title="Retained-point error P95",ylabel="pixels",ylim=(0,3))
    fig.savefig(figures/"corner_quality.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6.4),constrained_layout=True,sharex=True)
    for mode,color in (("baseline","#d75a4a"),("partial","#327ca8")):
        rows=[r for r in frames if r["mode"]==mode and r["sequence"]=="oblique_degraded"]
        axes[0].plot([r["frame"]for r in rows],[r.get("depth_mm",np.nan)/1000 for r in rows],label=mode,color=color,lw=1)
        axes[1].plot([r["frame"]for r in rows],[r.get("normal_error_deg",np.nan)for r in rows],label=mode,color=color,lw=1)
    gt=read("tests/data/mono_robustness_20261015/oblique_degraded/manifest.json")["frames"]
    axes[0].plot(list(range(len(gt))),[g["t_mm"][2]/1000 for g in gt],"k--",label="GT",lw=1)
    axes[0].set(yscale="log",ylabel="Depth (m, log scale)",title="Actual sequential YOLO + strategy + PnP outputs")
    axes[1].set(ylabel="Normal error (deg)",xlabel="Source frame (0 based)")
    for ax in axes:
        ax.fill_between(list(range(len(gt))),0,1,where=np.array([g["tilt_deg"]for g in gt])>30,
                        transform=ax.get_xaxis_transform(),color="gray",alpha=.10,label="GT tilt >30 deg")
        for frame in(75,149):ax.axvline(frame,color="gray",ls=":",lw=.8)
        ax.legend();ax.grid(alpha=.2)
    fig.savefig(figures/"degraded_depth_trace.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    selected=[next(s for s in timing if s["scope"]=="common_success"and s["mode"]==m)for m in("baseline","quality","partial")]
    x=np.arange(3)
    axes[0].bar(x-.18,[s["median_ms"]for s in selected],.36,label="median")
    axes[0].bar(x+.18,[s["p95_ms"]for s in selected],.36,label="P95")
    axes[0].set(xticks=x,xticklabels=["Baseline","Quality","Partial"],ylabel="ms",title="BC: 61 common-success ROIs x 12");axes[0].legend()
    means=[p["yolo_ms"]+p["track_ms"]*p["track_ms_calls"]/p["yolo_ms_calls"]for p in pipeline]
    axes[1].bar(labels,means);axes[1].set(ylabel="ms / input frame",title="Serial reference: YOLO + tracking mean")
    fig.savefig(figures/"performance.png",dpi=150);plt.close(fig)
    idx={(r["mode"],r["sequence"],r["frame"]):r for r in frames}
    common=[(r,idx["partial",r["sequence"],r["frame"]])for r in frames if r["mode"]=="baseline"and r["tilt_gt"]<=30 and r["success"]and idx["partial",r["sequence"],r["frame"]]["success"]]
    text=["# 单目 BC：ROI 物理先验、可靠角点与有限缺角优化",
         "",
         "2026-10-19。三项优化已经实现并完成 Docker 构建、单元/集成测试、固定 ROI 实验和完整单目序列对照。",
         "",
         "## 1. 主要结论",
         "",
         "- 30°以内完整序列：有效位姿由 **448/455 降至 439/455**，深度跳变由 **1 次降至 0 次**；法向跳变在完整方案下仍为 **9 次**。更保守的 ROI 拒绝有实际代价。",
         "- 固定标注 ROI 的 class0 BC：输出由 **398/411 增至 405/411**；保留点误差 P95 为 **2.420→2.399px**，编号异常诊断均为 0。新增输出不能直接等同于完整 YOLO 链路成功率提升。",
         "- oblique_degraded 第 75 帧由失败恢复为 BC 成功；第 149 帧深度相对误差从 **4083.02% 降至 20.28%**，该帧仍不够准确。",
         "- 本轮以鲁棒性为目标。BC 的可靠点检查与缺角匹配增加了部分计算量，不能宣称整体加速。",
         "",
         "## 2. 实现与接口",
         "",
         "### ROI 选择",
         "",
         "共享 ROI 层使用原始检测框检查 class1 被 class0 覆盖的比例，默认至少 95%，边界容差 2px；在外扩和立体尺寸归一化之前执行。同类别重复框去重，最终每类最多选择一个目标。class1 可参与 class0 候选评分，即使本帧不触发 DualROI。",
         "",
         "保留 class0-only、class1-only 近距离回退。单目按置信度、包含关系及最近可信 ROI 的软评分排序，最多验证三个候选，采用首个通过图案和观测跨度检查的候选。历史奖励随 15 帧龄衰减，不强制锁定旧 ROI。",
         "",
         "图案验证同时覆盖 BC 与 TT，要求匹配分数至少 0.75、观测横纵跨度不低于 ROI 的 25%，最短观测跨度至少 4px。阈值不是所有分辨率下的最佳值，本次确实出现误拒。主路径复用验证阶段的 BC/TT 提取结果；短路到 secondary 的路径可能需要重新提取。候选验证不提交位姿/策略状态，整帧最终只更新一次。",
         "",
         "YOLO 的 NMS 改为按类别抑制，避免相互嵌套的不同类别框被提前删除。双目归一化使用实际选中的右侧检测框；两侧 primary 类别不一致时不组成有效立体对。左图光流流程不增加右图 YOLO。",
         "",
         "### 可靠角点与编号",
         "",
         "H 验证对全部对应点计算逐点残差；必要时稳健估计后剔除不可靠点，并对保留点重新拟合。至少保留八个 class0 BC 点，模板凸包面积和横纵覆盖均不低于 60%；完整图案 IoU、可见区域与投影有效性检查继续生效。",
         "",
         "`good_matches.trainIdx` 保留真实模板编号。例如删除 2、7 后，其余点不会被重新解释为模板 0～7。二维点、模板点、三维点与残差同步筛选。DualROI、class1 回退和联合匹配同步适配；联合匹配不跨类别，已经建立 class0 身份的点不再任意改号。`pose_model_points` 是与模板编号对应的模型表。",
         "",
         "PnP 候选在相同可信点集上评价；class0 BC 还要求足够的重投影支持。TT 四点规则保持独立。最终可视化的颜色、编号和重投影都使用真实模板编号。新增失败原因、ROI 选择及可靠点统计；日志聚合补入 ROI 验证耗时。",
         "",
         "### 有限缺角",
         "",
         "优先保留十点快速路径。角点数不满足或十点验证失败时，尝试相邻简化尺度的一个实际轮廓候选，支持 8～12 个候选观测。四方向粗归一化之后，以距离和跳过惩罚进行有序循环序列对齐，允许遗漏少量模板点或跳过伪角点，每次部分匹配最多验证 16 个 H 候选。",
         "",
         "候选必须通过整体图案、空间覆盖、前向投影和身份评分差距验证。缺失位置不补成测量点；PnP 只使用真实观测。首版没有增加外层反复重匹配循环，保留稳定的预算上限。该能力不承诺大面积遮挡或目标出画恢复。class1 默认继续采用原提取流程。",
         "",
         "## 3. 实验设置",
         "",
         "数据集：`tests/data/mono_robustness_20261015`，六段合成序列共 1,020 帧。四组完整流程共 **4,080 帧**。内参、模板物理尺度、30°倾角约束以及中心偏移 `[0,0]` 固定。标注只进入固定 ROI 准备和离线误差统计，不进入实际 YOLO/匹配/PnP。",
         "",
         "| 组别 | 配置 |",
         "|---|---|",
         "| A baseline | 上一轮 refined BC；归档修改前二进制 |",
         "| B roi | A + NMS/ROI 物理关系、候选验证与重选 |",
         "| C quality | B + 可靠点筛选与编号传递 |",
         "| D partial | C + 有限缺角匹配 |",
         "",
         "固定 ROI 另运行 baseline/quality/partial 三组，每组 2,040 个 class0/class1 ROI，共 **6,120 次提取**。基线输出与上次归档坐标逐项一致；class1 在三组中的提取坐标也逐项一致。",
         "",
         "## 4. 30°以内完整单目流程",
         "",
         "| 版本 | 有效位姿 | 法向跳变 | 深度跳变 | 法向误差中位数/P95（°） | 深度相对误差中位数/P95 |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in active:text.append("| %s | %d/%d | %d | %d | %.3f / %.3f | %.2f%% / %.2f%% |"%(r["mode"],r["success"],r["frames"],r["normal_jumps"],r["depth_jumps"],r["normal_error_deg"]["median"],r["normal_error_deg"]["p95"],100*r["depth_rel_error"]["median"],100*r["depth_rel_error"]["p95"]))
    text += ["", "ROI-only 本次法向跳变为 7 次，完整方案为 9 次，因此不能称完整方案在每个指标上都优于 ROI-only。完整方案的最大法向误差从基线 35.21°降至 31.56°；最大深度相对误差从 4083.02%降至 27.16%。", "", "跳变沿用原实验口径：相邻有效帧法向变化超过 20°且比真实变化多 15°；深度相对步长超过 30%且比真实步长多 20%。工作范围按当前帧标注倾角≤30°筛选，断帧后的恢复跳变单独处理，不计入上述相邻帧数。", "", "基线与完整方案共同成功的 **%d 帧**对照："%len(common), "", "| 指标 | baseline | partial |", "|---|---:|---:|"]
    for key,label,factor in(("normal_error_deg","法向误差中位数/P95（°）",1),("depth_rel_error","深度相对误差中位数/P95（%）",100)):
        a=stat([r[0][key]*factor for r in common]);b=stat([r[1][key]*factor for r in common]);text.append("| %s | %.3f / %.3f | %.3f / %.3f |"%(label,a[0],a[1],b[0],b[1]))
    text += ["", "相同成功帧上的法向 P95 略有变差。新增拒绝并非全部合理：reference 第 2、3 帧旧版深度误差约为 3.84%、6.76%，新版直接拒绝，提示小目标验证阈值仍需后续标定。范围内共恢复两个原失败帧，同时失去十一个原成功帧，净少九帧。", "", "![完整流程指标](../output/mono_priority_20261019/figures/pipeline_quality.png)", "", "## 5. 固定 ROI 的 BC 质量", "", "以下为完整可见且倾角≤30°的 411 个 class0 ROI。误差只统计实际保留的点，不能把未观测角点当作零误差。", "", "| 版本 | 输出 | 八九点输出 | 保留点误差中位数/P95（px） | 最大误差（px） | 编号异常诊断帧 |", "|---|---:|---:|---:|---:|---:|"]
    for r in c:text.append("| %s | %d/%d | %d | %.3f / %.3f | %.3f | %d |"%(r["mode"],r["success"],r["attempts"],r["partial_outputs"],r["point_error_px"]["median"],r["point_error_px"]["p95"],r["point_error_px"]["max"],r["mismatch_frames"]))
    text += ["", "编号异常通过固定编号误差与最优指派误差作诊断，不等同于逐帧人工认证。缺角输出增加了可用观测，但中位数略升，未带来完整链路成功率增加。", "", "![固定 ROI 质量](../output/mono_priority_20261019/figures/corner_quality.png)", "", "## 6. 耗时", "", "BC 独立实验：相同解码图、相同 ROI、一个 OpenCV 线程、预热后交错执行 12 次；68 个 class0 加 17 个 class1 ROI，共 3,060 次调用。测试期间没有并行构建或序列实验。", "", "| 范围 | 版本 | 次数 | 中位数（ms） | P95（ms） |", "|---|---|---:|---:|---:|"]
    for r in timing:text.append("| %s | %s | %d | %.3f | %.3f |"%(r["scope"],r["mode"],r["calls"],r["median_ms"],r["p95_ms"]))
    text += ["", "共同成功的 61 个 ROI 上，完整方案中位数增加约 7.2%；包含失败与部分匹配尝试的全部 68 个 class0 ROI 上，中位数增加约 21.9%。P95 的小幅波动不能作为加速结论。", "", "完整流程另用 reference 120 帧逐组串行测量。以下是图像就绪后的检测与跟踪均值，不包含图像解码、模型初始化和显示；YOLO 和跟踪调用数量不同时按输入帧数加权。", "", "| 版本 | YOLO（ms/输入帧） | 跟踪（ms/调用） | YOLO＋跟踪（ms/输入帧） |", "|---|---:|---:|---:|"]
    for r,total in zip(pipeline,means):text.append("| %s | %.3f | %.3f | %.3f |"%(r["mode"],r["yolo_ms"],r["track_ms"],total))
    text += ["",
         "串行测量每组一次，整体差异受运行波动影响，不据此宣称 FPS 稳定提升。本轮没有重新测量 RSS 或 GPU 占用。",
         "",
         "![耗时](../output/mono_priority_20261019/figures/performance.png)",
         "",
         "## 7. 代表结果与边界",
         "",
         "以下首先展示完整原序列的位姿轨迹。第 149 帧：GT Z=551.75mm，旧版 23079.94mm，完整方案 663.64mm。剩余约 20.3%偏差没有解决；当前范围内最差约 27.2%偏差也来自 class1-only BC，需要继续单独评估 class1 角点及物理尺度。",
         "",
         "![完整序列深度与法向](../output/mono_priority_20261019/figures/degraded_depth_trace.png)",
         "",
         "下面的图由最终 C++ 主程序真实输出。它们是单图冷启动调试，不代替上述完整历史序列记录；八点示例使用标注 ROI 并在独立配置中强制 BC，以验证八点编号到 PnP 和可视化。",
         "",
         "![第75帧候选ROI](../output/mono_priority_20261019/visuals/roi_recovery/mono_roi_selection_f1.png)",
         "",
         "![八点编号与归一化](../output/mono_priority_20261019/visuals/eight_corners/mono_identity_f1.png)",
         "",
         "![八点重投影](../output/mono_priority_20261019/visuals/eight_corners/mono_bc_reproj_f1.png)",
         "",
         "超过30°的 565 帧作为压力测试保留，不用于部署范围成功率判断。全部1,020帧的四组有效输出为893/876/856/857，深度跳变为19/17/17/17，说明方案没有解决超范围输入下的所有异常。",
         "",
         "## 8. 配置、验证与复现",
         "",
         "主 JSON 已启用 `roi_selection.enabled/verify_mono` 与 `mono_identity.quality_filter/partial_matching`。三个开关可分别消融；旧配置缺省时新模块关闭。类别 NMS 修正独立保留。class1 新编号前端仍默认关闭，中心偏移仍为 `[0,0]`。",
         "",
         "- 第一阶段八个相关 CTest 目标全部通过：[日志](../output/mono_priority_20261019/ctest.log)。",
         "- 最终诊断、日志和可视化修复后针对性回归通过：[日志](../output/mono_priority_20261019/final_ctest.log)。",
         "- 共享 ROI 最终19项测试通过，含混类立体对拒绝和右侧选定检测锚点：[日志](../output/mono_priority_20261019/final_roi.log)。",
         "- 新增角点测试覆盖一两个坏点、8/9点缺失、11/12候选点、错误形状拒绝和随机数状态保护；原有单/双目隔离测试保留。",
         "- 完整实验使用归档二进制和配置；后续补充编号绘制、日志、拒绝原因及双目混类守卫。最终主程序另重放 reference 三种新配置与 oblique_degraded 完整方案，数值一致性记录：[验证](../output/mono_priority_20261019/final_numerical_verification.json)。",
         "",
         "实验目录：`output/mono_priority_20261019`。`experiment.json`、`completion.json`、`source_verification.json` 保存配置、二进制、源代码、模型和数据清单校验信息；逐帧原始记录位于 `runs/`。",
         "",
         "```bash",
         "# 以下命令在项目目录执行，经 Docker 工具链运行",
         "scripts/docker-toolchain.sh bash scripts/run_mono_priority_tests.sh",
         "scripts/docker-toolchain.sh python3 scripts/experiment_mono_priority.py run",
         "# 统计报告使用带 OpenCV / NumPy / SciPy / Matplotlib 的 Python",
         "python scripts/experiment_mono_priority.py analyze",
         "python scripts/report_mono_priority.py",
         "```",
         "",
         "`run` 校验并复用已完成的归档任务。完整对照完成后不应再次执行 `prepare` 覆盖归档；测试新参数应另建实验目录。"]
    marker=text.index("## 7. 代表结果与边界")
    scale=read(ROOT/"class1_scale_check.json")
    text.insert(marker+2,"另外核对了 class1 的尺度：现有 BC 模型包围尺寸约 %.2f×%.2fmm，而合成生成器的标注轮廓为 %.2f×%.2fmm，存在约15～16%%差异。这能解释一部分系统性深度偏差；其余部分仍需与 class1 定位误差分开验证。本轮保持原物理尺寸用于算法对照。详见 `class1_scale_check.json`。"%(*scale["bc_extents_mm"],*scale["gt_extents_mm"]))
    text.insert(marker+3,"")
    text.insert(text.index("## 3. 实验设置")+2,"文中帧号使用源图像的0起始编号；位姿 CSV 的 `frame` 列为1起始编号。")
    text.insert(text.index("## 3. 实验设置")+3,"")
    Path("docs/mono_bc_priority_20261019.md").write_bytes(("\n".join(text)+"\n").encode())
    print("Report and figures generated")

if __name__=="__main__":main()
