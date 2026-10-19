#!/usr/bin/env python3
"""Render frozen comparison results; no algorithm/config mutations."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_mono_robustness import read,dump,stats,sha

OUT=Path("output/mono_normalized_20261016")
NAMES={"current":"Current BC","normalized":"Normalized","refined":"Normalized + edge fit"}


def main():
    corners=read(OUT/"corner_summary.json");frames=read(OUT/"corner_frames.json")
    poses=read(OUT/"pose_summary.json");pose_frames=read(OUT/"pose_frames.json")
    timing=list(csv.DictReader((OUT/"paired_timing.csv").open()))
    timing_summary=[]
    for cls in (0,1):
        for mode in NAMES:
            rows=[r for r in timing if r["mode"]==mode and int(r["cls"])==cls]
            timing_summary.append(dict(cls=cls,mode=mode,calls=len(rows),jobs=len({r["job"] for r in rows}),
                success=sum(int(r["success"]) for r in rows),
                extract_ms=stats([float(r["extract_ms"]) for r in rows]),
                mean_ms=float(np.mean([float(r["extract_ms"]) for r in rows])),
                stages={k:stats([float(r[k]) for r in rows]) for k in ("normalize_ms","corners_ms","refine_ms","validate_ms")}))
    dump(OUT/"timing_summary.json",timing_summary)
    # Also compare per-job means and the common-success subset (reject-fast paths
    # must not masquerade as a speedup for successful feature extraction).
    jobs={int(r["job"]) for r in timing if r["cls"]=="0"}
    paired=[]
    for job in sorted(jobs):
        groups={mode:[r for r in timing if r["mode"]==mode and int(r["job"])==job] for mode in NAMES}
        assert all(len(g)==12 for g in groups.values())
        paired.append(dict(job=job,common_success=all(all(r["success"]=="1" for r in g) for g in groups.values()),
            means={mode:float(np.mean([float(r["extract_ms"]) for r in g])) for mode,g in groups.items()}))
    common=[r for r in paired if r["common_success"]]
    common_ids={r["job"] for r in common}
    common_calls={m:stats([float(r["extract_ms"]) for r in timing if r["mode"]==m and int(r["job"]) in common_ids]) for m in NAMES}
    dump(OUT/"timing_paired.json",dict(jobs=paired,common_success_jobs=len(common),
        common_success_calls=common_calls,
        common_success_means={m:stats([r["means"][m] for r in common]) for m in NAMES}))
    figdir=OUT/"figures";figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for mode,label in NAMES.items():
        er=np.sort([v for r in frames if r["mode"]==mode and r["cls"]==0 and r["visible"] and r["output"] for v in r["direct"]])
        axes[0].plot(er,np.arange(1,len(er)+1)/len(er),label=label)
    axes[0].set(xlim=(0,10),ylim=(0,1),xlabel="Fixed-ID corner error (px)",ylabel="CDF",title="Fully visible class0")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    selected=[next(r for r in corners if r["mode"]==m and r["cls"]==0 and r["scope"]=="full_visible") for m in NAMES]
    axes[1].bar(range(3),[r["outputs"]/r["attempts"] for r in selected])
    axes[1].set(xticks=range(3),xticklabels=["Current","Normalized","+ edge fit"],ylim=(0,1.05),ylabel="Output fraction",title="Coverage / rejected frames")
    for i,r in enumerate(selected):axes[1].text(i,.05,f"{r['outputs']}/{r['attempts']}",ha="center",color="white")
    times=[[float(r["extract_ms"]) for r in timing if r["mode"]==m and r["cls"]=="0"] for m in NAMES]
    axes[2].boxplot(times,labels=["Current","Normalized","+ edge fit"],showfliers=False)
    axes[2].set(yscale="log",ylabel="BC time (ms, log)",title="Paired isolated timing")
    fig.savefig(figdir/"corner_and_timing.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for axrow,sequence in zip(axes,["reference","oblique_degraded"]):
        truth=None
        for mode,label in NAMES.items():
            rows=[r for r in pose_frames if r["mode"]==mode and r["sequence"]==sequence]
            axrow[0].plot([r.get("depth_mm",np.nan) for r in rows],label=label,lw=1)
            axrow[1].plot([r.get("normal_error_deg",np.nan) for r in rows],label=label,lw=1)
            if truth is None:
                manifest=read(Path("tests/data/mono_robustness_20261015")/sequence/"manifest.json")
                truth=[r["t_mm"][2] for r in manifest["frames"]]
        axrow[0].plot(truth,"k--",lw=.8,label="GT")
        axrow[0].set(title=sequence+" / depth",xlabel="frame",ylabel="camera Z translation (mm)",yscale="log")
        axrow[1].set(title=sequence+" / normal",xlabel="frame",ylabel="normal error (deg)")
        axrow[0].legend(fontsize=7);axrow[1].legend(fontsize=7)
    fig.savefig(figdir/"pose_comparison.png",dpi=160);plt.close(fig)

    fmt=lambda v:"—" if v is None else f"{v:.3f}"
    lines=["## 实验结果","", "### 标注角点：完整可见 class0","",
        "| 版本 | 输出/尝试 | 编号异常诊断帧 | 固定编号误差中位数 / P95（px） | 最大误差（px） | 全部十点≤2px帧数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in selected:
        lines.append(f"| {r['mode']} | {r['outputs']}/{r['attempts']} | {r['mismatches']} | {fmt(r['direct']['median'])} / {fmt(r['direct']['p95'])} | {fmt(r['direct']['max'])} | {r['within2']} |")
    lines += ["", "≤30°且完整可见的 class0 ROI 单独统计：", "",
        "| 版本 | 输出/尝试 | 固定编号误差中位数 / P95（px） | 全部十点≤2px帧数 |",
        "|---|---:|---:|---:|"]
    for mode in NAMES:
        r=next(s for s in corners if s["mode"]==mode and s["cls"]==0 and s["scope"]=="tilt_le30")
        lines.append(f"| {mode} | {r['outputs']}/{r['attempts']} | {fmt(r['direct']['median'])} / {fmt(r['direct']['p95'])} | {r['within2']} |")
    lines += ["", "本应用角度范围内，新前端的独立 BC 输出数略低，主要是拒绝无法获得可靠十点结构的样本；需结合完整链路的后备策略成功率评估，不能只根据成功点误差宣布全面优于原方案。"]
    pair=read(OUT/"corner_paired.json")
    lines += ["",f"三组共同成功的 {pair['common_frames']} 帧：","",
        "| 版本 | 固定编号误差中位数 / P95（px） | 最优指派误差 P95（px） |",
        "|---|---:|---:|"]
    for mode,r in pair["modes"].items():lines.append(f"| {mode} | {fmt(r['direct']['median'])} / {fmt(r['direct']['p95'])} | {fmt(r['assigned']['p95'])} |")
    lines += ["", "编号异常使用固定编号与最优指派差异进行诊断，不等同于逐帧人工认证的准确率。class1 三组坐标一致性已逐条断言检查。", "",
        "### 独立耗时","", "| 版本 | class0 调用数 | BC 中位数 / P95（ms） | 平均耗时（ms） |",
        "|---|---:|---:|---:|"]
    for r in timing_summary:
        if r["cls"]==0:lines.append(f"| {r['mode']} | {r['calls']} | {fmt(r['extract_ms']['median'])} / {fmt(r['extract_ms']['p95'])} | {fmt(r['mean_ms'])} |")
    lines += ["",f"三组共同成功的 {len(common)} 个计时 ROI 另见 `timing_paired.json`，用于排除快速拒绝造成的选择偏差。", "",
        "| 版本 | 共同成功 ROI 逐次调用中位数 / P95（ms） | 逐 ROI 平均耗时中位数 / P95（ms） |", "|---|---:|---:|"]
    for mode in NAMES:
        r=stats([x["means"][mode] for x in common])
        call=common_calls[mode]
        lines.append(f"| {mode} | {fmt(call['median'])} / {fmt(call['p95'])} | {fmt(r['median'])} / {fmt(r['p95'])} |")
    lines += ["",
        "### 完整单目链路","", "| 范围 | 版本 | 成功/总帧 | 法向跳变 | 深度跳变 | 法向误差 P95（°） | 深度相对误差中位数 / P95 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in poses:
        if r["sequence"]!="all":continue
        lines.append(f"| {r['scope']} | {r['mode']} | {r['success']}/{r['frames']} | {r['normal_jumps']} | {r['depth_jumps']} | {fmt(r['normal_error_deg']['p95'])} | {fmt(r['depth_rel_error']['median'])} / {fmt(r['depth_rel_error']['p95'])} |")
    lines += ["", "分序列、策略分布、逐帧误差见 `pose_summary.json`、`pose_frames.json`。全部压力数据中含超过当前 30°先验的视角，其误差不能用于宣称应用范围内精度。", "",
        "主配置采用 refined。相较 current，它改善了角点精度与法向跳变，但不是所有位姿指标都改善：≤30°范围的深度相对误差中位数约由 1.43% 增至 1.92%，P95 约为 20.75% 且未变。未精化 normalized 组法向跳变反而升至 13 次，因此不作为默认路径。", "",
        "![角点与独立耗时](../output/mono_normalized_20261016/figures/corner_and_timing.png)","",
        "![位姿对比](../output/mono_normalized_20261016/figures/pose_comparison.png)","",
        "![新流程调试面板](../output/mono_normalized_20261016/debug_reference_f45/mono_identity_c0_f1.png)","",
        "### 局限","",
        "- 数据为同源贴图合成，尚不能替代实拍、曝光变化、反光与遮挡测试。",
        "- 四方向粗归一化仍有透视残差；无法形成可靠十点结构时会拒绝并触发后备策略。",
        "- 原图精化依赖分割边界，不能消除源图模糊、纹理裂缝及标注偏差。",
        "- 现有平面 PnP 的剩余二义性、class1 BC/TT 物理尺度差异不在本轮改动范围内，残余法向/深度跳变需如实保留。",
        "- 保留旧入口用于对照和回退，新入口只作用于单目 class0。"]
    doc=Path("docs/mono_bc_normalized_20261016.md");text=doc.read_text(encoding="utf8")
    start=text.index("<!-- RESULTS:BEGIN -->");end=text.index("<!-- RESULTS:END -->")+len("<!-- RESULTS:END -->")
    doc.write_text(text[:start]+"<!-- RESULTS:BEGIN -->\n"+"\n".join(lines)+"\n<!-- RESULTS:END -->"+text[end:],encoding="utf8")
    print("Wrote report and figures",doc)


if __name__=="__main__":main()
