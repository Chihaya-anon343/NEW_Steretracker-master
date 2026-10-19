#!/usr/bin/env python3
"""Render the frozen mono identity experiment, including rejected outcomes."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_mono_robustness import read,dump,sha

OUT=Path("output/mono_identity_mono_20261016")
CORNERS=Path("output/mono_identity_final_20261016")


def main():
    corners=read(CORNERS/"corner_summary.json")
    poses=read(OUT/"pose_summary.json")
    experimental=read(CORNERS/"pose_summary.json")
    frames=read(CORNERS/"corner_frames.json")
    (OUT/"figures").mkdir(exist_ok=True)
    timing=list(csv.DictReader((OUT/"paired_timing.csv").open(encoding="utf8")))
    jobs={j["job"]:j for j in read(CORNERS/"corner_jobs.json")}
    time_summary=[]
    for cls in (0,1):
        for mode in ("legacy","hybrid"):
            values=np.array([float(r["extract_ms"]) for r in timing if r["mode"]==mode and jobs[int(r["job"])]["cls"]==cls])
            time_summary.append(dict(cls=cls,mode=mode,n=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),mean=float(np.mean(values))))
    dump(OUT/"timing_summary.json",time_summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for mode in ("legacy","tt_fourway","projective","hybrid"):
        selected=[x for x in frames if x["mode"]==mode and x["cls"]==0 and x["visible"] and x["output"]]
        errors=np.sort([e for x in selected for e in x["direct"]])
        axes[0].plot(errors,np.arange(1,len(errors)+1)/len(errors),label=mode)
    axes[0].set_xscale("log");axes[0].set_xlabel("Fixed-ID point error (px), accepted frames");axes[0].set_ylabel("CDF");axes[0].legend();axes[0].grid(alpha=.3)
    groups=[x for x in corners if x["cls"]==0 and x["visible"] and x["group"]=="all"]
    axes[1].bar([x["mode"] for x in groups],[100*x["outputs"]/x["attempts"] for x in groups])
    axes[1].set_ylabel("Frames returning 10 points (%)");axes[1].set_ylim(0,100);axes[1].grid(axis="y",alpha=.3)
    fig.tight_layout();fig.savefig(str(OUT/"figures/corner_comparison.png"),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for root,mode,label in [(OUT,"legacy","Legacy"),(CORNERS,"hybrid","Class0 + class1 experiment"),(OUT,"hybrid","Default: class0 only")]:
        rows=read(root/"pose_frames.json");rows=[r for r in rows if r["sequence"]=="reference_"+mode and r["success"]]
        ax.plot([r["frame"] for r in rows],[r["depth_mm"]/1000 for r in rows],label=label)
    ax.set_xlabel("Reference frame (zero-based)");ax.set_ylabel("Estimated camera Z (m)");ax.set_yscale("log");ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(str(OUT/"figures/depth_comparison.png"),dpi=160);plt.close(fig)

    def point_table(cls):
        lines=["| 方法 | 尝试 / 输出 / 拒绝 | 固定编号中位数 / P95 | 对应偏移诊断帧 | 全十点≤2px帧数 |",
               "|---|---:|---:|---:|---:|"]
        for s in corners:
            if not(s["cls"]==cls and s["visible"] and s["group"]=="all"):continue
            error="—" if not s["direct"] else "%.2f / %.2f px"%(s["direct"]["median"],s["direct"]["p95"])
            lines.append("| %s | %d / %d / %d | %s | %d | %d |"%(s["mode"],s["attempts"],s["outputs"],s["rejections"],error,s["mismatch_frames"],s["all_points_within2"]))
        return "\n".join(lines)
    def pose_table(scope):
        lines=["| 序列 | 帧数 | 成功帧 原版→新方案 | 法向P95 原版→新方案 | 法向突跳 原版→新方案 | 深度突跳 原版→新方案 |",
               "|---|---:|---:|---:|---:|---:|"]
        for seq in ("reference","oblique_clean","grazing","normal_ambiguity","fast_rotation","oblique_degraded"):
            a=next(x for x in poses if x["sequence"]==seq and x["mode"]=="legacy" and x["scope"]==scope)
            b=next(x for x in poses if x["sequence"]==seq and x["mode"]=="hybrid" and x["scope"]==scope)
            if not a["frames"]:continue
            fmt=lambda v:"—" if v is None else "%.2f°"%v
            lines.append("| %s | %d | %d→%d | %s→%s | %d→%d | %d→%d |"%(seq,a["frames"],a["success"],b["success"],fmt(a["normal_error_deg"]["p95"]),fmt(b["normal_error_deg"]["p95"]),a["normal_jumps"],b["normal_jumps"],a["depth_jumps"],b["depth_jumps"]))
        return "\n".join(lines)
    lines=["# 单目 BC 编号修复：实现与实验（2026-10-16）", "",
        "本次在单目 BC 的编号环节实现 TT 几何四点粗归一化、四方向评分、十点 H 精化与轮廓验证，并完成角点消融、真实 YOLO 策略链对照、位姿回归和成对计时。双目未启用这套新编号逻辑。", "",
        "## 实现范围", "",
        "- 保留既有 BC 原始角点产生过程，以便隔离编号修复的效果；四方向粗定向用于最终物理编号之前，尚未用它替换整个旧 BC 前处理或其 24 角度旋转选域。没有据此声称全前端加速。",
        "- 四点来自已选前景轮廓的 minAreaRect，不新增完整 TT ×4 放大链。归一化到 128×128，四个方向从统一标准模型派生。",
        "- 可信的粗方向生成十点初始对应并精化 H；不可信时检查完整循环起点。统一绕向并检查投影极点与图像有效覆盖。",
        "- H 使用全部点，噪声较大时进行稳健估计，十点至少需要八点支持。实际前景轮廓提供独立于稀疏角点的图案证据；检查 IoU 和候选区分度。",
        "- 验证前景从最终选中的稠密轮廓映射回原 ROI，确保与角点来自同一选域。新增目标相对 ROI 占比检查，避免内图案套用外图案尺度。",
        "- 接受后只重排实际检测点，保持固定模板 ID；没有用模板预测点替换检测点。物理尺寸、原点、class1 偏移 [0,0] 和 30°位姿先验沿用当前配置。",
        "- 只有 MonoTracker 显式启用单目上下文；StereoTracker 的左图 extractMono 调用不会因此改变。",
        "- **默认仅 class0 启用；class1_enabled=false 保持原链路。** class1 同样支持实验开关，但全类别实验发现更多 BC/TT 尺度切换造成的深度跳变，因此没有作为默认行为。", "",
        "## 配置", "", "位于 `strategies.binary_corner.mono_identity`；缺省或 enabled=false 保留旧行为。", "",
        "```json",json.dumps(dict(enabled=True,class1_enabled=False,method="hybrid",normalized_size=128,min_span_px=16.0,min_roi_span_ratio=.25,min_iou=.75,min_score_gap=.05,max_fit_error_px=4.0,max_fit_error_ratio=.03),ensure_ascii=False,indent=2),"```", "",
        "method 可取 tt_fourway（仅四向分支）、projective（完整候选）或 hybrid。粗分支还要求较高的方向 IoU 与方向区分度；四点配准本身不证明编号正确。", "",
        "## 数据与统计口径", "",
        "- 复用 `tests/data/mono_robustness_20261015` 六组 1020 张合成图。每张图分别在标注 ROI 对 class0/class1 强制运行 BC；每种模式 2040 个输入，共 8160 条原版/新方案对照记录。",
        "- 原版基线复用已完成、参数一致的实验，逐点核对与最初角点实验一致。新方案三种模式共实际运行 6120 次提取；复用来源和哈希见 baseline_reuse.json。",
        "- 像素误差按固定物理编号计算，只在有完整十点输出的帧上统计；同时列出拒绝数与全点达标帧数，避免依赖丢弃样本获得好看 P95。以下主表仅含完整可见目标。",
        "- class1 角点消融是显式启用其新编号的风险评估，早期实验可执行文件等效于 class1_enabled=true；实际默认跟踪器配置为 false。",
        "- 每帧真值仅用于事后误差评估，不进入 C++ 候选选择。图案标注是源图人工角点，仍有标注不确定性。", "",
        "## class0 角点消融", "",point_table(0), "",
        "![角点误差与输出率](../output/mono_identity_mono_20261016/figures/corner_comparison.png)", "",
        "## class1 实验结果（默认不开启）", "",point_table(1), "",
        "## 默认配置的完整单目链路", "",
        "完整链路使用真实 YOLO ROI、现有面积策略、Dual-ROI、退化链与 PnP。原版 1020 帧基线复用；默认新配置新运行 1020 帧。两者运行配置除新编号开关和输出目录外保持一致。", "",
        "用户设定总倾角上限 30°，以下先报告真实倾角≤30°的适用域：", "",pose_table("tilt_le30"), "",
        "全部压力样本（含超出30°先验适用域的倾角）单列，不能将这些输出成功等同于物理位姿正确：", "",pose_table("all"), "",
        "法向突跳：相邻有效帧法向变化>20°，且超出真实变化>15°。深度突跳：相邻帧 t.z 变化>30%，且超出真实相对变化>20个百分点。中间有失败帧的跳变另有原始记录，未混入相邻帧指标。", "",
        "## class1 默认隔离的依据", "",
        "全类别实验保留在 `output/mono_identity_final_20261016`。reference 序列原版深度突跳为0，而 class0/class1 都启用新编号时为12。BC 拒绝后切换 TT，当前 class1 两套物理尺寸的差异会放大为深度变化；匹配清楚与物理尺度一致是两个独立条件。该实验结果没有被隐藏或作为默认配置推广。", "",
        "![三种配置深度对比](../output/mono_identity_mono_20261016/figures/depth_comparison.png)", "",
        "## 成对耗时", "",
        "在其他实验结束后测量：同一已解码 ROI，旧/新提取器交替执行，先预热，OpenCV 单线程；22个 ROI、每种8次，共352条计时观测，不计图像读取、YOLO和可视化。class1 按默认旧链路处理。", "",
        "| 类别 | 模式 | 样本 | 中位数 | P95 | 均值 |","|---|---|---:|---:|---:|---:|"]
    for s in time_summary:lines.append("| %d | %s | %d | %.3f ms | %.3f ms | %.3f ms |"%(s["cls"],s["mode"],s["n"],s["median"],s["p95"],s["mean"]))
    lines += ["", "这是编号可靠性改进；当前保留原 BC 前处理并增加验证，有额外计算开销。并行实验中的整机 FPS 不用于推断新模块速度收益。", "",
        "## 验证与可视化", "",
        "- CTest 三个目标通过：新增9组身份匹配测试，既有7组位姿先验和16组位姿求解测试，共32组。",
        "- 新测试覆盖循环起点、轮廓反向、透视、四向快速分支、错误图案、对称歧义、少数坏点支持、RNG状态保持、内图案尺度保护、class1默认隔离以及双目上下文不变。",
        "- Debug 模式新增 mono_identity*.png：标准前景、TT粗归一化、精化H和固定物理编号，以及接受/拒绝状态。", "",
        "![实际单目编号面板](../output/mono_identity_mono_20261016/debug_reference_f22/mono_identity_f1.png)", "",
        "结果：`output/mono_identity_mono_20261016`（默认单目链路、计时、回归）与 `output/mono_identity_final_20261016`（完整角点消融及class1风险实验）。原始坐标、逐帧测量、分组统计及可执行文件保留在相应目录。", "",
        "## 限制与后续", "",
        "TT 四点是图像外接矩形，无法独自消除板面透视；因此完整 H 后备与图案验证仍有必要。本次重点修复编号，未提升灰度角点定位精度，也未保证消除平面 PnP 二义性。class1 新编号仅供独立实验，正式推广前应统一两套物理尺寸并完善其轮廓点质量和退化策略。", ""]
    Path("docs/mono_corner_identity_20261016.md").write_text("\n".join(lines),encoding="utf8")
    print("Report saved;",len(timing),"paired timing observations")

if __name__=="__main__":main()
