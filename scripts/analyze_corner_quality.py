#!/usr/bin/env python3
"""Run prepare/analyze for unchanged C++ BC/TT vs synthetic annotation GT.

Uses oracle bounding-box ROIs to isolate extraction from YOLO and PnP.
BC uses immutable annotation IDs; TT uses projected SOURCE bbox vertices,
not a minAreaRect fitted after projection. Assignment is diagnostic only.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points(path):
    return np.array([list(map(float, m)) for m in re.findall(
        r"Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)",
        Path(path).read_text(encoding="utf-8-sig"))])


def project(H, p):
    q = np.c_[p, np.ones(len(p))] @ np.array(H).T
    return q[:, :2] / q[:, 2:]


def size_bin(size):
    if size < 20: return "<20"
    if size < 50: return "20-50"
    if size < 100: return "50-100"
    if size < 300: return "100-300"
    return ">=300"


def tilt_bin(tilt):
    if tilt <= 5: return "0-5"
    if tilt <= 30: return "5-30"
    if tilt <= 50: return "30-50"
    return ">50"


def prepare(args):
    out, dataset = Path(args.out), Path(args.dataset)
    if (out / "jobs.json").exists():
        raise ValueError("Refusing to overwrite an existing experiment")
    out.mkdir(parents=True, exist_ok=True)
    cfg = read(args.config)
    dump(out / "config_snapshot.json", cfg)
    source = [points("scripts/class%d_points.txt" % c) for c in (0, 1)]
    template = points(Path(cfg["strategies"]["binary_corner"]["template_dir"])/"0_degrees.txt")
    # Confirm the existing 0-degree template and hand annotation have the same
    # immutable IDs; never fit/relabel the observed image to reduce the error.
    mapping = []
    for src in source:
        a = (src-src.min(0))/np.ptp(src, axis=0)
        b = (template-template.min(0))/np.ptp(template, axis=0)
        cost = np.linalg.norm(b[:, None]-a[None, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        if not np.array_equal(cols, np.arange(10)):
            raise ValueError("Template order differs from canonical annotation IDs")
        mapping.append(dict(template_to_annotation=cols.tolist(),
                            max_normalized_deviation=float(cost[rows, cols].max())))
    jobs, lines, manifests = [], [], {}
    max_projection_error = 0.
    suite = read(dataset / "suite_manifest.json")
    for profile in suite["profiles"]:
        sequence = profile["name"]
        manifest_path = dataset / profile["manifest"]
        manifests[sequence] = sha(manifest_path)
        manifest = read(manifest_path)
        width, height = manifest["config"]["canvas"]
        for frame in manifest["frames"]:
            for cls in (0, 1):
                p = np.array(frame["class%d" % cls]["points_px"])
                projection_error = float(np.linalg.norm(project(frame["H"], source[cls])-p, axis=1).max())
                max_projection_error = max(max_projection_error, projection_error)
                if projection_error > 1e-7:
                    raise ValueError("Annotation/source homography inconsistency")
                lo, hi = p.min(0), p.max(0)
                src = source[cls]
                sl, sh = src.min(0), src.max(0)
                square = np.array([[sl[0], sl[1]], [sh[0], sl[1]],
                                   [sh[0], sh[1]], [sl[0], sh[1]]])
                for method, strategy in enumerate(("BC", "TT")):
                    pad = cfg["strategies"]["binary_corner" if method == 0 else "tiny_target"]["roi_pad_pixels"]
                    start = np.maximum(0, np.floor(lo).astype(int)-pad)
                    end = np.minimum([width, height], np.ceil(hi).astype(int)+pad+1)
                    if np.any(end-start < 2):
                        continue
                    gt = p if method == 0 else project(frame["H"], square)
                    visible = bool(all(frame["class%d" % cls]["visible"]))
                    visible &= bool(np.all((gt >= 0) & (gt < [width, height])))
                    occ = frame["degradation"].get("occluder")
                    if occ:
                        x1, y1, x2, y2 = occ
                        visible &= not bool(np.any((gt[:, 0] >= x1) & (gt[:, 0] <= x2) &
                                                   (gt[:, 1] >= y1) & (gt[:, 1] <= y2)))
                    roi = [*start.tolist(), *(end-start).tolist()]
                    job = dict(job=len(jobs), sequence=sequence, frame=frame["frame"],
                               path=(dataset/sequence/frame["path"]).as_posix(),
                               cls=cls, strategy=strategy, roi=roi, gt=gt.tolist(),
                               fully_visible=visible, size=float(max(hi-lo)),
                               tilt=frame["tilt_deg"])
                    jobs.append(job)
                    lines.append("%d %s %d %d %d %d %d %d" % (
                        job["job"], job["path"], cls, method, *roi))
    dump(out/"jobs.json", jobs)
    (out/"jobs.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    dump(out/"experiment.json", dict(dataset=dataset.as_posix(), images=1020,
        jobs=len(jobs), config_sha256=sha(args.config), manifest_sha256=manifests,
        fixed_id_mapping=mapping, max_annotation_projection_error_px=max_projection_error,
        roi="annotation bbox floor/ceil inclusive plus configured BC/TT padding, clipped to image",
        bc_truth="projected 10 manually annotated contour vertices with fixed template IDs",
        tt_truth="projected 4 SOURCE annotation bbox vertices TL/TR/BR/BL, virtual corners",
        scope="isolated forced extraction; no YOLO, strategy selection, fallback, PnP, or pose prior"))
    print("Prepared %d extraction jobs" % len(jobs))


def stats(values):
    a = np.array(values, dtype=float)
    if not len(a): return None
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), rmse=float(np.sqrt(np.mean(a*a))),
                max=float(a.max()), within1=float(np.mean(a <= 1)),
                within2=float(np.mean(a <= 2)))


def analyze(args):
    out = Path(args.out)
    jobs = read(out/"jobs.json")
    experiment = read(out/"experiment.json")
    dataset = Path(experiment["dataset"])
    truth = {}
    max_projection_error = 0.
    source = [points("scripts/class%d_points.txt" % c) for c in (0, 1)]
    for profile in read(dataset/"suite_manifest.json")["profiles"]:
        for f in read(dataset/profile["manifest"])["frames"]:
            for cls in (0, 1):
                p = np.array(f["class%d" % cls]["points_px"])
                truth[(profile["name"],f["frame"],cls)] = p
                error = float(np.linalg.norm(project(f["H"],source[cls])-p,axis=1).max())
                max_projection_error = max(max_projection_error,error)
                if error > 1e-7: raise ValueError("Source/annotation projection mismatch")
    raw = list(csv.DictReader((out/"detections.csv").open(encoding="utf-8")))
    if len(raw) != len(jobs): raise ValueError("Missing detection rows")
    groups = defaultdict(list)
    frames, point_rows = [], []
    for j, row in zip(jobs, raw):
        if int(row["job"]) != j["job"]: raise ValueError("Misaligned jobs")
        n = int(row["n"])
        gt = np.array(j["gt"])
        pred = np.array([[float(row["x%d" % i]), float(row["y%d" % i])] for i in range(n)])
        ok = row["success"] == "1" and n == len(gt) and bool(np.isfinite(pred).all())
        item = dict(j, output_ok=ok, n=n, pred=pred.tolist(),
                    size_bin=size_bin(j["size"]), tilt_bin=tilt_bin(j["tilt"]))
        if j["strategy"] == "TT":
            contour=truth[(j["sequence"],j["frame"],j["cls"])].astype(np.float32)
            rectangle=cv2.boxPoints(cv2.minAreaRect(contour))
            rect_cost=np.linalg.norm(rectangle[:,None]-gt[None,:],axis=2)
            ri,ci=linear_sum_assignment(rect_cost**2)
            item["ideal_rectangle_error"] = rect_cost[ri,ci].tolist()
        if ok:
            cost = np.linalg.norm(pred[:, None]-gt[None, :], axis=2)
            ri, ci = linear_sum_assignment(cost*cost)
            direct, assigned = np.diag(cost), cost[ri, ci]
            direct_rmse, best_rmse = np.sqrt(np.mean(direct**2)), np.sqrt(np.mean(assigned**2))
            if best_rmse > direct_rmse+1e-8:
                raise ValueError("Assignment error exceeds fixed-ID error")
            # A diagnostic signature of index/rotation mismatch, not an oracle
            # repair applied to extraction output and not proof of true IDs.
            mismatch = bool(not np.array_equal(ci, np.arange(n)) and
                            direct_rmse-best_rmse > max(1., .01*j["size"]))
            item.update(direct=direct.tolist(), assigned=assigned.tolist(),
                        permutation=ci.tolist(), mismatch=mismatch,
                        direct_rmse=float(direct_rmse), best_rmse=float(best_rmse))
            for i in range(n):
                point_rows.append(dict(job=j["job"],sequence=j["sequence"],frame=j["frame"],
                    cls=j["cls"],strategy=j["strategy"],fully_visible=j["fully_visible"],
                    point_id=i,x=pred[i,0],y=pred[i,1],gt_x=gt[i,0],gt_y=gt[i,1],
                    error_px=direct[i],assigned_gt_id=ci[i],assigned_error_px=assigned[i]))
        frames.append(item)
        visibility = "visible" if j["fully_visible"] else "partial_or_occluded"
        prefix = (j["cls"], j["strategy"], visibility)
        groups[prefix+("all", "all")].append(item)
        groups[prefix+("sequence", j["sequence"])].append(item)
        groups[prefix+("size", item["size_bin"])].append(item)
        groups[prefix+("tilt", item["tilt_bin"])].append(item)
    summaries = []
    for key, items in sorted(groups.items()):
        valid = [x for x in items if x["output_ok"]]
        direct = [e for x in valid for e in x["direct"]]
        assigned = [e for x in valid for e in x["assigned"]]
        aligned = [x for x in valid if not x["mismatch"]]
        summaries.append(dict(cls=key[0],strategy=key[1],visibility=key[2],
            dimension=key[3],group=key[4],attempts=len(items),outputs=len(valid),
            output_rate=len(valid)/len(items),mismatch_frames=sum(x["mismatch"] for x in valid),
            direct=stats(direct),assigned=stats(assigned),
            direct_percent_of_size=stats([100*e/x["size"] for x in valid for e in x["direct"]]),
            assigned_percent_of_size=stats([100*e/x["size"] for x in valid for e in x["assigned"]]),
            no_mismatch_direct=stats([e for x in aligned for e in x["direct"]]),
            ideal_rectangle_error=stats([e for x in items for e in x.get("ideal_rectangle_error",[])]),
            frame_all_within2=sum(max(x["direct"]) <= 2 for x in valid),
            frame_all_within2_rate=sum(max(x["direct"]) <= 2 for x in valid)/len(items)))
    dump(out/"frame_metrics.json",frames)
    dump(out/"summary.json",summaries)
    if point_rows:
        with (out/"point_errors.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(point_rows[0])); writer.writeheader(); writer.writerows(point_rows)
    flat=[]
    for s in summaries:
        names=("direct","assigned","no_mismatch_direct","direct_percent_of_size","assigned_percent_of_size","ideal_rectangle_error")
        r={k:v for k,v in s.items() if k not in names}
        for name in names:
            if s[name]: r.update({name+"_"+k:v for k,v in s[name].items()})
        flat.append(r)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r})); writer.writeheader(); writer.writerows(flat)
    make_figures(out,frames)
    dump(out/"verification.json",dict(expected_jobs=len(jobs),observed_rows=len(raw),
        max_annotation_projection_error_px=max_projection_error,
        source_config_preserved=sha(args.config)==experiment["config_sha256"],
        sources_sha256={p:sha(p) for p in ["scripts/evaluate_corner_extraction.cpp",
            "scripts/analyze_corner_quality.py","src/feature/BinaryCornerExtractor.cpp",
            "src/feature/TinyTargetExtractor.cpp","scripts/class0_points.txt","scripts/class1_points.txt"]},
        rows_unique=len({int(r["job"]) for r in raw})==len(jobs),
        no_nonfinite_success=all(np.isfinite(x["direct"]).all() for x in frames if x["output_ok"])))
    write_report(out,summaries,frames,experiment,max_projection_error)
    print(json.dumps([s for s in summaries if s["dimension"]=="all"],ensure_ascii=False,indent=2))


def write_report(out,summaries,frames,experiment,projection_error):
    def table(items):
        lines=["| 类别 | 方法 | 分组 | 尝试/有效输出帧 | 固定编号中位数 / P95 (px) | 最优一一匹配中位数 / P95 (px) | 对应偏移诊断帧 | 全部点≤2px帧率 |",
               "|---|---|---|---:|---:|---:|---:|---:|"]
        for s in items:
            def fmt(v): return "—" if not v else "%.2f / %.2f" % (v["median"],v["p95"])
            lines.append("| %d | %s | %s | %d / %d | %s | %s | %d | %.1f%% |" % (
                s["cls"],s["strategy"],s["group"],s["attempts"],s["outputs"],fmt(s["direct"]),
                fmt(s["assigned"]),s["mismatch_frames"],100*s["frame_all_within2_rate"]))
        return "\n".join(lines)
    visible=[s for s in summaries if s["visibility"]=="visible"]
    lines=["# BC / TT 角点提取质量实验（2026-10-16）", "",
        "本实验对已有合成数据的角点进行直接评估，未修改 BC、TT、PnP 或主配置。测量对象是角点与标注的像素距离，不是重投影残差或位姿误差。", "",
        "## 数据与范围", "",
        "- 数据：`tests/data/mono_robustness_20261015`，6 组共 1020 张 1744×1744 图像；包含 reference、oblique_clean、grazing、normal_ambiguity、fast_rotation、oblique_degraded。",
        "- 靶图：`data/clean_target2.png`；class 0、class 1 各有 10 个源图人工标注角点。",
        "- 每张图、每个类别分别强制调用真实 C++ `extractMono()` 的 BC、TT，总计 %d 次；不是原跟踪策略自动选档后的输出。" % len(frames),
        "- 使用真值角点包围框，floor/ceil 并包含端点，按配置外扩 BC 3 px / TT 5 px，裁到图像内；排除了 YOLO 检测框误差。配置副本见实验目录。",
        "- 不经过 PnP、时序门控、Dual-ROI 合并或退化链。因此不能把本实验输出率当作整机跟踪成功率，也不能直接与原位姿实验逐帧归因。",
        "- 完整可见与出画面/遮挡样本分开统计。下面主表仅用完整可见样本，失败帧计入输出率和全点达标帧率分母；像素误差仅统计输出点数正确且有限的帧。",
        "- 各尺寸均强制运行两种方法用于诊断。TT 大尺寸结果不代表应在该尺寸启用 TT，BC 极小尺寸也同理。", "",
        "## 真值、编号与误差定义", "",
        "BC 真值为 manifest 中 10 个投影轮廓角点，编号与 0° 模板逐一核对一致；检测点不允许逐帧重新编号来降低主指标。",
        "TT 真值为源图该类别标注包围框的 TL/TR/BR/BL 四个虚拟顶点，再通过每帧 H 做透视投影。这四点通常不是原图案的四个实角；TT 数值表示四点矩形近似输出与该参照模型的偏差，不能与 BC 实角定位误差混称。",
        "本实验使用图案标注的图像几何参照，没有将配置中独立设定的 BC/TT 物理尺寸重新投影作为主真值，避免把尺寸不一致混入提取定位误差。因此结果不等同于完整 2D–3D 模型一致性评估。",
        "源图标注重新经 H 投影，与 manifest 最大差异 %.3g px。人工标注本身有不确定性；合成透视映射不使人工标注成为无限精确的边界真值。" % projection_error,
        "", "设检测点 p_i，固定编号真值 g_i，误差 e_i=||p_i-g_i||₂，单位均为原图像素。中位数、P95、均值及 RMSE 按点统计；RMSE=sqrt(mean(e_i²))。", "",
        "同时给出 Hungarian 最优一一分配（最小化距离平方和）后的诊断误差，防止多个检测点匹配同一真值。它仅表示忽略编号后的点集位置质量，不是可用于 PnP 的精度保证，也没有修正实际输出。",
        "若最优分配改变编号，且固定编号 RMSE 比分配后 RMSE 高出 max(1 px, 目标跨度×1%)，记为‘对应偏移诊断帧’。该指标提示起点/方向错位或形状失败；不是对真实编号错误的人工审定。",
        "全点≤2px帧率：固定编号下所有输出点误差≤2px的帧数 / 该分组全部尝试帧数。输出点数正确仅称‘有效输出’，不等于角点正确。", "",
        "## 完整可见样本总览", "", table([s for s in visible if s["dimension"]=="all"]), "",
        "## class 0 按序列", "", table([s for s in visible if s["dimension"]=="sequence" and s["cls"]==0]), "",
        "## 按目标尺寸", "",
        "尺寸为投影的 10 个标注点轴对齐包围框最长边，单位 px；并非面积阈值，也未复现策略粘滞。区间为左闭右开。", "",
        table([s for s in visible if s["dimension"]=="size"]), "",
        "## class 0 按总倾角", "", table([s for s in visible if s["dimension"]=="tilt" and s["cls"]==0]), "",
        "## TT 的理想矩形几何偏差", "",
        "独立对真值 10 点轮廓求 minAreaRect，不加检测噪声，再将其四顶点与投影源框四点作最优一一分配。这里排除了编号偏差，用于量化矩形近似的几何限制；实际 TT 还包含图像阈值、轮廓与精化偏差。", "",
        "| 类别 | 倾角 | 理想矩形误差中位数 / P95 (px) | 实际 TT 最优分配中位数 / P95 (px) |",
        "|---|---|---:|---:|"]
    for s in visible:
        if s["strategy"]!="TT" or s["dimension"]!="tilt": continue
        a,b=s["ideal_rectangle_error"],s["assigned"]
        if a and b: lines.append("| %d | %s | %.2f / %.2f | %.2f / %.2f |" % (
            s["cls"],s["group"],a["median"],a["p95"],b["median"],b["p95"]))
    lines += ["", "## 遮挡或出画面样本", "",
        "这些帧可能缺少真实角点；返回固定点数可能包含补点或错误轮廓，单独列出以避免掩盖完整可见条件下的精度。", "",
        table([s for s in summaries if s["visibility"]=="partial_or_occluded" and s["dimension"]=="all"]), "",
        "## 可视化与复现", "",
        "![固定编号角点对比](../%s/figures/examples.png)" % out.as_posix(), "",
        "绿色加号为真值，橙红圆圈为检测点，黄色线按固定编号连接。每格标题给出帧、尺寸、倾角与固定编号 RMSE。reference BC、near-frontal TT 示例按候选帧 RMSE 中位数选择，其余为对应分组最坏帧，不代表总体平均。", "",
        "![角点误差分布](../%s/figures/error_cdf.png)" % out.as_posix(), "",
        "结果目录：`%s`。`summary.csv/json` 包含均值、中位数、P95、RMSE、≤1/2px比例及按跨度归一化误差；`point_errors.csv` 保存逐点原坐标、真值、误差与诊断分配；`frame_metrics.json` 保存逐帧输入范围与预测；`verification.json` 保存输入一致性及源码哈希。" % out.as_posix(), "",
        "```text", "C:/Develop/Conda/envs/DCL/python.exe scripts/analyze_corner_quality.py prepare",
        "scripts/docker-toolchain.sh bash scripts/run_corner_extraction.sh",
        "C:/Develop/Conda/envs/DCL/python.exe scripts/analyze_corner_quality.py analyze", "```", "",
        "重复实验应使用新的 `--out`，或直接分析本次已有输出；prepare 拒绝覆盖已有 jobs。", "",
        "## 改进方向与解释边界", "",
        "1. 对应偏移明显的 BC 帧，应先检查轮廓起点、遍历方向、角度模板与固定编号的联合几何约束。位置精化无法修复错误的 2D–3D 编号。",
        "2. 点集几何误差仍较大的 BC 帧，应检查选域、形态学圆角及强制补成 10 点的影响，再采用原灰度边缘拟合/受限亚像素精化；加入逐点质量与形状一致性拒绝条件。",
        "3. TT 在透视下的虚拟矩形与靶板投影不同，应先限定使用条件与可信度，避免将四点数量齐全当作物理模型正确；若要提高大倾角性能，需要能描述透视的模板轮廓或真实特征点模型。",
        "4. 小到无法分辨内凹结构的 class 1 样本需要最小可辨尺寸门限。不能把形态学处理后凑齐点数当作 10 个物理角点被测到。",
        "5. 下一轮端到端验证应记录真实 YOLO ROI 下各级策略的提取点、物理编号与 PnP 输入，再关联位姿跳变。本次没有调整提取算法，也未据此宣称已解决位姿二义性。", ""]
    def get(cls,strategy,dimension,group):
        return next(s for s in visible if (s["cls"],s["strategy"],s["dimension"],s["group"]) ==
                    (cls,strategy,dimension,group))
    b=get(0,"BC","all","all")
    t=get(0,"TT","size","20-50")
    c=get(1,"BC","size","<20")
    h=get(0,"TT","tilt",">50")
    example=next(x for x in frames if x["cls"]==0 and x["strategy"]=="BC" and
                 x["sequence"]=="oblique_degraded" and x["frame"]==172)
    lines[4:4] = ["## 结论摘要", "",
        "- class 0 BC 在完整可见的 %d 次尝试中有 %d 次返回 10 点，但 %d 帧（占有效输出 %.1f%%）满足对应偏移诊断条件。固定编号中位数 %.2f px、P95 %.2f px；忽略编号后的点集诊断中位数 %.2f px、P95 %.2f px。当前突出问题是部分帧的编号/起点错位，而不是所有角点都定位不准。" % (
            b["attempts"],b["outputs"],b["mismatch_frames"],100*b["mismatch_frames"]/b["outputs"],
            b["direct"]["median"],b["direct"]["p95"],b["assigned"]["median"],b["assigned"]["p95"]),
        "- 实例：oblique_degraded 第 172 帧，固定编号 RMSE %.2f px，最优分配后 %.2f px；诊断对应序列为 %s，表现为轮廓编号循环移位。该例与图中角点位置接近、连线却跨越整个目标的现象一致。" % (
            example["direct_rmse"],example["best_rmse"],example["permutation"]),
        "- class 0 TT 在跨度 20–50 px 的 %d 帧中，固定编号误差中位数 %.2f px、P95 %.2f px，全部四点≤2px的帧率 %.1f%%。这是混合倾角的小尺寸分组，不是完整实飞精度保证。" % (
            t["attempts"],t["direct"]["median"],t["direct"]["p95"],100*t["frame_all_within2_rate"]),
        "- class 0 总倾角>50°时，TT 理想矩形近似 P95 %.2f px，实际 TT 忽略编号后 P95 %.2f px。两者接近支持矩形模型在该场景下存在主要几何限制；这部分偏差不能仅靠亚像素细化消除。" % (
            h["ideal_rectangle_error"]["p95"],h["assigned"]["p95"]),
        "- class 1 跨度<20 px 的 %d 帧中，BC 仅 %d 帧返回完整 10 点（%.1f%%）；输出存在也不保证正确。小内图案需要可辨尺寸和点质量门限。" % (
            c["attempts"],c["outputs"],100*c["output_rate"]),
        "- 优先级建议：先修复/拒绝 BC 编号错位，再改善灰度角点位置；TT 先处理透视模型与启用条件，class 1 增加可辨性门限。当前结果不能证明原位姿跳变均由角点问题造成，PnP 二义性仍应独立评估。", ""]
    Path("docs/corner_quality_20261016.md").write_text("\n".join(lines),encoding="utf-8")


def make_figures(out,frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images=out/"figures"; images.mkdir(exist_ok=True)
    # Fixed representative conditions; select median/worst by stated metric.
    choices=[]
    for strategy, sequence, maxsize, worst in [
        ("BC","reference",300,False), ("BC","grazing",10000,True),
        ("BC","oblique_degraded",10000,True), ("TT","normal_ambiguity",50,False),
        ("TT","grazing",10000,True), ("TT","oblique_degraded",10000,True)]:
        candidates=[x for x in frames if x["cls"]==0 and x["strategy"]==strategy and
                    x["sequence"]==sequence and x["fully_visible"] and x["output_ok"] and x["size"]<=maxsize]
        if not candidates: continue
        candidates.sort(key=lambda x:x["direct_rmse"])
        choices.append(candidates[-1 if worst else len(candidates)//2])
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    for ax,item in zip(axes.flat,choices):
        img=cv2.imread(item["path"]); gt=np.array(item["gt"]); pred=np.array(item["pred"])
        lo=np.floor(np.minimum(gt.min(0),pred.min(0))-8).astype(int)
        hi=np.ceil(np.maximum(gt.max(0),pred.max(0))+8).astype(int)
        lo=np.maximum(lo,0); hi=np.minimum(hi,[img.shape[1],img.shape[0]])
        ax.imshow(cv2.cvtColor(img[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB),
                  extent=(lo[0],hi[0],hi[1],lo[1]))
        ax.scatter(gt[:,0],gt[:,1],c="lime",marker="+",s=65,label="GT")
        ax.scatter(pred[:,0],pred[:,1],facecolors="none",edgecolors="orangered",s=40,label="Detected")
        for i,(g,p) in enumerate(zip(gt,pred)):
            ax.plot([g[0],p[0]],[g[1],p[1]],color="yellow",lw=.7)
            ax.annotate(str(i),g,color="lime",fontsize=8)
        ax.set_title("%s %s f%d\nsize %.1fpx tilt %.1fdeg RMSE %.2fpx" % (
            item["strategy"],item["sequence"],item["frame"],item["size"],item["tilt"],item["direct_rmse"]),fontsize=10)
        ax.legend(fontsize=7,loc="best")
    fig.tight_layout(); fig.savefig(str(images/"examples.png"),dpi=160); plt.close(fig)
    dump(images/"examples.json",[{k:x[k] for k in ("job","sequence","frame","strategy","direct_rmse")} for x in choices])
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for strategy,color in [("BC","tab:blue"),("TT","tab:orange")]:
        selected=[x for x in frames if x["cls"]==0 and x["fully_visible"] and x["output_ok"]]
        selected=[x for x in selected if x["strategy"]==strategy]
        for ax,metric,label in zip(axes,("direct","assigned"),("Fixed-ID error","Assignment diagnostic error")):
            err=np.sort([e for x in selected for e in x[metric]])
            ax.plot(err,np.arange(1,len(err)+1)/len(err),label=strategy,color=color)
            ax.set_xscale("log"); ax.set_xlabel(label+" (original image px)"); ax.set_ylabel("Point CDF")
            ax.grid(alpha=.3); ax.legend(); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(str(images/"error_cdf.png"),dpi=160); plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["prepare","analyze"])
    p.add_argument("--dataset",default="tests/data/mono_robustness_20261015")
    p.add_argument("--config",default="config/tracker_config.json")
    p.add_argument("--out",default="output/corner_quality_20261016")
    args=p.parse_args()
    (prepare if args.action=="prepare" else analyze)(args)
