"""Analyze an imported single SSH run without changing its raw CSV/log files.

Truth is joined by manifest order (data frame), not source filename numbers.
Plots use ReportLab charts; PDFium rasterizes the same PDF for Markdown.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from analyze_stereo_mixed_strategies import family, stats, summarize, write_table


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(raw, manifest):
    truth = json.loads(manifest.read_text(encoding="utf-8-sig"))
    text = (raw / "tracking_log.txt").read_text(encoding="utf-8-sig")
    status, table = {}, {}
    for line in text.splitlines():
        if line.startswith("[StereoFrontend]"):
            v = dict(re.findall(r"([a-z_0-9]+)=([^ ]+)", line))
            v["route"] = line.split(" route=", 1)[-1]
            index = int(v["frame"])
            assert index not in status, f"Duplicate log frame {index}"
            status[index] = v
        if line.startswith("|"):
            c = [p.strip() for p in line.strip("|").split("|")]
            if c[0].isdigit():
                index = int(c[0])
                assert index not in table, f"Duplicate table frame {index}"
                table[index] = {"left_kp": int(c[4]), "pairs": int(c[5]),
                                "fallback": int(c[3] == "Y"), "pnp": c[10]}
    expected = set(range(1, len(truth["frames"]) + 1))
    found = {int(p.stem.split("f")[-1]) for p in raw.glob("disparity_f*.csv")}
    assert found == expected == set(status) == set(table), "Incomplete or extra frame IDs"
    frames, points = [], []
    for index, gt in enumerate(truth["frames"], 1):
        s, meta = status[index], table[index]
        with (raw / f"disparity_f{index}.csv").open(encoding="utf-8-sig") as stream:
            pairs = list(csv.DictReader(stream))
        assert len(pairs) == int(s["pairs"]) == meta["pairs"]
        assert len({p["left_index"] for p in pairs}) == len(pairs)
        assert s["disparity_only"] == "1" and s["pnp"] == meta["pnp"] == "SKIP"
        assert s["source"] == "Unspecified" and s["stereo"] == "0"
        strategy = pairs[0]["strategy"] if pairs else "Unknown"
        group = family(strategy)
        entry = "Dual-ROI" if s["route"].startswith("C0:") else (
            "BC" if re.match(r"C[01]BC:", s["route"]) else "TT" if re.match(r"C[01]TT:", s["route"]) else "Unknown")
        assert entry == group, f"Unexpected fallback/strategy frame {index}"
        current = []
        for p in pairs:
            assert int(p["frame_match_ok"]) == int(s["match_ok"]) and p["strategy"] == strategy
            row = {"frame": index, "source": gt["src"], "entry_strategy": entry,
                   "final_strategy": strategy, "frame_ok": int(s["match_ok"])}
            for key in ("left_index", "region", "original_index", "template_index"):
                row[key] = int(p[key])
            for key in ("left_x", "left_y", "right_x", "right_y", "disparity_px"):
                row[key] = float(p[key])
                assert math.isfinite(row[key])
            row.update(model_frame=p["model_frame"], truth_disparity_px=gt["disparity"],
                       truth_dy_px=gt["dy_px"], disparity_error_px=row["disparity_px"]-gt["disparity"],
                       vertical_error_px=row["right_y"]-row["left_y"]-gt["dy_px"])
            # CSV coordinates are rounded separately; allow their documented rounding.
            assert abs(row["left_x"]-row["right_x"]-row["disparity_px"]) < .02
            current.append(row)
        assert collections.Counter(p["region"] for p in current) == collections.Counter(
            {k: v for k, v in [(0, int(s["c0"])), (1, int(s["c1"]))] if v})
        assert len(pairs) <= meta["left_kp"]
        points.extend(current)
        frames.append({"frame": index, "source": gt["src"], "entry_strategy": entry,
                       "final_strategy": strategy, "ok": int(s["match_ok"]), "pairs": len(pairs), **meta,
                       "truth_disparity_px": gt["disparity"], "truth_dy_px": gt["dy_px"],
                       "median_disparity_px": statistics.median(p["disparity_px"] for p in current) if current else None,
                       "disparity_mae_px": statistics.mean(abs(p["disparity_error_px"]) for p in current) if current else None,
                       "max_disparity_error_px": max((abs(p["disparity_error_px"]) for p in current), default=None),
                       "match_ms": float(s["match_ms"]), "planar_ms": float(s["planar_ms"]),
                       "planar_status": s["planar"], "planar_rejected": int(s["planar_rejected"]),
                       "coarse_attempts": int(s["coarse"]), "pyramids": int(s["pyramids"]),
                       "full_retry": int(s["full_retry"]), "legacy": int(s["legacy"]),
                       "c0_pairs": int(s["c0"]), "c1_pairs": int(s["c1"]), "route": s["route"]})
    labels = {"fps": "实际处理 FPS", "yolo_ms": "YOLO 平均用时",
              "tracker_ms": "单帧跟踪平均用时 \\(提取\\+解算\\+ROI等\\)",
              "extract_match_ms": "特征提取平均用时", "pnp_ms": "位姿解算平均用时"}
    runtime = {}
    for key, label in labels.items():
        matches = re.findall(label + r": ([0-9.eE+-]+)", text)
        assert len(matches) == 1, (key, matches)
        runtime[key] = float(matches[0])
    calls = set(re.findall(r"YOLO推理次数: left=(\d+) right=(\d+)", text))
    assert len(calls) == 1
    runtime["left_yolo"], runtime["right_yolo"] = map(int, next(iter(calls)))
    assert runtime["left_yolo"] == len(frames) and runtime["right_yolo"] == 0 and runtime["pnp_ms"] == 0
    def group_summary(fs, ps):
        result = summarize(fs, ps, fs)
        result.update(raw_left_points=sum(f["left_kp"] for f in fs),
                      point_retention=len(ps)/sum(f["left_kp"] for f in fs),
                      planar_rejected=sum(f["planar_rejected"] for f in fs),
                      planar_ms=stats([f["planar_ms"] for f in fs]))
        return result
    groups = {name: group_summary([f for f in frames if f["entry_strategy"] == name],
                                 [p for p in points if p["entry_strategy"] == name]) for name in ("Dual-ROI", "BC", "TT")}
    transitions = []
    for f in frames:
        if not transitions or transitions[-1]["strategy"] != f["final_strategy"]:
            transitions.append({"frame": f["frame"], "strategy": f["final_strategy"]})
    summary = {"run_count": 1, "warmup": None, "repeated_stability": None,
               "overall": group_summary(frames, points), "by_entry_strategy": groups,
               "runtime": runtime, "transitions": transitions,
               "planar_status_counts": dict(collections.Counter(f["planar_status"] for f in frames)),
               "planar_rejection_frames": [f["frame"] for f in frames if f["planar_rejected"]],
               "points_within_0p05_px": sum(abs(p["disparity_error_px"]) <= .05 for p in points),
               "max_error_point": max(points, key=lambda p: abs(p["disparity_error_px"])),
               "reduced_point_frames": [{"frame": f["frame"], "pairs": f["pairs"], "raw": f["left_kp"],
                                         "planar_rejected": f["planar_rejected"]} for f in frames if f["pairs"] < f["left_kp"]]}
    return summary, frames, points


def compare(raw, local):
    rows = lambda folder: {(i, r["left_index"]): r for i in range(1, 151)
                          for r in csv.DictReader((folder/f"disparity_f{i}.csv").open(encoding="utf-8-sig"))}
    a, b = rows(local), rows(raw)
    common = a.keys() & b.keys()
    numeric = ("left_x", "left_y", "right_x", "right_y", "disparity_px")
    categorical = ("left_index", "region", "original_index", "template_index", "model_frame", "strategy", "frame_match_ok")
    return {"local_reference": local.as_posix(), "common_points": len(common),
            "identical_rows": sum(a[k] == b[k] for k in common),
            "categorical_differences": sum(any(a[k][c] != b[k][c] for c in categorical) for k in common),
            "removed_keys": sorted(a.keys()-b.keys()), "added_keys": sorted(b.keys()-a.keys()),
            "max_abs_numeric_delta": {c: max(abs(float(a[k][c])-float(b[k][c])) for k in common) for c in numeric}}


def plots(assets, summary, frames, points):
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.graphics import renderPDF, renderSVG
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    import pypdfium2 as pdfium
    pdfmetrics.registerFont(TTFont("Chinese", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
    colors = [HexColor(c) for c in ("#2563eb", "#059669", "#d97706")]
    names = ("Dual-ROI", "BC", "TT")
    def title(d, x, y, text, size=13, color="#17253b"):
        d.add(String(x, y, text, fontName="Chinese", fontSize=size, fillColor=HexColor(color)))
    def base(height, label):
        d = Drawing(1040, height)
        d.add(Rect(0, 0, 1040, height, fillColor=HexColor("#ffffff"), strokeColor=None))
        title(d, 40, height-40, label, 23)
        title(d, 40, height-65, "SSH 单轮 150 帧；仅视差，PnP 未运行；真值来自同一数据集 manifest", 12)
        return d
    def line(d, x, y, title_text, series, palette, ylim, ylabel, xmax=150, xmin=1):
        title(d, x, y+238, title_text, 14)
        title(d, x, y+217, ylabel, 10, "#52617b")
        p = LinePlot()
        p.x, p.y, p.width, p.height = x+42, y+30, 404, 174
        p.data = series
        p.joinedLines = 1
        p.xValueAxis.valueMin, p.xValueAxis.valueMax = xmin, xmax
        p.xValueAxis.valueSteps = [1, 25, 50, 75, 100, 125, 150] if xmax == 150 else [0,.01,.02,.03,.04,.05]
        p.yValueAxis.valueMin, p.yValueAxis.valueMax = ylim
        p.xValueAxis.labels.fontName = p.yValueAxis.labels.fontName = "Chinese"
        p.xValueAxis.labels.fontSize = p.yValueAxis.labels.fontSize = 9
        p.yValueAxis.visibleGrid = True
        p.yValueAxis.gridStrokeColor = HexColor("#e5e7eb")
        for i, col in enumerate(palette):
            p.lines[i].strokeColor = col
            p.lines[i].strokeWidth = 1.35
            if xmax == 150:
                marker = makeMarker("FilledCircle")
                marker.size = 2.5
                marker.fillColor = marker.strokeColor = col
                p.lines[i].symbol = marker
        d.add(p)
    def export(d, name):
        pdf = assets/f"{name}.pdf"
        renderPDF.drawToFile(d, str(pdf))
        renderSVG.drawToFile(d, str(assets/f"{name}.svg"))
        doc = pdfium.PdfDocument(pdf)
        page = doc[0]
        image = page.render(scale=1.7).to_pil()
        image.save(assets/f"{name}.png")
        page.close(); doc.close()
    d = base(670, "SSH 双目视差实验总览")
    for i, name in enumerate(names):
        title(d, 560+i*145, 595, name, 11, colors[i].hexval())
    gray = HexColor("#8996aa")
    # Do not interpolate across the frame-10 strategy change.
    segments = []
    for f in frames:
        if not segments or segments[-1][0]["entry_strategy"] != f["entry_strategy"]:
            segments.append([])
        segments[-1].append(f)
    frame_colors = [colors[names.index(s[0]["entry_strategy"])] for s in segments]
    def frame_series(key):
        return [[(f["frame"], f[key]) for f in segment] for segment in segments]
    line(d, 35, 317, "真值与逐帧视差中位数", [[(f["frame"], f["truth_disparity_px"]) for f in frames]]+
         frame_series("median_disparity_px"), [gray]+frame_colors, (0, 105), "视差 / px")
    line(d, 548, 317, "逐帧最大单点误差", frame_series("max_disparity_error_px"),
         frame_colors, (0, .06), "绝对误差 / px")
    line(d, 35, 38, "逐帧匹配阶段耗时", frame_series("match_ms"),
         frame_colors, (0, 10), "耗时 / ms；单轮原始测量")
    title(d, 548, 276, "策略覆盖与点保留", 14)
    for i, n in enumerate(names):
        g = summary["by_entry_strategy"][n]
        y = 235-i*62
        title(d, 555, y, n, 13, colors[i].hexval())
        title(d, 700, y, f"帧 {g['passed']}/{g['frames']}；点 {g['accepted_pairs']}/{g['raw_left_points']}", 13)
        title(d, 700, y-22, f"点保留率 {100*g['point_retention']:.2f}%", 11, "#52617b")
    title(d, 40, 16, "总体：1558/1574 点；MAE 0.008539 px；匹配均值 3.342 ms。不同策略使用不同帧，不能作为同输入消融。", 11)
    export(d, "remote_150_overview")
    d = base(390, "SSH 视差误差分布与平面过滤")
    series = []
    for n in names:
        es = sorted(abs(p["disparity_error_px"]) for p in points if p["entry_strategy"] == n)
        series.append([(0,0)]+[xy for i,e in enumerate(es) for xy in [(e,i/len(es)), (e,(i+1)/len(es))]] )
    line(d, 35, 69, "已保留点的绝对误差累积分布", series, colors, (0, 1.02), "累计比例；横轴：绝对误差 / px", .055, 0)
    title(d, 548, 307, "几何过滤与保留边界", 14)
    lines = ["1558 个保留点均不超过 0.05 px", "最大误差：第 92 帧原角点 4，0.0496 px",
             "平面过滤累计剔除 1 点：第 29 帧", "第 29 帧保留 9/10 点，原编号 9 不在 CSV 中",
             "124 帧执行视差平面检查；26 帧 TT 冗余不足",
             "共丢弃 16 点；其余 15 点不归因于平面过滤"]
    for i, text in enumerate(lines): title(d, 550, 268-i*29, text, 12)
    for i, n in enumerate(names): title(d, 100+i*130, 65, n, 11, colors[i].hexval())
    title(d, 40, 22, "分布只统计保留点；帧通过不等于所有原始点通过，也不证明模板物理身份与真实位姿准确。", 11)
    export(d, "remote_error_distribution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("remote_output"))
    parser.add_argument("--manifest", type=Path, default=Path("data/stereo_crop/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("output/remote_stereo_150_analysis"))
    parser.add_argument("--assets", type=Path, default=Path("docs/assets/algorithm_v1_4"))
    args = parser.parse_args()
    sources = [args.raw/"tracking_log.txt", *sorted(args.raw.glob("disparity_f*.csv")), args.manifest]
    hashes = {p.as_posix(): digest(p) for p in sources}
    summary, frames, points = analyze(args.raw, args.manifest)
    local = Path("output/solver_geometry_optimization/after/run1")
    if local.exists(): summary["local_optimized_comparison"] = compare(args.raw, local)
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    write_table(args.output/"frame_metrics.csv", frames)
    write_table(args.output/"point_metrics.csv", points)
    write_table(args.output/"strategy_summary.csv", [{"strategy": name, **{k:v for k,v in g.items() if not isinstance(v,dict)},
                  "match_mean_ms": g["match_ms"]["mean"], "match_p95_ms": g["match_ms"]["p95"],
                  "planar_mean_ms": g["planar_ms"]["mean"]} for name,g in summary["by_entry_strategy"].items()])
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    plots(args.assets, summary, frames, points)
    assert hashes == {p.as_posix(): digest(p) for p in sources}, "Raw sources changed during analysis"
    provenance = {"analysis_date": "2026-10-14", "raw_import_hashes": hashes,
                  "raw_sources_unchanged": True,
                  "remote_platform": {"cpu": None,"gpu": None,"build_mode": None,"opencv": None},
                  "remote_binary_hash": None,"remote_config_snapshot": None,
                  "dataset_identity": "same 150-pair dataset per user; local manifest is truth reference; no remote image hashes supplied",
                  "script_sha256": digest(Path(__file__)),
                  "note": "Hashes fingerprint imported files now; they are not pre-run remote provenance."}
    (args.output/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
