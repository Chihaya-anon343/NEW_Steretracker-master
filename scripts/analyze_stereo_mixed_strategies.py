"""Analyze actual entry/final strategies, signed disparity truth and repeated timings.

Uses only the standard library; an optional Pillow chart is written when available.
No tracker configuration or input image is modified.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stats(values):
    values = sorted(values)
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1]}


def write_table(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family(strategy):
    if strategy.startswith("DualRoi"):
        return "Dual-ROI"
    return {"BinaryCorner": "BC", "TinyTarget": "TT"}.get(strategy, "Unknown")


def parse(root, run, truth):
    text = (root / run / "tracking_log.txt").read_text(encoding="utf-8-sig")
    console = (root / f"{run}_console.log").read_text(encoding="utf-8-sig")
    status, extraction = {}, {}
    for line in text.splitlines():
        if line.startswith("[StereoFrontend]"):
            values = dict(re.findall(r"([a-z_0-9]+)=([^ ]+)", line))
            values["route"] = line.split(" route=", 1)[-1] if " route=" in line else ""
            index = int(values["frame"])
            if index in status:
                raise ValueError(f"Duplicate frame {run}/{index}")
            status[index] = values
        if line.startswith("|"):
            columns = [c.strip() for c in line.strip("|").split("|")]
            if columns and columns[0].isdigit() and len(columns) >= 11:
                extraction[int(columns[0])] = {"fallback": columns[3] == "Y", "left_kp": int(columns[4]),
                                               "pnp": columns[10]}
    frames, points = [], []
    for index, item in enumerate(truth["frames"], 1):
        if index not in status:
            raise ValueError(f"Missing frontend status {run}/{index}; inspect skipped frames before summarizing")
        s = status[index]
        ok = int(s["match_ok"])
        path = root / run / f"disparity_f{index}.csv"
        with path.open(encoding="utf-8-sig") as stream:
            pairs = list(csv.DictReader(stream))
        if len(pairs) != int(s["pairs"]):
            raise ValueError(f"Point-count mismatch {run}/{index}")
        if len({p["left_index"] for p in pairs}) != len(pairs):
            raise ValueError(f"Duplicate original point ID {run}/{index}")
        route = s["route"]
        if "FRAME:" in route or route.startswith("C0:"):
            entry = "Dual-ROI"
        elif re.match(r"C[01]BC:", route):
            entry = "BC"
        elif re.match(r"C[01]TT:", route):
            entry = "TT"
        else:
            entry = "Unknown"
        strategy = pairs[0]["strategy"] if pairs else {"BC": "BinaryCorner", "TT": "TinyTarget"}.get(entry, "DualRoi_unknown")
        frame_points = []
        for p in pairs:
            if int(p["frame_match_ok"]) != ok or p["strategy"] != strategy:
                raise ValueError(f"CSV/log status or strategy disagreement {run}/{index}")
            row = {"run": run, "frame": index, "source": item["src"], "entry_strategy": entry,
                   "final_strategy": strategy, "frame_ok": ok}
            for key in ("left_index", "region", "original_index", "template_index"):
                row[key] = int(p[key])
            for key in ("left_x", "left_y", "right_x", "right_y", "disparity_px"):
                row[key] = float(p[key])
            row["truth_disparity_px"] = item["disparity"]
            row["disparity_error_px"] = row["disparity_px"] - item["disparity"]
            row["truth_dy_px"] = item["dy_px"]
            row["observed_dy_px"] = row["right_y"] - row["left_y"]
            row["vertical_error_px"] = row["observed_dy_px"] - item["dy_px"]
            points.append(row)
            frame_points.append(row)
        errors = [abs(p["disparity_error_px"]) for p in frame_points]
        vertical = [abs(p["vertical_error_px"]) for p in frame_points]
        f = {"run": run, "frame": index, "source": item["src"], "entry_strategy": entry,
             "final_strategy": strategy, "final_family": family(strategy), "ok": ok,
             "pairs": len(pairs), "left_kp": extraction.get(index, {}).get("left_kp"),
             "fallback": int(extraction.get(index, {}).get("fallback", False)),
             "truth_disparity_px": item["disparity"], "truth_dy_px": item["dy_px"],
             "median_disparity_px": statistics.median([p["disparity_px"] for p in frame_points]) if pairs else None,
             "disparity_mae_px": statistics.mean(errors) if errors else None,
             "max_disparity_error_px": max(errors) if errors else None,
             "vertical_mae_px": statistics.mean(vertical) if vertical else None,
             "match_ms": float(s["match_ms"]), "full_retry": int(s.get("full_retry", 0)),
             "pyramids": int(s.get("pyramids", 0)), "legacy": int(s.get("legacy", 0)),
             "coarse_attempts": int(s["coarse"]), "c0_pairs": int(s.get("c0", 0)),
             "c1_pairs": int(s.get("c1", 0)), "roi": s.get("roi", ""),
             "reason": s.get("reason", ""), "route": route}
        if s.get("pnp") != "SKIP" or s.get("source") != "Unspecified" or int(s["stereo"]) != 0:
            raise ValueError(f"Unexpected pose result in disparity-only experiment {run}/{index}")
        frames.append(f)
    runtime = {}
    for key, label in (("fps", "实际处理 FPS"), ("yolo_ms", "YOLO 平均用时"),
                       ("tracker_ms", "单帧跟踪平均用时 \\(提取\\+解算\\+ROI等\\)"),
                       ("extract_match_ms", "特征提取平均用时"), ("pnp_ms", "位姿解算平均用时")):
        m = re.search(label + r": ([0-9.eE+-]+)", console)
        runtime[key] = float(m.group(1)) if m else None
    m = re.search(r"YOLO推理次数: left=([0-9]+) right=([0-9]+)", text)
    runtime["left_yolo"], runtime["right_yolo"] = (int(m[1]), int(m[2])) if m else (None, None)
    return frames, points, runtime


def summarize(frames, points, timings):
    accepted = [p for p in points if p["frame_ok"]]
    errors = [abs(p["disparity_error_px"]) for p in accepted]
    return {"frames": len(frames), "passed": sum(f["ok"] for f in frames),
            "success_rate": sum(f["ok"] for f in frames) / len(frames),
            "accepted_pairs": len(accepted), "all_retained_pairs": len(points),
            "mean_pairs": statistics.mean(f["pairs"] for f in frames),
            "mae_px": statistics.mean(errors) if errors else None,
            "rmse_px": math.sqrt(statistics.mean(p["disparity_error_px"] ** 2 for p in accepted)) if accepted else None,
            "p95_abs_error_px": stats(errors)["p95"], "max_abs_error_px": max(errors) if errors else None,
            "vertical_mae_px": statistics.mean(abs(p["vertical_error_px"]) for p in accepted) if accepted else None,
            "vertical_max_error_px": max(abs(p["vertical_error_px"]) for p in accepted) if accepted else None,
            "outliers_gt_0p5_px": sum(e > .5 for e in errors), "outliers_gt_1_px": sum(e > 1 for e in errors),
            "fallback_frames": sum(f["fallback"] for f in frames), "full_retries": sum(f["full_retry"] for f in frames),
            "legacy_frames": sum(f["legacy"] for f in frames), "pyramids": sum(f["pyramids"] for f in frames),
            "coarse_attempts": sum(f["coarse_attempts"] for f in frames),
            "match_ms": stats([f["match_ms"] for f in timings]),
            "first_frame": min(f["frame"] for f in frames), "last_frame": max(f["frame"] for f in frames)}


def plot_overview(root, result, frames, timed_frames):
    # Draw a new data visualization from numeric measurements, not from source photos.
    from PIL import Image, ImageDraw, ImageFont
    font_path = "C:/Windows/Fonts/msyh.ttc"
    font = lambda size: ImageFont.truetype(font_path, size)
    image = Image.new("RGB", (1280, 1540), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    colors = {"Dual-ROI": "#2563eb", "BC": "#059669", "TT": "#d97706"}
    draw.text((54, 30), "150 对双目：Dual-ROI / BC / TT", fill="#142137", font=font(32))
    draw.text((54, 83), "1 次预热 + 3 次正式重复；精度统计使用 run1，耗时汇总三轮", fill="#52617b", font=font(20))
    for j, (name, color) in enumerate(colors.items()):
        draw.ellipse((760+j*153, 84, 775+j*153, 99), fill=color)
        draw.text((780+j*153, 78), name, fill="#142137", font=font(20))

    def axes(top, title, ymax, ylabel):
        draw.rounded_rectangle((36, top, 1244, top+317), 16, fill="white")
        draw.text((62, top+18), title, fill="#142137", font=font(23))
        x0, y0, x1, y1 = 115, top+72, 1195, top+268
        for n in range(5):
            y = y1 - n*(y1-y0)/4
            draw.line((x0, y, x1, y), fill="#e5eaf1", width=1)
            value = ymax*n/4
            draw.text((52, y-12), f"{value:g}", fill="#66748a", font=font(16))
        draw.line((x0, y0, x0, y1, x1, y1), fill="#75859b", width=2)
        for n in (1, 25, 50, 75, 100, 125, 150):
            x = x0+(n-1)/149*(x1-x0)
            draw.text((x-12, y1+10), str(n), fill="#66748a", font=font(16))
        draw.text((1070, top+289), "帧编号", fill="#66748a", font=font(16))
        draw.text((62, top+48), ylabel, fill="#66748a", font=font(16))
        return lambda n, v: (x0+(n-1)/149*(x1-x0), y1-v/ymax*(y1-y0))

    xy = axes(132, "视差随帧变化：灰线为真值，彩点为匹配中位数", 100, "px")
    draw.line([xy(f["frame"], f["truth_disparity_px"]) for f in frames], fill="#9ca8b9", width=3)
    for f in frames:
        if f["median_disparity_px"] is not None:
            x, y = xy(f["frame"], f["median_disparity_px"])
            draw.ellipse((x-3, y-3, x+3, y+3), fill=colors[f["entry_strategy"]])
    xy = axes(465, "逐帧最大单点绝对误差：第 29 帧保留点存在明显偏差", .7, "px")
    for f in frames:
        if f["max_disparity_error_px"] is not None:
            x, y = xy(f["frame"], f["max_disparity_error_px"])
            draw.line((x, xy(f["frame"], 0)[1], x, y), fill=colors[f["entry_strategy"]], width=2)
            draw.ellipse((x-3, y-3, x+3, y+3), fill=colors[f["entry_strategy"]])
    x, y = xy(29, .6371)
    draw.text((x+15, y-5), "原角点 9：0.6371 px", fill="#b91c1c", font=font(18))
    mean_times = {f["frame"]: statistics.mean(g["match_ms"] for g in timed_frames if g["frame"] == f["frame"])
                  for f in frames}
    ceiling = math.ceil(max(mean_times.values()) / 2)*2
    xy = axes(798, "匹配阶段耗时：每帧三轮均值，包含匹配内重试", ceiling, "ms")
    draw.line([xy(f["frame"], mean_times[f["frame"]]) for f in frames], fill="#aeb9c9", width=2)
    for f in frames:
        x, y = xy(f["frame"], mean_times[f["frame"]])
        draw.ellipse((x-3, y-3, x+3, y+3), fill=colors[f["entry_strategy"]])
    draw.rounded_rectangle((36, 1131, 1244, 1474), 16, fill="white")
    draw.text((62, 1153), "三种策略的实际覆盖与通过率", fill="#142137", font=font(23))
    for j, (name, color) in enumerate(colors.items()):
        s = result["by_entry_strategy"][name]
        y = 1212+j*68
        draw.text((65, y), name, fill="#142137", font=font(21))
        draw.rounded_rectangle((240, y, 1070, y+30), 6, fill="#e5eaf1")
        draw.rounded_rectangle((240, y, 240+830*s["success_rate"], y+30), 6, fill=color)
        draw.text((1085, y), f"{s['passed']}/{s['frames']}", fill="#142137", font=font(21))
    draw.text((62, 1440), "不同策略对应不同尺度和点数；耗时大小不构成同输入算法速度对照。", fill="#66748a", font=font(17))
    image.save(root / "overview.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--runs", nargs="+", default=["run1", "run2", "run3"])
    parser.add_argument("--plot", action="store_true", help="Create numeric-data PNG with Pillow on Windows")
    args = parser.parse_args()
    root = args.directory
    truth = read_json(root / "manifest_snapshot.json")
    all_frames, all_points, run_stats = [], [], []
    primary_frames, primary_points = [], []
    stability = []
    for run in args.runs:
        frames, points, runtime = parse(root, run, truth)
        all_frames.extend(frames)
        all_points.extend(points)
        run_stats.append({"run": run, "frames": len(frames), "passed": sum(f["ok"] for f in frames), **runtime})
        if not primary_frames:
            primary_frames, primary_points = frames, points
        else:
            for a, b in zip(primary_frames, frames):
                name = f"disparity_f{a['frame']}.csv"
                stability.append({"run": run, "frame": a["frame"], "same_entry": a["entry_strategy"] == b["entry_strategy"],
                                  "same_final": a["final_strategy"] == b["final_strategy"], "same_status": a["ok"] == b["ok"],
                                  "csv_identical": (root / args.runs[0] / name).read_bytes() == (root / run / name).read_bytes()})
    groups = {}
    for group in ("Dual-ROI", "BC", "TT"):
        fs = [f for f in primary_frames if f["entry_strategy"] == group]
        ps = [p for p in primary_points if p["entry_strategy"] == group]
        ts = [f for f in all_frames if f["entry_strategy"] == group]
        if fs:
            groups[group] = summarize(fs, ps, ts)
    transitions = []
    previous = None
    for f in primary_frames:
        key = (f["entry_strategy"], f["final_strategy"])
        if key != previous:
            transitions.append({"frame": f["frame"], "entry": key[0], "final": key[1]})
            previous = key
    result = {"accuracy_reference_run": args.runs[0], "timing_runs": args.runs,
              "overall": summarize(primary_frames, primary_points, all_frames), "by_entry_strategy": groups,
              "final_strategy_counts": {s: sum(f["final_strategy"] == s for f in primary_frames)
                                        for s in sorted({f["final_strategy"] for f in primary_frames})},
              "runtime": run_stats, "transitions": transitions,
              "repeat_comparisons": len(stability), "identical_csv_comparisons": sum(s["csv_identical"] for s in stability),
              "same_entry_comparisons": sum(s["same_entry"] for s in stability),
              "same_status_comparisons": sum(s["same_status"] for s in stability),
              "top_error_frames": sorted(primary_frames, key=lambda f: f["max_disparity_error_px"] or 0, reverse=True)[:10]}
    write_table(root / "frame_metrics.csv", all_frames)
    write_table(root / "point_metrics.csv", all_points)
    write_table(root / "repeat_consistency.csv", stability)
    write_table(root / "runtime_summary.csv", run_stats)
    write_table(root / "strategy_summary.csv", [{"strategy": group, **{k: v for k, v in value.items() if not isinstance(v, dict)},
                                                 "mean_match_ms": value["match_ms"]["mean"],
                                                 "p95_match_ms": value["match_ms"]["p95"]} for group, value in groups.items()])
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot_overview(root, result, primary_frames, all_frames)
    print(json.dumps({"overall": result["overall"], "strategies": groups, "runtime": run_stats,
                      "transitions": transitions, "identical_csv_comparisons": result["identical_csv_comparisons"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
