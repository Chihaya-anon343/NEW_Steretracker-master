#!/usr/bin/env python3
"""Run the saved original and band matcher on identical stereo_crop inputs.

Invoke through scripts/docker-toolchain.sh. Truth is used only after tracking,
never as a matcher seed. Matching timings exclude disk output and visualization;
these are integrated Release measurements, not an end-to-end FPS benchmark.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = sorted(values)
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[max(0, math.ceil(len(values) * .95) - 1)], "max": values[-1]}


def parse_run(root, name, manifest):
    text = (root / name / "tracking_log.txt").read_text(encoding="utf-8")
    statuses = {}
    for line in text.splitlines():
        if not line.startswith("[StereoFrontend]"):
            continue
        values = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
        frame = int(values["frame"])
        if frame in statuses:
            raise RuntimeError(f"Duplicate frame {name}/{frame}")
        statuses[frame] = values
    frames, points = [], []
    if len(statuses) != len(manifest["frames"]):
        raise RuntimeError(f"Missing frames in {name}: {len(statuses)}")
    for frame, truth in enumerate(manifest["frames"], 1):
        status = statuses[frame]
        with (root / name / f"disparity_f{frame}.csv").open() as stream:
            pairs = list(csv.DictReader(stream))
        if len(pairs) != int(status["pairs"]):
            raise RuntimeError(f"Point count mismatch {name}/{frame}")
        ok = int(status["match_ok"])
        frames.append({"run": name, "frame": frame, "ok": ok, "pairs": len(pairs),
                       "match_ms": float(status["match_ms"]), "full_retry": int(status.get("full_retry", "0")),
                       "coarse_attempts": int(status["coarse"]),
                       "reason": status.get("reason", ""), "left_indices": "|".join(p["left_index"] for p in pairs)})
        for pair in pairs:
            row = {"run": name, "frame": frame, "frame_ok": ok, "left_index": int(pair["left_index"])}
            row.update({k: float(v) for k, v in pair.items() if k != "left_index"})
            row["disparity_error_px"] = row["disparity_px"] - truth["disparity"]
            row["vertical_error_px"] = row["right_y"] - row["left_y"] - truth["dy_px"]
            points.append(row)
    return frames, points


def summarize(frames, points):
    passed = [p for p in points if p["frame_ok"]]
    return {"frames": len(frames), "successful_frames": sum(f["ok"] for f in frames),
            "retained_pairs": len(points), "successful_frame_pairs": len(passed),
            "full_retries": sum(f["full_retry"] for f in frames),
            "coarse_attempts": sum(f["coarse_attempts"] for f in frames),
            "match_ms": stats([f["match_ms"] for f in frames]),
            "all_disparity_abs_error_px": stats([abs(p["disparity_error_px"]) for p in points]),
            "successful_disparity_abs_error_px": stats([abs(p["disparity_error_px"]) for p in passed]),
            "vertical_abs_error_px": stats([abs(p["vertical_error_px"]) for p in points]),
            "failed_frames": [f["frame"] for f in frames if not f["ok"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="output/stereo_epipolar_band_test")
    parser.add_argument("--binary", default="build-codex-bc-bench/GPNP")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    config = read_json(root / "tracker_config_snapshot.json")
    manifest = read_json(root / "manifest_snapshot.json")
    data = Path(config["input_system"]["image"]["directory_path"])
    manifest_hash = digest(data / "manifest.json")
    if manifest_hash != digest(root / "manifest_snapshot.json"):
        raise RuntimeError("Dataset manifest changed since snapshot")
    hashes = {str(data / frame[k]): digest(data / frame[k])
              for frame in manifest["frames"] for k in ("left", "right")}
    provenance = {"manifest_sha256": manifest_hash, "original_binary_sha256": digest(root / "GPNP_original"),
                  "band_binary_sha256": digest(args.binary), "image_sha256": hashes,
                  "measured_repeats": args.repeats, "visualize_measured": False,
                  "timing": "Release integrated matcher only; excludes image decoding, YOLO, BC, PnP and output"}
    write_json(root / "provenance.json", provenance)

    def run(name, band, binary, cold_frame=None, visualize=False, full=False):
        cfg = json.loads(json.dumps(config))
        cfg["mode"] = "debug" if cold_frame else "normal"
        cfg["output"].update(directory=str(root / name), visualize=visualize, log_file=True)
        cfg["stereo_frontend"]["coarse_epipolar_band"] = band
        if full:
            cfg["stereo_frontend"]["roi_pyramid"] = False
        if cold_frame:
            truth = manifest["frames"][cold_frame - 1]
            cfg["input"].update(left=str(data / truth["left"]), right=str(data / truth["right"]))
        cfg_path = root / f"{name}.json"
        write_json(cfg_path, cfg)
        print(f"Running {name}", flush=True)
        with (root / f"{name}_console.txt").open("w", encoding="utf-8") as stream:
            subprocess.run([str(Path(binary).resolve()), str(cfg_path)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    # One complete warmup sequence per implementation, not included in timings.
    run("original_warmup", False, root / "GPNP_original")
    run("band_warmup", True, args.binary)
    all_frames, all_points, summaries = [], [], {}
    for round_index in range(args.repeats):
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for band in order:
            name = f"{'band' if band else 'original'}_r{round_index + 1}"
            run(name, band, args.binary if band else root / "GPNP_original")
    # Verify the compatibility switch against the actual saved original.
    run("line_switch", False, args.binary)
    run("band_full", True, args.binary, full=True)
    for name in ["original_warmup", "band_warmup", "line_switch", "band_full"] + [
            f"{method}_r{r+1}" for r in range(args.repeats) for method in ("original", "band")]:
        frames, points = parse_run(root, name, manifest)
        all_frames.extend(frames)
        all_points.extend(points)
        summaries[name] = summarize(frames, points)
    for method in ("original", "band"):
        measured_frames = [f for f in all_frames if f["run"].startswith(method + "_r")]
        measured_points = [p for p in all_points if p["run"].startswith(method + "_r")]
        if measured_frames:
            summaries[method + "_measured"] = summarize(measured_frames, measured_points)
    table(root / "frames.csv", all_frames)
    table(root / "points.csv", all_points)
    original = [f for f in all_frames if f["run"] == "original_warmup"]
    changes = []
    for other in ("band_warmup", "line_switch", "band_full"):
        for a, b in zip(original, [f for f in all_frames if f["run"] == other]):
            changes.append({"variant": other, "frame": a["frame"], "original_ok": a["ok"], "variant_ok": b["ok"],
                            "original_pairs": a["pairs"], "variant_pairs": b["pairs"],
                            "same_indices": a["left_indices"] == b["left_indices"]})
    table(root / "comparison.csv", changes)
    write_json(root / "summary.json", summaries)
    if args.diagnostics:
        run("band_visual", True, args.binary, visualize=True)
        for frame in (29, 60):
            for band in (False, True):
                run(f"diagnostic_{'band' if band else 'original'}_f{frame}", band,
                    args.binary if band else root / "GPNP_original", cold_frame=frame, visualize=True)
    if manifest_hash != digest(data / "manifest.json") or any(digest(p) != h for p, h in hashes.items()):
        raise RuntimeError("Dataset changed during comparison")
    print(json.dumps({k: v for k, v in summaries.items() if k.endswith("warmup") or k.endswith("measured")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
