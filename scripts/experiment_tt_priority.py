#!/usr/bin/env python3
"""Frozen TT P0/P1 ablation. No command writes the user's main configuration.

prepare/analyze use host Python (numpy/scipy/cv2). run/quality use container
Python and archived Linux executables. Replays are deliberately serial.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

OUT = Path("output/tt_priority_20261019")
DATA = Path("tests/data/mono_robustness_20261015")
OLD = Path("output/corner_quality_20261016")
MODES = tuple("ABCDEFG")
DEFAULT_SEQUENCES = ("reference", "normal_ambiguity", "oblique_degraded")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n").encode("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def configured(base, mode):
    cfg = copy.deepcopy(base)
    cfg["strategies"]["tiny_target"]["robustness"] = dict(
        enabled=mode in "DEFG", diagnostics=mode != "A", direction_match=True,
        guarded_refinement=mode in "EFG", pixel_center_mapping=mode == "F",
        min_direction_iou=.55, min_direction_gap=.04, min_span_px=4.,
        max_refine_shift_px=1., min_gradient_ratio=.02)
    cfg.setdefault("roi_selection", {})["tt_verification"] = dict(
        enabled=mode in "CDEFG", allow_small_target_path=mode in "CDEF", min_shape_class0=.75, min_shape_class1=.75,
        small_min_shape_class0=.65, small_min_shape_class1=.65,
        small_span_class0=40., small_span_class1=24.,
        min_direction_iou=.60, min_direction_gap=.03)
    return cfg


def audit_model(base):
    """Audit independent renderer mm/px and TT square; do not silently rescale."""
    import numpy as np
    suite = read(DATA / "suite_manifest.json")
    mm_per_px = np.asarray(suite["board_size_mm"], float) / suite["target_size_px"]
    audit = dict(reference="SOURCE annotation axis-aligned bbox vertices projected by per-frame H; virtual corners, not visible material vertices",
                 target_size_px=suite["target_size_px"], board_size_mm=suite["board_size_mm"],
                 source_mm_per_px=mm_per_px.tolist(), classes=[],
                 configured_class1_offset_mm=base.get("template_geometry", {}).get("class1_center_in_class0_mm"),
                 rendered_class1_offset_mm=suite["class1_actual_offset_mm"])
    centers = []
    for cls in (0, 1):
        path = Path(suite["config"]["class%d_points" % cls])
        pts = np.asarray([list(map(float, pair)) for pair in re.findall(
            r"Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)", path.read_text(encoding="utf-8-sig"))])
        lo, hi = pts.min(0), pts.max(0)
        centers.append((lo + hi) / 2)
        rendered = (hi - lo) * mm_per_px
        configured_mm = 1000 * base["strategies"]["tiny_target"]["square_size_m_class%d" % cls]
        audit["classes"].append(dict(cls=cls, annotation_path=path.as_posix(), annotation_sha256=sha(path),
            source_bbox_px=[*lo.tolist(), *hi.tolist()], virtual_width_height_mm=rendered.tolist(),
            tt_square_side_mm=configured_mm, configured_to_rendered_ratio=(configured_mm / rendered).tolist(),
            discrepancy_percent=(100 * (configured_mm / rendered - 1)).tolist()))
    audit["annotation_offset_mm"] = ((centers[1] - centers[0]) * mm_per_px).tolist()
    return audit


def prepare(args):
    if (OUT / "experiment.json").exists():
        raise RuntimeError("Refusing to overwrite an experiment manifest")
    base = read(OUT / "config_before.json")
    binary_before = OUT / "before/build-codex-bc-bench/GPNP"
    binary_after = OUT / "GPNP_after"
    if binary_after.exists():
        raise RuntimeError("Refusing to overwrite archived current binary")
    shutil.copy2(args.binary, binary_after)
    profiles = read(DATA / "suite_manifest.json")["profiles"]
    sequences = [p for p in profiles if args.sequences == ["all"] or p["name"] in args.sequences]
    if not sequences:
        raise ValueError("No requested sequences")
    frozen = []
    for profile in sequences:
        for mode in args.modes:
            cfg = configured(base, mode)
            cfg["mode"], cfg["mono_mode"] = "normal", True
            cfg["manual_roi"]["enabled"] = False
            cfg["input_system"].update(max_frames=0, use_threaded_capture=False)
            cfg["input_system"]["image"].update(type="sequence", directory_path=(DATA / profile["name"]).as_posix(), sequence_pattern="frame")
            for key in ("imu", "altimeter"):
                cfg["input_system"][key]["enabled"] = False
            cfg["eskf"] = {"enabled": False}
            name = profile["name"] + "_" + mode
            cfg["output"] = dict(visualize=False, log_file=True, pose_csv=True,
                                 directory=(OUT / "runs" / name).as_posix())
            path = OUT / "configs" / (name + ".json")
            dump(path, cfg)
            binary = binary_before if mode == "A" else binary_after
            frozen.append(dict(name=name, mode=mode, sequence=profile["name"], frames=profile["frames"],
                config=path.as_posix(), config_sha256=sha(path), binary=binary.as_posix(), binary_sha256=sha(binary)))
    source_jobs = [j for j in read(OLD / "jobs.json") if j["strategy"] == "TT"]
    jobs = [dict(j, virtual_corner_gt=j["gt"]) for j in source_jobs if j["roi"][2] * j["roi"][3] <= args.max_roi_area]
    for j in jobs:
        j.pop("gt")
    dump(OUT / "corner_jobs.json", jobs)
    lines = ["%d %s %d 1 %d %d %d %d" % (j["job"], j["path"], j["cls"], *j["roi"]) for j in jobs]
    (OUT / "corner_jobs.txt").write_bytes(("\n".join(lines) + "\n").encode())
    timed = [j for j in jobs if j["frame"] % (15 if j["cls"] == 0 else 60) == 0]
    timed_lines = ["%d %s %d 1 %d %d %d %d" % (j["job"], j["path"], j["cls"], *j["roi"]) for j in timed]
    (OUT / "timing_jobs.txt").write_bytes(("\n".join(timed_lines) + "\n").encode())
    for mode in args.modes:
        dump(OUT / "configs" / ("forced_" + mode + ".json"), configured(base, mode))
    dump(OUT / "model_audit.json", audit_model(base))
    dump(OUT / "experiment.json", dict(dataset=DATA.as_posix(), jobs=frozen, modes=args.modes,
        source_config_sha256=sha(OUT / "config_before.json"), root_config_sha256_at_prepare=sha("config/tracker_config.json"),
        quality_jobs=len(jobs), max_roi_area=args.max_roi_area, temporal=base.get("temporal", {}),
        prior=base.get("mono_pose_prior", {}), mode_semantics=dict(A="frozen baseline executable",
            B="diagnostics only", C="TT ROI policy", D="C plus four-way direction",
            E="D plus guarded refinement", F="E plus separate pixel-center mapping ablation"),
        suite_sha256=sha(DATA / "suite_manifest.json"),
        manifest_sha256={p["name"]: sha(DATA / p["manifest"]) for p in profiles},
        quality_source_sha256=sha(OLD / "jobs.json"), yolo_sha256=sha(base["yolo"]["model_path"])))
    print("Prepared %d pipeline jobs and %d forced TT ROIs" % (len(frozen), len(jobs)), flush=True)


def add_strict(args):
    """Append a separately frozen conservative production candidate; never rewrite A-F."""
    exp = read(OUT / "experiment.json")
    if "G" in exp["modes"]:
        raise RuntimeError("Conservative candidate already frozen")
    if not (OUT / "experiment_exploratory.json").exists():
        shutil.copy2(OUT / "experiment.json", OUT / "experiment_exploratory.json")
    binary = OUT / "GPNP_strict"
    if binary.exists():
        raise RuntimeError("Refusing to overwrite conservative binary")
    shutil.copy2(args.binary, binary)
    cfg = configured(read(OUT / "config_before.json"), "G")
    dump(OUT / "configs/forced_G.json", cfg)
    selected = [j for j in exp["jobs"] if j["mode"] == "A"]
    for before in selected:
        original = read(before["config"])
        original["strategies"]["tiny_target"]["robustness"] = cfg["strategies"]["tiny_target"]["robustness"]
        original["roi_selection"]["tt_verification"] = cfg["roi_selection"]["tt_verification"]
        name = before["sequence"] + "_G"
        original["output"]["directory"] = (OUT / "runs" / name).as_posix()
        path = OUT / "configs" / (name + ".json")
        dump(path, original)
        exp["jobs"].append(dict(name=name, mode="G", sequence=before["sequence"], frames=before["frames"],
             config=path.as_posix(), config_sha256=sha(path), binary=binary.as_posix(), binary_sha256=sha(binary)))
    exp["modes"].append("G")
    exp["mode_semantics"]["G"] = "Four-way and guarded refinement, original 0.75 gate; experimental weak-shape admission disabled"
    dump(OUT / "experiment.json", exp)
    print("Appended G conservative candidate; A-F archive preserved", flush=True)


def execute(command, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("w", encoding="utf-8") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def cached_jobs(original):
    """Use only the optional cache whose decoded pixels were checked exactly."""
    cache = OUT / "image_cache.csv"
    if not (OUT / "image_cache.complete.json").exists():
        return Path(original)
    assert sha(cache) == read(OUT / "image_cache.complete.json")["manifest_sha256"]
    mapping = {r["source"]: r["cached"] for r in rows(cache)}
    target = OUT / (Path(original).stem + "_cached.txt")
    translated = []
    for line in Path(original).read_text().splitlines():
        parts = line.split()
        parts[1] = mapping[parts[1]]
        translated.append(" ".join(parts))
    content = ("\n".join(translated) + "\n").encode()
    if target.exists():
        assert target.read_bytes() == content
    else:
        target.write_bytes(content)
    return target


def cached_config(original):
    cache = OUT / "image_cache.csv"
    if not (OUT / "image_cache.complete.json").exists():
        return Path(original)
    assert sha(cache) == read(OUT / "image_cache.complete.json")["manifest_sha256"]
    config = read(original)
    directory = config["input_system"]["image"]["directory_path"]
    mapping = {r["source"]: r["cached"] for r in rows(cache)}
    examples = [value for key, value in mapping.items() if str(Path(key).parent) == directory]
    assert examples
    config["input_system"]["image"]["directory_path"] = str(Path(examples[0]).parent)
    target = OUT / "configs" / (Path(original).stem + "_cached.json")
    if target.exists():
        assert read(target) == config
    else:
        dump(target, config)
    return target


def run(args):
    completed = []
    for job in read(OUT / "experiment.json")["jobs"]:
        if job["mode"] not in args.modes or (args.sequences != ["all"] and job["sequence"] not in args.sequences):
            continue
        assert sha(job["config"]) == job["config_sha256"], "Configuration changed after freeze"
        assert sha(job["binary"]) == job["binary_sha256"], "Binary changed after freeze"
        target = OUT / "runs" / job["name"] / "poses.csv"
        marker = OUT / "runs" / (job["name"] + "_complete.json")
        if target.exists() or marker.exists():
            if not (args.resume and target.exists() and marker.exists() and sha(target) == read(marker)["poses_sha256"]):
                raise RuntimeError("Refusing overwrite of incomplete/non-resumed job " + job["name"])
        else:
            print("RUN", job["name"], flush=True)
            runtime_config = cached_config(job["config"])
            execute([job["binary"], runtime_config], OUT / "runs" / (job["name"] + "_stdout.log"))
            raw = rows(target)
            assert len(raw) == job["frames"]
            assert [int(r["frame"]) for r in raw] == list(range(1, job["frames"] + 1))
            dump(marker, dict(name=job["name"], rows=len(raw), poses_sha256=sha(target),
                             binary_sha256=sha(job["binary"]), config_sha256=sha(job["config"]),
                             runtime_config=str(runtime_config), runtime_config_sha256=sha(runtime_config)))
        completed.append(read(marker))
        print("DONE", job["name"], flush=True)
    dump(OUT / "last_run_completion.json", completed)


def quality(args):
    for mode in args.modes:
        binary = Path(args.evaluator_before if mode == "A" else args.evaluator)
        config = OUT / "configs" / ("forced_" + mode + ".json")
        target = OUT / ("corners_" + mode + ".csv")
        marker = target.with_suffix(".complete.json")
        signature = dict(binary_sha256=sha(binary), config_sha256=sha(config),
                         jobs_sha256=sha(OUT / "corner_jobs.txt"), repeats=args.repeats)
        if target.exists() or marker.exists():
            if not (args.resume and marker.exists() and target.exists() and all(read(marker).get(k) == v for k, v in signature.items()) and sha(target) == read(marker)["csv_sha256"]):
                raise RuntimeError("Refusing overwrite of fixed ROI output " + mode)
        else:
            print("QUALITY", mode, flush=True)
            runtime_jobs = cached_jobs(OUT / "corner_jobs.txt")
            execute([binary, config, runtime_jobs, target, args.repeats, 0], target.with_suffix(".log"))
            assert len(rows(target)) == read(OUT / "experiment.json")["quality_jobs"] * args.repeats
            dump(marker, dict(signature, csv_sha256=sha(target), runtime_jobs=str(runtime_jobs), runtime_jobs_sha256=sha(runtime_jobs)))


def metric_stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def timing(args):
    """Run after all builds/replays, with alternated serial order and warmup."""
    for round_index in range(args.rounds):
        order = args.modes if round_index % 2 == 0 else list(reversed(args.modes))
        for mode in order:
            binary = Path(args.evaluator_before if mode == "A" else args.evaluator)
            target = OUT / ("timing_%d_%s.csv" % (round_index, mode))
            if target.exists():
                raise RuntimeError("Refusing to overwrite independent timing " + str(target))
            print("TIMING", round_index, mode, flush=True)
            execute([binary, OUT / "configs" / ("forced_" + mode + ".json"),
                     cached_jobs(OUT / "timing_jobs.txt"), target, args.repeats, 0], target.with_suffix(".log"))


def analyze(args):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from analyze_mono_robustness import analyze_job
    exp = read(OUT / "experiment.json")
    frames, completed_jobs = [], []
    for job in exp["jobs"]:
        marker = OUT / "runs" / (job["name"] + "_complete.json")
        if not marker.exists():
            continue
        _, measured, _ = analyze_job(job, DATA, OUT)
        raw = rows(OUT / "runs" / job["name"] / "poses.csv")
        for r, detail in zip(measured, raw):
            r.update(mode=job["mode"], sequence=job["sequence"], failure=detail["failure_reason"],
                     roi_details=detail.get("roi_verification_details", ""), tt_status=detail.get("tt_status", ""),
                     roi_verify_ms=float(detail.get("roi_verify_ms", 0)))
        frames.extend(measured)
        completed_jobs.append(job)
    summaries = []
    scopes = ("all", "tilt_le30", "tilt_30_50", "tilt_gt50")
    def in_scope(r, scope):
        return scope == "all" or (scope == "tilt_le30" and r["tilt_gt"] <= 30) or (scope == "tilt_30_50" and 30 < r["tilt_gt"] <= 50) or (scope == "tilt_gt50" and r["tilt_gt"] > 50)
    for mode in exp["modes"]:
        for scope in scopes:
            selected = [r for r in frames if r["mode"] == mode and in_scope(r, scope)]
            if not selected:
                continue
            good = [r for r in selected if r["success"]]
            summaries.append(dict(mode=mode, scope=scope, frames=len(selected), success=len(good),
                normal_jumps=sum(r["z_jump"] for r in selected), depth_jumps=sum(r["depth_jump"] for r in selected),
                depth_error_over100pct=sum(r["depth_rel_error"] > 1.0 for r in good),
                normal_error_deg=metric_stats([r["normal_error_deg"] for r in good]),
                depth_rel_error=metric_stats([r["depth_rel_error"] for r in good]),
                extract_ms=metric_stats([r["extract_ms"] for r in selected]),
                roi_verify_ms=metric_stats([r["roi_verify_ms"] for r in selected]),
                failures=dict(Counter(r["failure"] for r in selected if not r["success"])),
                successful_strategy_counts=dict(Counter(r["strategy"] for r in good))))
    comparisons = []
    lookup = {(r["mode"], r["sequence"], r["frame"]): r for r in frames}
    for mode in exp["modes"]:
        if mode == "A":
            continue
        pairs = [(r, lookup[(mode, r["sequence"], r["frame"])]) for r in frames if r["mode"] == "A" and (mode, r["sequence"], r["frame"]) in lookup]
        if not pairs:
            continue
        common = [(a, b) for a, b in pairs if a["success"] and b["success"]]
        common_adjacent = [(a, b) for a, b in common if a["frame"] > 0 and
            lookup.get(("A", a["sequence"], a["frame"] - 1), {}).get("success") and
            lookup.get((mode, a["sequence"], a["frame"] - 1), {}).get("success")]
        recovered = [b for a, b in pairs if not a["success"] and b["success"]]
        lost = [a for a, b in pairs if a["success"] and not b["success"]]
        comparisons.append(dict(mode=mode, paired_frames=len(pairs), common_success=len(common),
            common_adjacent_pairs=len(common_adjacent),
            common_adjacent_normal_jumps_before=sum(a["z_jump"] for a, b in common_adjacent),
            common_adjacent_normal_jumps_after=sum(b["z_jump"] for a, b in common_adjacent),
            common_adjacent_depth_jumps_before=sum(a["depth_jump"] for a, b in common_adjacent),
            common_adjacent_depth_jumps_after=sum(b["depth_jump"] for a, b in common_adjacent),
            recovered_depth_error_over100pct=sum(r["depth_rel_error"] > 1.0 for r in recovered),
            recovered=[dict(sequence=r["sequence"], frame=r["frame"], strategy=r["strategy"], normal_error_deg=r["normal_error_deg"], depth_rel_error=r["depth_rel_error"]) for r in recovered],
            lost=[dict(sequence=r["sequence"], frame=r["frame"], strategy=r["strategy"]) for r in lost],
            common_normal_error_before=metric_stats([a["normal_error_deg"] for a, b in common]),
            common_normal_error_after=metric_stats([b["normal_error_deg"] for a, b in common]),
            common_depth_error_before=metric_stats([a["depth_rel_error"] for a, b in common]),
            common_depth_error_after=metric_stats([b["depth_rel_error"] for a, b in common])))
    # B must preserve the numerical/state portion of A. Newly added diagnostics
    # have no corresponding old column and are deliberately not compared.
    exact = []
    ignore = {"extract_ms", "pnp_ms", "roi_verify_ms", "total_ms"}
    for sequence in {j["sequence"] for j in completed_jobs}:
        paths = [OUT / "runs" / (sequence + "_" + m) / "poses.csv" for m in "AB"]
        markers = [OUT / "runs" / (sequence + "_" + m + "_complete.json") for m in "AB"]
        if not all(p.exists() for p in paths + markers):
            continue
        a, b = map(rows, paths)
        keys = set(a[0]) & set(b[0]) - ignore
        differences = [(x["frame"], k, x[k], y[k]) for x, y in zip(a, b) for k in sorted(keys) if x[k] != y[k]]
        exact.append(dict(sequence=sequence, frames=len(a), differences=len(differences), examples=differences[:20]))
    dump(OUT / "pose_frames.json", frames)
    dump(OUT / "pose_summary.json", summaries)
    dump(OUT / "pose_comparison.json", comparisons)
    dump(OUT / "diagnostic_only_comparison.json", exact)
    corner_jobs = {j["job"]: j for j in read(OUT / "corner_jobs.json")}
    corner_frames = []
    for mode in exp["modes"]:
        path = OUT / ("corners_" + mode + ".csv")
        if not path.exists() or not path.with_suffix(".complete.json").exists():
            continue
        for r in rows(path):
            j = corner_jobs[int(r["job"])]
            n = int(r["n"])
            gt = np.asarray(j["virtual_corner_gt"])
            pred = np.asarray([[float(r["gx%d" % i] if "gx%d" % i in r else r["x%d" % i]), float(r["gy%d" % i] if "gy%d" % i in r else r["y%d" % i])] for i in range(n)])
            # gx/gy are full-image coordinates; old evaluators emitted full x/y.
            ok = r["success"] == "1" and n == 4 and bool(np.isfinite(pred).all())
            record = dict(job=j["job"], sequence=j["sequence"], frame=j["frame"], cls=j["cls"], mode=mode,
                repeat=int(r.get("repeat", 0)), success=ok, fully_visible=j["fully_visible"], size=j["size"], tilt=j["tilt"],
                status=r.get("status", ""), extract_ms=float(r["extract_ms"]), pred=pred.tolist(),
                virtual_corner_gt=gt.tolist(), direction_iou=float(r.get("direction_iou", 0)), direction_gap=float(r.get("direction_gap", 0)))
            if ok:
                distance = np.linalg.norm(pred[:, None] - gt[None, :], axis=2)
                ri, ci = linear_sum_assignment(distance ** 2)
                direct, assigned = np.diag(distance), distance[ri, ci]
                mismatch = bool(not np.array_equal(ci, np.arange(4)) and np.sqrt(np.mean(direct ** 2)) - np.sqrt(np.mean(assigned ** 2)) > max(1., .01 * j["size"]))
                record.update(virtual_corner_errors_px=direct.tolist(), assigned_errors_px=assigned.tolist(),
                              virtual_corner_rmse_px=float(np.sqrt(np.mean(direct ** 2))),
                              permutation=ci.tolist(), numbering_mismatch_diagnostic=mismatch)
            corner_frames.append(record)
    corner_summary = []
    for mode in exp["modes"]:
        for cls in (0, 1):
            for size in ("all", "lt20", "20_50", "ge50"):
                selected = [r for r in corner_frames if r["mode"] == mode and r["cls"] == cls and r["fully_visible"] and (size == "all" or (size == "lt20" and r["size"] < 20) or (size == "20_50" and 20 <= r["size"] < 50) or (size == "ge50" and r["size"] >= 50))]
                if not selected:
                    continue
                good = [r for r in selected if r["success"]]
                corner_summary.append(dict(mode=mode, cls=cls, size=size, attempts=len(selected), success=len(good),
                    numbering_mismatch_diagnostic=sum(r["numbering_mismatch_diagnostic"] for r in good),
                    virtual_corner_error_px=metric_stats([e for r in good for e in r["virtual_corner_errors_px"]]),
                    all_virtual_corners_within2=sum(max(r["virtual_corner_errors_px"]) <= 2 for r in good),
                    extract_ms=metric_stats([r["extract_ms"] for r in selected]),
                    statuses=dict(Counter(r["status"] for r in selected))))
    dump(OUT / "corner_frames.json", corner_frames)
    dump(OUT / "corner_summary.json", corner_summary)
    corner_comparison = []
    corner_lookup = {(r["mode"], r["job"], r["repeat"]): r for r in corner_frames}
    for mode in exp["modes"]:
        if mode == "A":
            continue
        for cls in (0, 1):
            pairs = [(a, corner_lookup[(mode, a["job"], a["repeat"])]) for a in corner_frames
                     if a["mode"] == "A" and a["cls"] == cls and a["fully_visible"]
                     and (mode, a["job"], a["repeat"]) in corner_lookup]
            if not pairs:
                continue
            common = [(a, b) for a, b in pairs if a["success"] and b["success"]]
            corner_comparison.append(dict(mode=mode, cls=cls, paired_visible_attempts=len(pairs),
                common_success=len(common), recovered=sum(not a["success"] and b["success"] for a, b in pairs),
                lost=sum(a["success"] and not b["success"] for a, b in pairs),
                common_virtual_error_before=metric_stats([v for a, b in common for v in a["virtual_corner_errors_px"]]),
                common_virtual_error_after=metric_stats([v for a, b in common for v in b["virtual_corner_errors_px"]]),
                common_mismatch_before=sum(a["numbering_mismatch_diagnostic"] for a, b in common),
                common_mismatch_after=sum(b["numbering_mismatch_diagnostic"] for a, b in common)))
    dump(OUT / "corner_comparison.json", corner_comparison)
    exact_corners = None
    quality_paths = [OUT / ("corners_" + m + ".csv") for m in "AB"]
    if all(p.exists() for p in quality_paths):
        a, b = map(rows, quality_paths)
        keys = ["job", "cls", "repeat", "success", "n", "angle", "overlap"] + [key + str(i) for i in range(4) for key in ("x", "y", "gx", "gy", "id")]
        differences = [(x["job"], k, x[k], y[k]) for x, y in zip(a, b) for k in keys
                       if x[k] != y[k] and not (x["success"] == y["success"] == "0" and k in ("angle", "overlap"))]
        stale_resets = [(x["job"], k, x[k], y[k]) for x, y in zip(a, b) for k in ("angle", "overlap")
                        if x["success"] == y["success"] == "0" and x[k] != y[k]]
        exact_corners = dict(rows_before=len(a), rows_after=len(b), difference_count=len(differences), examples=differences[:30],
                            failed_call_stale_state_resets=stale_resets,
                            comparison="Success/count/coordinates/IDs and successful-call angle/IoU; failed-call stale angle/IoU reset is reported separately.")
        dump(OUT / "diagnostic_only_corner_comparison.json", exact_corners)
    timing_rows = []
    for path in OUT.glob("timing_*.csv"):
        _, round_index, mode = path.stem.split("_")
        for r in rows(path):
            timing_rows.append(dict(mode=mode, round=int(round_index), cls=int(r["cls"]), extract_ms=float(r["extract_ms"])))
    timing_summary = []
    for mode in exp["modes"]:
        for cls in (0, 1):
            selected = [r for r in timing_rows if r["mode"] == mode and r["cls"] == cls]
            if selected:
                timing_summary.append(dict(mode=mode, cls=cls, calls=len(selected), extract_ms=metric_stats([r["extract_ms"] for r in selected]),
                    round_mean_ms={str(index): float(np.mean([r["extract_ms"] for r in selected if r["round"] == index])) for index in sorted({r["round"] for r in selected})}))
    dump(OUT / "timing_summary.json", timing_summary)
    activation_path = OUT / "production_activation.json"
    activation = read(activation_path) if activation_path.exists() else {}
    dump(OUT / "verification.json", dict(root_config_unchanged=sha("config/tracker_config.json") == exp["source_config_sha256"],
        root_config_verified_activation=bool(activation) and activation.get("unrelated_configuration_preserved", False) and
            sha("config/tracker_config.json") == activation.get("after_sha256"),
        production_mode=activation.get("mode"),
        source_config_sha256=exp["source_config_sha256"], current_root_config_sha256=sha("config/tracker_config.json"),
        completed_pipeline_jobs=len(completed_jobs), diagnostic_only_exact=all(r["differences"] == 0 for r in exact),
        diagnostic_only_pairs=len(exact), forced_roi_rows=len(corner_frames),
        diagnostic_only_corner_exact=exact_corners is not None and exact_corners["difference_count"] == 0 and exact_corners["rows_before"] == exact_corners["rows_after"],
        note="Pipeline timing includes normal run noise; use independent repeated serial timing to claim speed gains."))
    print(json.dumps(dict(pipeline_jobs=len(completed_jobs), pipeline_frames=len(frames), fixed_roi_rows=len(corner_frames), diagnostics=exact), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "add_strict", "run", "quality", "timing", "analyze"))
    parser.add_argument("--binary", default="build-codex-bc-bench/GPNP")
    parser.add_argument("--evaluator", default="output/tt_priority_20261019/evaluate_tt_priority")
    parser.add_argument("--evaluator-before", default="output/tt_priority_20261019/evaluate_tt_priority_before")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--max-roi-area", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    globals()[args.action](args)
