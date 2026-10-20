"""Frozen before/after contour optimization checks; never overwrite prior runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path("output/contour_scan_20261019")
OLD = Path("output/mono_priority_20261019")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def execute(command, log):
    with Path(log).open("w") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def quality():
    jobs = OLD / "corner_jobs.txt"
    for version in ("before", "after"):
        target = OUT / ("quality_" + version + ".csv")
        if target.exists():
            raise RuntimeError("Quality outputs already exist: " + str(target))
        print("QUALITY", version, flush=True)
        execute([OUT / ("benchmark_" + version), OUT / "before/config/tracker_config.json",
                 jobs, target, 1], OUT / ("quality_" + version + ".log"))
    compare_quality()


def compare_quality():
    before = list(csv.DictReader((OUT / "quality_before.csv").open()))
    after = list(csv.DictReader((OUT / "quality_after.csv").open()))
    assert len(before) == len(after)
    errors = []
    for a, b in zip(before, after):
        for key in a:
            if key not in ("extract_ms", "refine_ms") and a[key] != b[key]:
                errors.append(dict(job=a["job"], extractor=a["extractor"], capture=a["capture"],
                                   key=key, before=a[key], after=b[key]))
    counts = {}
    for row in before:
        key = row["extractor"] + "_capture" + row["capture"]
        counts[key] = counts.get(key, 0) + 1
    result = dict(calls_per_version=len(before), groups=counts,
                  field_differences=len(errors), examples=errors[:30])
    dump(OUT / "quality_comparison.json", result)
    print(json.dumps(result), flush=True)
    assert not errors, "Extraction numerical/debug differences detected"


def timing():
    # Serial ABBA order; both binaries see the same decoded ROI workload.
    # Timing inside the executable excludes input decode and CSV/debug hashing.
    for round_index, versions in enumerate((("before", "after"), ("after", "before"),
                                            ("before", "after"), ("after", "before"))):
        for version in versions:
            target = OUT / ("timing_%d_%s.csv" % (round_index, version))
            if target.exists():
                raise RuntimeError("Timing output already exists: " + str(target))
            print("TIMING", round_index, version, flush=True)
            execute([OUT / ("benchmark_" + version), OUT / "before/config/tracker_config.json",
                     OLD / "timing_jobs.txt", target, 4], target.with_suffix(".log"))


def pipeline():
    for sequence in ("reference", "normal_ambiguity", "oblique_degraded"):
        for version in ("before", "after"):
            name = sequence + "_" + version
            cfg = read(OLD / "configs" / (sequence + "_partial.json"))
            cfg["output"]["directory"] = (OUT / "pipeline" / name).as_posix()
            config = OUT / (name + ".json")
            if (OUT / "pipeline" / name / "poses.csv").exists():
                raise RuntimeError("Pipeline output already exists: " + name)
            dump(config, cfg)
            print("PIPELINE", name, flush=True)
            execute([OUT / ("GPNP_" + version), config], OUT / (name + ".log"))
    fields = ("frame", "success", "strategy", "is_class1", "model_frame", "n_matched",
              "warm_start", "gate_status", "seed_age", "fallback", "failure_reason",
              "roi_candidates", "roi_verified", "roi_selected", "bc_detected", "bc_rejected", "pose_points",
              "tx", "ty", "tz", *["R%d%d" % (i, j) for i in range(3) for j in range(3)])
    comparisons = []
    for sequence in ("reference", "normal_ambiguity", "oblique_degraded"):
        rows = [list(csv.DictReader((OUT / "pipeline" / (sequence + "_" + v) / "poses.csv").open()))
                for v in ("before", "after")]
        assert len(rows[0]) == len(rows[1])
        mismatches = [(a["frame"], key) for a, b in zip(*rows) for key in fields if a[key] != b[key]]
        comparisons.append(dict(sequence=sequence, frames=len(rows[0]),
                                successes=sum(r["success"] == "1" for r in rows[1]),
                                differences=mismatches[:30], difference_count=len(mismatches)))
    dump(OUT / "pipeline_comparison.json", comparisons)
    print(json.dumps(comparisons), flush=True)
    assert all(not r["difference_count"] for r in comparisons)


def summarize():
    import numpy as np
    result = []
    all_rows = []
    for path in sorted(OUT.glob("timing_*.csv")):
        version = path.stem.rsplit("_", 1)[1]
        for r in csv.DictReader(path.open()):
            r["version"] = version
            r["round"] = int(path.stem.split("_")[1])
            all_rows.append(r)
    for kind in ("BC", "TT"):
        for capture in ("0", "1"):
            for size in ("all", "small", "medium", "large"):
                group = [r for r in all_rows if r["extractor"] == kind and r["capture"] == capture
                         and (size == "all" or ("small" if int(r["area"]) <= 4096 else
                              "medium" if int(r["area"]) <= 90000 else "large") == size)]
                for version in ("before", "after"):
                    rows = [r for r in group if r["version"] == version]
                    if not rows:
                        continue
                    values = [float(r["extract_ms"]) for r in rows]
                    refine = [float(r["refine_ms"]) for r in rows]
                    result.append(dict(extractor=kind, capture=int(capture), size=size, version=version,
                                       calls=len(rows), median_ms=float(np.median(values)),
                                       p95_ms=float(np.percentile(values, 95)), mean_ms=float(np.mean(values)),
                                       refine_median_ms=float(np.median(refine))))
    dump(OUT / "timing_summary.json", result)
    class_groups=[]
    for kind in ("BC", "TT"):
        for cls in ("0", "1"):
            for capture in ("0", "1"):
                for version in ("before", "after"):
                    rows=[r for r in all_rows if r["extractor"]==kind and r["cls"]==cls
                          and r["capture"]==capture and r["version"]==version]
                    if not rows:
                        continue
                    class_groups.append(dict(extractor=kind,cls=int(cls),capture=int(capture),version=version,
                        calls=len(rows),median_ms=float(np.median([float(r["extract_ms"]) for r in rows])),
                        mean_ms=float(np.mean([float(r["extract_ms"]) for r in rows])),
                        p95_ms=float(np.percentile([float(r["extract_ms"]) for r in rows],95)),
                        refine_median_ms=float(np.median([float(r["refine_ms"]) for r in rows])),
                        round_mean_ms=[float(np.mean([float(r["extract_ms"]) for r in rows if r["round"]==i])) for i in range(4)]))
    dump(OUT / "timing_class_summary.json",class_groups)
    files = [*OUT.glob("benchmark_*"), OUT / "GPNP_before", OUT / "GPNP_after",
             Path("src/matching/MonoCornerIdentity.cpp"), Path("src/feature/BinaryCornerExtractor.cpp"),
             Path("src/feature/TinyTargetExtractor.cpp"), Path("include/matching/MonoCornerIdentity.hpp")]
    dump(OUT / "after_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print(json.dumps(result, indent=2))


def visuals():
    comparisons=[]
    for case in ("roi_recovery", "depth_recovery", "eight_corners"):
        for version in ("before", "after"):
            name=case+"_"+version
            cfg=read(OLD / "final_configs" / ("debug_"+case+".json"))
            cfg["output"]["directory"]=(OUT / "visuals" / name).as_posix()
            path=OUT / ("debug_"+name+".json")
            if (OUT / "visuals" / name).exists():
                raise RuntimeError("Debug output already exists: "+name)
            dump(path,cfg)
            print("VISUAL",name,flush=True)
            execute([OUT / ("GPNP_"+version),path],OUT / ("debug_"+name+".log"))
        before=OUT / "visuals" / (case+"_before")
        after=OUT / "visuals" / (case+"_after")
        # These panels contain no elapsed-time labels. Compare actual PNGs.
        files=sorted(before.glob("mono_bc_contour_input*.png"))
        assert files, case
        comparisons.append(dict(case=case,panels=len(files),
            identical=all(p.read_bytes()==(after/p.name).read_bytes() for p in files)))
    dump(OUT / "visual_comparison.json",comparisons)
    assert all(r["identical"] for r in comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("quality", "compare", "timing", "pipeline", "summarize", "visuals"))
    action = parser.parse_args().action
    {"quality": quality, "compare": compare_quality, "timing": timing,
     "pipeline": pipeline, "summarize": summarize, "visuals": visuals}[action]()
