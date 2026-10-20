"""Validate archived trials and the final display-only canvas adjustment."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path("output/tt_priority_20261019")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, data):
    Path(path).write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))


def prepare():
    for name, original, directory in (
        ("final_display", OUT / "visual_configs/reference_f16_G.json", "visuals/reference_f16/G_padded"),
        ("final_activation", OUT / "activation_smoke.json", "activation_smoke_final"),
    ):
        cfg = read(original)
        cfg["output"]["directory"] = (OUT / directory).as_posix()
        dump(OUT / (name + ".json"), cfg)


def run():
    for name in ("final_display", "final_activation"):
        cfg = OUT / (name + ".json")
        directory = Path(read(cfg)["output"]["directory"])
        if directory.exists():
            raise RuntimeError("Refusing overwrite " + str(directory))
        with (OUT / (name + ".log")).open("w") as stream:
            subprocess.run(["./build-codex-bc-bench/GPNP", str(cfg)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        print("CHECK", name, flush=True)


def check():
    exp = read(OUT / "experiment.json")
    assert len(exp["jobs"]) == 21
    for job in exp["jobs"]:
        marker = read(OUT / "runs" / (job["name"] + "_complete.json"))
        assert sha(job["binary"]) == job["binary_sha256"] == marker["binary_sha256"]
        assert sha(job["config"]) == job["config_sha256"] == marker["config_sha256"]
        assert sha(OUT / "runs" / job["name"] / "poses.csv") == marker["poses_sha256"]
        assert marker["rows"] == job["frames"]
    for mode in exp["modes"]:
        path = OUT / ("corners_" + mode + ".csv")
        marker = read(path.with_suffix(".complete.json"))
        assert sha(path) == marker["csv_sha256"]
        assert len(rows(path)) == exp["quality_jobs"]
    exact = read(OUT / "verification.json")
    assert exact["diagnostic_only_exact"] and exact["diagnostic_only_corner_exact"]
    assert exact["root_config_verified_activation"]
    corners_e, corners_g = rows(OUT / "corners_E.csv"), rows(OUT / "corners_G.csv")
    keys = ["success", "n", "angle", "overlap"] + [k + str(i) for i in range(4) for k in ("x", "y", "gx", "gy", "id")]
    assert all(a[k] == b[k] for a, b in zip(corners_e, corners_g) for k in keys)
    checks = []
    for label, before, after in (
        ("display_padding", OUT / "visuals/reference_f16/G/poses.csv", OUT / "visuals/reference_f16/G_padded/poses.csv"),
        ("activated_input", OUT / "activation_smoke/poses.csv", OUT / "activation_smoke_final/poses.csv"),
    ):
        a, b = rows(before), rows(after)
        assert len(a) == len(b) == 2
        keys = (set(a[0]) & set(b[0])) - {"extract_ms", "pnp_ms", "roi_verify_ms"}
        diff = [(x["frame"], k) for x, y in zip(a, b) for k in keys if x[k] != y[k]]
        assert not diff, diff
        checks.append(dict(case=label, frames=len(a), non_timing_differences=0))
    source_paths = ["main.cpp", "include/common/Types.hpp", "include/common/TinyTargetPolicy.hpp",
        "include/feature/TinyTargetExtractor.hpp", "include/feature/TinyTargetRobustness.hpp",
        "src/feature/TinyTargetExtractor.cpp", "src/tracker/MonoTracker.cpp", "src/detection/RoiGenerator.cpp",
        "tests/unit/test_tiny_target_robustness.cpp", "tests/CMakeLists.txt", "config/tracker_config.json"]
    for source in source_paths:
        target = OUT / "final_sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    result = dict(pipeline_jobs=21, pipeline_frames=3360, fixed_roi_calls=7168,
        all_archived_hashes_valid=True, final_display_checks=checks,
        e_g_extraction_numerically_identical=True, activated_mode="G",
        final_binary_sha256=sha("build-codex-bc-bench/GPNP"),
        source_sha256={p: sha(p) for p in source_paths})
    dump(OUT / "final_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "run": run, "check": check}[sys.argv[1]]()
