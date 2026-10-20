"""Actual C++ fixed-ROI TT replays and measured-corner/pose comparison figures."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path("output/tt_priority_20261019")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())


def prepare():
    jobs = {r["job"]: r for r in read(OUT / "corner_jobs.json")}
    cache = {r["source"]: r["cached"] for r in csv.DictReader((OUT / "image_cache.csv").open())}
    cases = []
    for job_id in (65, 1961, 3405):
        job = jobs[job_id]
        name = job["sequence"] + "_f" + str(job["frame"])
        item = dict(job, name=name)
        cases.append(item)
        for mode in ("A", "G"):
            cfg = read(OUT / "configs" / ("forced_" + mode + ".json"))
            cfg["mode"], cfg["mono_mode"] = "debug", True
            cfg["temporal"]["enabled"] = False
            cfg["strategies"]["tiny_max_area"] = 10000000
            cfg["input"]["left"] = cfg["input"]["right"] = cache[job["path"]]
            x, y, w, h = job["roi"]
            roi = dict(x=x, y=y, width=w, height=h)
            cfg["manual_roi"] = dict(enabled=True, left=roi, right=roi)
            directory = OUT / "visuals" / name / mode
            cfg["output"] = dict(visualize=True, pose_csv=True, log_file=True, directory=directory.as_posix())
            dump(OUT / "visual_configs" / (name + "_" + mode + ".json"), cfg)
    dump(OUT / "visual_cases.json", cases)


def run():
    for case in read(OUT / "visual_cases.json"):
        for mode in ("A", "G"):
            cfg = OUT / "visual_configs" / (case["name"] + "_" + mode + ".json")
            target = OUT / "visuals" / case["name"] / mode
            if target.exists():
                raise RuntimeError("Refusing overwrite " + str(target))
            binary = OUT / ("before/build-codex-bc-bench/GPNP" if mode == "A" else "GPNP_strict")
            with cfg.with_suffix(".log").open("w") as stream:
                subprocess.run([str(binary), str(cfg)], stdout=stream, stderr=subprocess.STDOUT, check=True)
            print("VISUAL", case["name"], mode, flush=True)


def render():
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = read(OUT / "config_before.json")
    c = base["camera"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.]])
    extracted = {m: {int(r["job"]): r for r in csv.DictReader((OUT / ("corners_" + m + ".csv")).open())} for m in ("A", "G")}
    colors = ("#ff8800", "cyan", "magenta", "yellow")
    results = []
    for case in read(OUT / "visual_cases.json"):
        image = cv2.cvtColor(cv2.imread(case["path"]), cv2.COLOR_BGR2RGB)
        x, y, w, h = case["roi"]
        margin = max(10, int(max(w, h) * .4))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        gt = np.array(case["virtual_corner_gt"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        measurements = []
        for ax, mode in zip(axes, ("A", "G")):
            row = extracted[mode][case["job"]]
            points = np.array([[float(row[f"gx{i}"]), float(row[f"gy{i}"])] for i in range(4)])
            pose_path = OUT / "visuals" / case["name"] / mode / "poses.csv"
            pose = next(csv.DictReader(pose_path.open()))
            assert pose["success"] == "1" and pose["strategy"] == "TinyTarget" and pose["n_matched"] == "4"
            R = np.array([[float(pose[f"R{i}{j}"]) for j in range(3)] for i in range(3)])
            t = np.array([float(pose[k]) for k in ("tx", "ty", "tz")])
            rv = cv2.Rodrigues(R)[0]
            axis = np.array([[0., 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, -100]])
            projected = cv2.projectPoints(axis, rv, t, K, None)[0].reshape(-1, 2)
            half = base["strategies"]["tiny_target"]["square_size_m_class0"] * 500
            model = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
            reproj = cv2.projectPoints(model, rv, t, K, None)[0].reshape(-1, 2)
            residual = float(np.linalg.norm(reproj - points, axis=1).mean())
            rmse = float(np.sqrt(np.mean(np.sum((points - gt) ** 2, axis=1))))
            ax.imshow(image[y0:y1, x0:x1], extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5), interpolation="nearest")
            ax.scatter(gt[:, 0], gt[:, 1], color="lime", marker="+", s=110, linewidths=1.5, label="Virtual reference")
            for i, point in enumerate(points):
                ax.scatter(*point, facecolors="none", edgecolors=colors[i], s=85, linewidths=2)
                ax.annotate(str(i), point, xytext=(5, 7), textcoords="offset points", color=colors[i], weight="bold", fontsize=12,
                            bbox=dict(facecolor="black", alpha=.65, pad=1))
                ax.plot([point[0], gt[i, 0]], [point[1], gt[i, 1]], color="white", lw=.8, alpha=.8)
            for j, color in enumerate(("red", "lime", "deepskyblue"), 1):
                ax.annotate("", projected[j], xytext=projected[0], arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax.set_title(f'{"Original" if mode == "A" else "Final TT"} | virtual-point RMSE {rmse:.2f}px\nmean reprojection {residual:.2f}px')
            ax.set_xlabel("Original image x (px)")
            ax.set_ylabel("Original image y (px)")
            ax.legend(fontsize=8, loc="upper left")
            measurements.append(dict(mode=mode, corners=points.tolist(), virtual_corner_rmse_px=rmse, reprojection_px=residual,
                                     pose_csv=pose_path.as_posix()))
        fig.suptitle(f'{case["sequence"]}, source frame {case["frame"]} | span {case["size"]:.1f}px, GT tilt {case["tilt"]:.1f} deg\n'
                     'Annotated ROI, forced TT, actual C++ poses | axes: red +X, green +Y, blue -Z (up), 100mm', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .9))
        target = OUT / "visuals" / (case["name"] + "_comparison.png")
        fig.savefig(target, dpi=150)
        plt.close(fig)
        results.append(dict(name=case["name"], source=case["path"], roi=case["roi"], measurements=measurements,
                            figure=target.as_posix()))
    dump(OUT / "visualization_verification.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "run": run, "render": render}[sys.argv[1]]()
