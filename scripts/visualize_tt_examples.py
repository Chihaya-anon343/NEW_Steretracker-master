"""Replay archived TT examples and plot measured corners with actual MonoPnP axes."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/tt_visualization_20261019"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def prepare():
    OUT.mkdir(exist_ok=True)
    config_path = ROOT / "config/tracker_config.json"
    dump(OUT / "root_config_sha256.json", hashlib.sha256(config_path.read_bytes()).hexdigest())
    base = read(config_path)
    jobs = []
    for i, item in enumerate(read(OUT / "selection.json")):
        config = json.loads(json.dumps(base))
        config["mode"] = "debug"
        config["mono_mode"] = True
        config["temporal"]["enabled"] = False
        config["roi_selection"]["enabled"] = False
        config["strategies"]["tiny_max_area"] = 10000000
        config["input"]["left"] = config["input"]["right"] = item["path"]
        x, y, w, h = item["roi"]
        config["manual_roi"] = {"enabled": True, "left": dict(x=x, y=y, width=w, height=h),
                                "right": dict(x=x, y=y, width=w, height=h)}
        config["output"].update(visualize=True, pose_csv=True,
                                 directory="output/tt_visualization_20261019/" + item["case"])
        dump(OUT / (item["case"] + ".json"), config)
        jobs.append(f'{i} {item["path"]} 0 1 {x} {y} {w} {h}\n')
    (OUT / "jobs.txt").write_bytes("".join(jobs).encode())


def run():
    import os
    os.chdir(ROOT)
    for item in read(OUT / "selection.json"):
        with (OUT / (item["case"] + ".log")).open("w") as stream:
            subprocess.run(["./build-codex-bc-bench/GPNP", str(OUT / (item["case"] + ".json"))],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("REPLAY", item["case"], flush=True)
    with (OUT / "extraction.log").open("w") as stream:
        subprocess.run([str(OUT / "evaluate_corner_extraction"), str(OUT / "small.json"),
                        str(OUT / "jobs.txt"), str(OUT / "detections.csv")],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def render():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    config = read(OUT / "small.json")
    c = config["camera"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.]])
    rows = list(csv.DictReader((OUT / "detections.csv").open()))
    manifest = []
    colors = ["#ff8800", "#00ffff", "#ff55ff", "#ffff00"]
    for i, item in enumerate(read(OUT / "selection.json")):
        row = next(r for r in rows if int(r["job"]) == i)
        assert row["success"] == "1" and row["n"] == "4"
        x, y, w, h = item["roi"]
        points = np.array([[float(row[f"x{k}"]), float(row[f"y{k}"])] for k in range(4)])
        difference = float(np.max(np.abs(points - np.array(item["pred"]))))
        # The old evaluator added the ROI offset in float32; this renderer uses float64.
        assert difference < 2e-4, (item["case"], difference)
        pose = next(csv.DictReader((OUT / item["case"] / "poses.csv").open()))
        assert pose["success"] == "1" and pose["strategy"] == "TinyTarget" and pose["n_matched"] == "4", pose
        R = np.array([[float(pose[f"R{a}{b}"]) for b in range(3)] for a in range(3)])
        t = np.array([float(pose[key]) for key in ("tx", "ty", "tz")])
        estimated_tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1., 1.))))
        rvec, _ = cv2.Rodrigues(R)
        axis_len = 100.
        axis = np.array([[0., 0, 0], [axis_len, 0, 0], [0, axis_len, 0],
                         [0, 0, -min(axis_len, .5 * t[2])]])
        projected = cv2.projectPoints(axis, rvec, t, K, None)[0].reshape(-1, 2)
        # Reprojection is an independent check that the replay pose uses this same point set.
        half = config["strategies"]["tiny_target"]["square_size_m_class0"] * 500
        model = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
        reprojected = cv2.projectPoints(model, rvec, t, K, None)[0].reshape(-1, 2)
        reproj = float(np.linalg.norm(reprojected - points, axis=1).mean())
        image = cv2.cvtColor(cv2.imread(str(ROOT / item["path"])), cv2.COLOR_BGR2RGB)
        margin = max(10, int(max(w, h) * .45))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax in axes:
            ax.imshow(image[y0:y1, x0:x1], extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5), interpolation="nearest")
            ax.set_aspect("equal")
            ax.set_xlabel("Image x (original pixels)")
            ax.set_ylabel("Image y (original pixels)")
        axes[0].set_title("Actual TT corners + MonoPnP axes")
        for k, p in enumerate(points):
            axes[0].scatter(*p, s=90, facecolors="none", edgecolors=colors[k], linewidths=2, zorder=5)
            axes[0].annotate(str(k), p, xytext=(6, 7), textcoords="offset points", color=colors[k],
                             fontsize=12, weight="bold", bbox=dict(facecolor="black", alpha=.6, pad=1))
        for j, (label, color) in enumerate([( "+X", "red"), ("+Y", "lime"), ("-Z (up)", "deepskyblue")], 1):
            axes[0].annotate("", projected[j], xytext=projected[0],
                             arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=4)
        axes[0].text(.02, .98, "Red: +X | Green: +Y | Blue: -Z (up)\nAxes length: 100 mm", transform=axes[0].transAxes,
                     va="top", color="white", fontsize=9, bbox=dict(facecolor="black", alpha=.75))
        axes[1].set_title("Measured corners vs annotated corners")
        gt = np.array(item["gt"])
        axes[1].scatter(gt[:, 0], gt[:, 1], marker="+", s=150, linewidths=2, color="lime", label="Annotated corners")
        for k, (p, g) in enumerate(zip(points, gt)):
            axes[1].plot([p[0], g[0]], [p[1], g[1]], color="white", lw=1)
            axes[1].scatter(*p, s=90, facecolors="none", edgecolors=colors[k], linewidths=2,
                            label="Extracted corners" if k == 0 else None)
            axes[1].annotate(str(k), p, xytext=(6, 7), textcoords="offset points", color=colors[k], fontsize=12,
                             weight="bold", bbox=dict(facecolor="black", alpha=.6, pad=1))
        axes[1].legend(fontsize=8, loc="upper left")
        fig.suptitle(f'{item["sequence"]} / source frame {item["frame"]} | class0 TT | span {item["size"]:.1f}px | tilt GT/estimated {item["tilt"]:.1f}/{estimated_tilt:.1f} deg\n'
                     f'Corner RMSE {item["direct_rmse"]:.2f}px | mean reprojection {reproj:.2f}px | annotated ROI, forced TT, cold start', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .9))
        target = OUT / (item["case"] + "_corners_axes.png")
        fig.savefig(target, dpi=160)
        plt.close(fig)
        manifest.append(dict(case=item["case"], source=item["path"], source_frame=item["frame"],
                             roi=item["roi"], corners=points.tolist(), gt_corner_rmse_px=item["direct_rmse"],
                             gt_tilt_deg=item["tilt"], estimated_tilt_deg=estimated_tilt,
                             mean_reprojection_px=reproj, extracted_vs_archived_max_abs_px=difference,
                             pose_csv=str(OUT / item["case"] / "poses.csv"), visualization=str(target)))
    assert read(OUT / "root_config_sha256.json") == hashlib.sha256((ROOT / "config/tracker_config.json").read_bytes()).hexdigest()
    dump(OUT / "visualization_manifest.json", dict(examples=manifest, root_config_preserved=True,
         note="Annotated ROI and forced class0 TT diagnostic replay, current 50-degree prior; not end-to-end YOLO results."))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "run": run, "render": render}[sys.argv[1]]()
