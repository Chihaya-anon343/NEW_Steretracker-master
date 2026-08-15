#!/usr/bin/env python3
"""可视化 ESKF 多模态融合轨迹 (真值 vs 融合, 标注错误段)。

用法:
    python3 plot_eskf_traj.py --traj eskf_traj.csv --events eskf_events.csv --out eskf_traj.png

依赖: numpy, matplotlib
"""

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (投影注册)

# 中文字体支持 (Windows 常见 CJK 字体, 按优先级回退)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'DengXian', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 负号用 ASCII 连字符, 避免显示成方框

# FusionQuality 枚举 -> 标签
QUALITY_LABEL = {0: "Uninitialized", 1: "Normal", 2: "Degraded", 3: "Stale"}


def load_traj(path):
    ts, gt, fused, err, qual = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["t"]))
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
            fused.append([float(row["fused_x"]), float(row["fused_y"]), float(row["fused_z"])])
            err.append(float(row["err_3d"]))
            qual.append(int(row["quality"]))
    return (np.array(ts), np.array(gt), np.array(fused),
            np.array(err), np.array(qual))


def load_events(path):
    events = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                events.append((row["type"], float(row["t_start"]),
                               float(row["t_end"]), float(row["magnitude"])))
    except FileNotFoundError:
        pass
    return events


def main():
    ap = argparse.ArgumentParser(description="ESKF 融合轨迹可视化")
    ap.add_argument("--traj", default="eskf_traj.csv")
    ap.add_argument("--events", default="eskf_events.csv")
    ap.add_argument("--out", default="eskf_traj.png")
    args = ap.parse_args()

    t, gt, fused, err, qual = load_traj(args.traj)
    events = load_events(args.events)

    fig = plt.figure(figsize=(13, 5))

    # ---- 左: 3D 轨迹 ----
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "k--", lw=1.0, label="ground truth")
    sc = ax.scatter(fused[:, 0], fused[:, 1], fused[:, 2], c=qual, s=10,
                    cmap="viridis", label="fused")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("螺旋上升轨迹 (颜色 = quality)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_ticks(sorted(QUALITY_LABEL))
    cbar.set_ticklabels([QUALITY_LABEL[k] for k in sorted(QUALITY_LABEL)])

    # ---- 右: 3D 误差曲线 + 错误段标注 ----
    ax2 = fig.add_subplot(122)
    ax2.plot(t, err, "b-", lw=1.0, label="3D error (m)")
    ymax = max(err.max() * 1.2, 1.0) if len(err) else 1.0
    for typ, ts, te, mag in events:
        ax2.axvspan(ts, te, color="r", alpha=0.15)
        ax2.axvline(ts, color="r", ls=":", lw=0.6)
        ax2.text(ts, ymax * 0.95, typ, rotation=90, fontsize=7, va="top")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("3D error (m)")
    ax2.set_ylim(0, ymax)
    ax2.set_title("融合误差 (红带 = 错误段)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[plot] 已输出 {args.out}  (轨迹 {len(t)} 点, 错误段 {len(events)} 个)")


if __name__ == "__main__":
    main()
