#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续接近 + 旋转序列合成脚本
============================
复用 generate_synthetic_dataset.py 的平面单应几何 (make_homography / render /
write_points / load_points), 把"每帧随机采样"换成"时间参数化的确定性轨迹":

    size(i)  = SIZE_START · GROWTH^i                  短边视尺寸, 几何增长 (匀速接近)
    yaw(i)   = YAW_RATE · i                           面内匀速旋转
    pitch(i) = PITCH_AMP · (sin(2πi/Pp) − sin(0))     俯仰慢摆 (i=0 时为 0)
    roll(i)  = ROLL_AMP · (sin(2πi/Pr+φ) − sin(φ))    滚转慢摆 (i=0 时为 0)
    center   = 画布正中, 全程固定

第 0 帧为严格正位 (size=SIZE_START, yaw=pitch=roll=0, 居中)。
终止条件: 任一 class1 特征点投影越出画布 → 该帧起停止 (全程 class1 可见)。
画布外的 class0 点不写入 txt (仅保留画布内可见点)。

输出 (单目序列, 文件名匹配 input_system.image.sequence_pattern="frame";
SequenceSource 按扩展名过滤图像, 同目录 txt 不会被误读):
    <out>/frame_000000.png ...
    <out>/frame_000000_class0.txt / _class1.txt
    <out>/manifest.json

用法:
    python scripts/generate_approach_sequence.py [--out tests/data/approach_seq]

依赖:
    numpy, opencv-python
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError as e:
    print("[ERROR] 需要 opencv-python: pip install opencv-python numpy", file=sys.stderr)
    raise e

# 复用参考脚本的单应几何与 IO (脚本目录自动在 sys.path 上)
from generate_synthetic_dataset import (
    FOCAL_LEN,
    TARGET_IMG,
    BACKGROUND_IMG,
    CLASS0_POINTS,
    CLASS1_POINTS,
    load_points,
    make_homography,
    project_points,
    render,
    write_points,
)

# ============================================================================
# 轨迹参数 (硬编码, 按需修改)
# ============================================================================
CANVAS = (1744, 1744)     # 画布尺寸 (W, H)

# 相机内参 — 必须与 config/tracker_config.json camera 节一致 (渲染与 PnP 解算共用同一套 K)
CAM_FX = 1370.1815888598676
CAM_FY = 1373.0103114278359
CAM_CX = 516.4761400598917
CAM_CY = 856.0291167973786

SIZE_START = 20.0         # 起始短边视尺寸 (px, 第 0 帧正位 20x20)
GROWTH = 1.06             # 每帧尺寸倍率 (几何增长 = 匀速接近)
YAW_RATE = 2.5            # 面内旋转速度 (度/帧)
PITCH_AMP = 8.0           # 俯仰摆动幅度 (度)
PITCH_PERIOD = 40.0       # 俯仰摆动周期 (帧)
ROLL_AMP = 8.0            # 滚转摆动幅度 (度)
ROLL_PERIOD = 55.0        # 滚转摆动周期 (帧, 与俯仰不同周期 → 非重复组合)
ROLL_PHASE = 90.0         # 滚转初相 (度)
MAX_FRAMES = 200          # 安全上限帧数

# 策略档位面积阈值 (仅 manifest 信息性标注, 以 tracker_config.json 实际值为准)
TINY_MAX_AREA = 900
AKAZE_MIN_AREA = 200001


# ----------------------------------------------------------------------------
# 轨迹函数
# ----------------------------------------------------------------------------
def osc(i, amp, period, phase_deg):
    """平滑正弦摆动: amp·(sin(2πi/period + φ) − sin(φ))。

    减去初相保证 i=0 时恒为 0 (第 0 帧严格正位), 且逐帧连续无跳变。
    """
    return amp * (np.sin(2.0 * np.pi * i / period + np.deg2rad(phase_deg))
                  - np.sin(np.deg2rad(phase_deg)))


def trajectory_pose(i):
    """第 i 帧的 (size, yaw, pitch, roll)。第 0 帧为严格正位。"""
    size = SIZE_START * (GROWTH ** i)
    yaw = 0.0 if i == 0 else YAW_RATE * i
    pitch = osc(i, PITCH_AMP, PITCH_PERIOD, 0.0)
    roll = osc(i, ROLL_AMP, ROLL_PERIOD, ROLL_PHASE)
    return size, yaw, pitch, roll


def visible_area(H, target_size, canvas):
    """目标源图经 H 投影后落在画布内的像素面积 (截断后实际可见面积)。"""
    Ht, Wt = target_size[1], target_size[0]
    mask = cv2.warpPerspective(np.full((Ht, Wt), 255, np.uint8), H, canvas,
                               flags=cv2.INTER_NEAREST, borderValue=0)
    return int(cv2.countNonZero(mask))


def area_band(area):
    if area <= TINY_MAX_AREA:
        return "tiny"
    if area < AKAZE_MIN_AREA:
        return "bc"
    return "akaze"


# ============================================================================
# 主流程
# ============================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="生成连续接近+旋转的单目序列及特征点标注",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default="tests/data/approach_seq", help="输出目录")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    target = cv2.imread(TARGET_IMG, cv2.IMREAD_COLOR)
    bg_src = cv2.imread(BACKGROUND_IMG, cv2.IMREAD_COLOR)
    if target is None:
        print(f"[ERROR] 无法读取目标图: {TARGET_IMG}", file=sys.stderr)
        return 1
    if bg_src is None:
        print(f"[ERROR] 无法读取背景图: {BACKGROUND_IMG}", file=sys.stderr)
        return 1
    class0 = load_points(CLASS0_POINTS)
    class1 = load_points(CLASS1_POINTS)
    print(f"[INFO] class0 点 {len(class0)} 个, class1 点 {len(class1)} 个")

    target_size = (target.shape[1], target.shape[0])
    bg = cv2.resize(bg_src, CANVAS, interpolation=cv2.INTER_AREA)
    center = (CANVAS[0] / 2.0, CANVAS[1] / 2.0)

    Wc, Hc = CANVAS

    def inside(p):
        return 0.0 <= p[0] < Wc and 0.0 <= p[1] < Hc

    manifest = {
        "version": 1,
        "note": "连续接近+旋转确定性序列 (单应透视生成), 终止于 class1 点越出画布前",
        "canvas": list(CANVAS),
        "n_class0_points": len(class0),
        "n_class1_points": len(class1),
        "params": {
            "size_start": SIZE_START, "growth": GROWTH, "yaw_rate_deg": YAW_RATE,
            "pitch_amp_deg": PITCH_AMP, "pitch_period_frames": PITCH_PERIOD,
            "roll_amp_deg": ROLL_AMP, "roll_period_frames": ROLL_PERIOD,
            "roll_phase_deg": ROLL_PHASE,
            "fx": CAM_FX, "fy": CAM_FY, "cx": CAM_CX, "cy": CAM_CY,
        },
        "frames": [],
    }

    n_frames = 0
    for i in range(MAX_FRAMES):
        size, yaw, pitch, roll = trajectory_pose(i)
        H = make_homography(CANVAS, target_size, size, yaw, pitch, roll,
                            center, FOCAL_LEN,
                            intrinsics=(CAM_FX, CAM_FY, CAM_CX, CAM_CY))

        c0 = project_points(H, class0)
        c1 = project_points(H, class1)

        # 终止条件: 任一 class1 点越出画布 → 该帧起停止
        if any(not inside(p) for p in c1):
            print(f"[INFO] 帧 {i}: class1 点越出画布, 序列终止 (共 {n_frames} 帧)")
            break

        img = render(bg, target, H)
        vis_area = visible_area(H, target_size, CANVAS)

        base = f"frame_{i:06d}"
        cv2.imwrite(os.path.join(args.out, f"{base}.png"), img)

        c0_vis = [p for p in c0 if inside(p)]
        c1_vis = [p for p in c1 if inside(p)]
        write_points(os.path.join(args.out, f"{base}_class0.txt"),
                     c0_vis, "Class 0", CANVAS)
        write_points(os.path.join(args.out, f"{base}_class1.txt"),
                     c1_vis, "Class 1", CANVAS)

        manifest["frames"].append({
            "frame": i,
            "path": f"{base}.png",
            "class0_txt": f"{base}_class0.txt",
            "class1_txt": f"{base}_class1.txt",
            "size_px": round(size, 2),
            "yaw_deg": round(yaw % 360.0, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "visible_area_px": vis_area,
            "band_estimate": area_band(vis_area),
            "n_class0_visible": len(c0_vis),
            "n_class1_visible": len(c1_vis),
        })
        n_frames += 1

        if i % 10 == 0:
            print(f"[INFO] 帧 {i}: size={size:.1f}px yaw={yaw % 360.0:.1f}° "
                  f"pitch={pitch:.2f}° roll={roll:.2f}° area={vis_area}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] 序列生成完毕 -> {os.path.abspath(args.out)}")
    print(f"  帧数: {n_frames} (canvas {CANVAS[0]}x{CANVAS[1]})")
    if n_frames:
        last = manifest["frames"][-1]
        print(f"  末帧: size={last['size_px']}px visible_area={last['visible_area_px']}px "
              f"band={last['band_estimate']}")
    print(f"  manifest.json: {n_frames} 条记录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
