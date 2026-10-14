#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从单目序列图裁剪合成双目数据集。

几何约定 (与项目相机模型 t_rl=[+baseline,0,0] 一致, u_L - u_R = d > 0):
    左图 = 原图裁掉【右侧】d 列   -> 保留 [0, W-d)
    右图 = 原图裁掉【左侧】d 列   -> 保留 [d, W)
两图分辨率一致 (W-d) x H。视差与深度关系: Z = fx * baseline / d。

右图额外施加 warpAffine 亚像素垂直平移 (0, dy) 模拟真实装配失配,
默认 dy 全序列固定 (真实相机装配偏移是常量), 可选每帧随机。

输出文件平铺在输出目录 (left_XXXXXX.png / right_XXXXXX.png),
可直接被 DirectoryStereoSource (left_pattern="left", right_pattern="right") 消费。

用法: 直接修改下方 CONFIG 后运行
    python scripts/generate_stereo_crop_dataset.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# CONFIG — 按需修改后直接运行
# ============================================================================
CONFIG = {
    # 输入: 单目序列图目录 (png/jpg/jpeg/bmp/tiff)
    "input_dir": r"C:\D\Codes\NEW_Steretracker-master_COPY\data\上升image_00_间隔10",

    # 输出目录 (left_XXXXXX.png / right_XXXXXX.png + manifest.json)
    "out_dir": r"C:\D\Codes\NEW_Steretracker-master_COPY\data\stereo_crop",

    # 视差范围 [d_min, d_max] (px)。d_mode:
    #   "ramp"   = 全序列 d 从 d_max 线性降至 d_min (模拟远离, 视差减小=深度增大)
    #   "random" = 每帧独立均匀采样
    "d_min": 1,
    "d_max": 100,
    "d_mode": "ramp",

    # 右图垂直失配范围 [dy_min, dy_max] (px, 亚像素)。dy_mode:
    #   "fixed"  = 全序列一次采样 (真实装配偏移为常量)
    #   "random" = 每帧独立采样
    # dy_max <= 0 关闭垂直失配
    "dy_min": 1.0,
    "dy_max": 2.0,
    "dy_mode": "random",

    # 左右独立光度增强 (亮度/对比度 ±5% + 噪声 sigma∈[0,2])
    "photometric": False,

    # 随机种子 (可复现)
    "seed": 42,
}
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
MIN_WIDTH_MARGIN = 32  # 裁剪后要求保留的最小宽度余量 (px)


def natural_key(p: Path):
    """文件名自然排序键: frame2 < frame10。"""
    name = p.stem
    parts, buf, has_digit = [], [], False
    for ch in name:
        if ch.isdigit():
            buf.append(ch)
            has_digit = True
        else:
            if has_digit:
                parts.append(int("".join(buf)))
                buf, has_digit = [], False
            parts.append(ch.lower())
    if has_digit:
        parts.append(int("".join(buf)))
    return parts


def imread_unicode(path: Path):
    """Windows 中文路径安全读取。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError("imencode failed: %s" % str(path))
    buf.tofile(str(path))


def apply_photometric(img: np.ndarray, rng: np.random.Generator):
    """独立光度扰动: 亮度增益 ±5% / 对比度 ±5% / 高斯噪声 sigma∈[0,2]。"""
    gain = float(rng.uniform(0.95, 1.05))
    contrast = float(rng.uniform(0.95, 1.05))
    sigma = float(rng.uniform(0.0, 2.0))
    out = img.astype(np.float32) * gain
    out = (out - 128.0) * contrast + 128.0
    if sigma > 0.01:
        out += rng.normal(0.0, sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out, {"gain": round(gain, 4), "contrast": round(contrast, 4),
                 "noise_sigma": round(sigma, 4)}


def main(cfg: dict):
    in_dir = Path(cfg["input_dir"])
    if not in_dir.is_dir():
        print("[Error] 输入目录不存在: %s" % str(in_dir), file=sys.stderr)
        return 1

    d_lo, d_hi = int(cfg["d_min"]), int(cfg["d_max"])
    if d_lo < 1 or d_hi < d_lo:
        print("[Error] 非法视差范围: [%d, %d] (要求 1 <= d_min <= d_max)" % (d_lo, d_hi),
              file=sys.stderr)
        return 1
    d_mode = cfg["d_mode"]
    if d_mode not in ("ramp", "random"):
        print("[Error] 非法 d_mode: %r (应为 ramp/random)" % d_mode, file=sys.stderr)
        return 1
    dy_lo, dy_hi = float(cfg["dy_min"]), float(cfg["dy_max"])
    if dy_lo < 0 or dy_hi < dy_lo:
        print("[Error] 非法 dy 范围: [%g, %g]" % (dy_lo, dy_hi), file=sys.stderr)
        return 1
    dy_mode = cfg["dy_mode"]
    if dy_mode not in ("fixed", "random"):
        print("[Error] 非法 dy_mode: %r (应为 fixed/random)" % dy_mode, file=sys.stderr)
        return 1

    files = sorted((f for f in in_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTS),
                   key=natural_key)
    if not files:
        print("[Error] 目录中无图像文件 (%s)" % str(in_dir), file=sys.stderr)
        return 1

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    # dy 全序列固定时只采样一次
    dy_fixed = float(rng.uniform(dy_lo, dy_hi)) if dy_hi > 0 else 0.0
    n = len(files)

    frames_meta = []
    n_skipped = 0
    seq = 0
    for i, src in enumerate(files):
        img = imread_unicode(src)
        if img is None:
            print("[Warn] 无法读取, 跳过: %s" % src.name, file=sys.stderr)
            n_skipped += 1
            continue
        h, w = img.shape[:2]
        if w <= d_hi + MIN_WIDTH_MARGIN:
            print("[Warn] 宽度 %d 不足 (需 > d_max+%d), 跳过: %s"
                  % (w, MIN_WIDTH_MARGIN, src.name), file=sys.stderr)
            n_skipped += 1
            continue

        if d_mode == "random" or d_lo == d_hi:
            d = int(rng.integers(d_lo, d_hi + 1))
        else:
            # ramp: 从 d_max 线性降至 d_min (远离序列)
            t = i / (n - 1) if n > 1 else 0.0
            d = int(round(d_hi + (d_lo - d_hi) * t))

        left = img[:, :w - d]
        right = img[:, d:]

        if dy_mode == "random":
            dy = float(rng.uniform(dy_lo, dy_hi)) if dy_hi > 0 else 0.0
        else:
            dy = dy_fixed
        if dy != 0.0:
            M = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, dy]])
            right = cv2.warpAffine(right, M, (right.shape[1], right.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

        photo = None
        if cfg["photometric"]:
            left, pl = apply_photometric(left, rng)
            right, pr = apply_photometric(right, rng)
            photo = {"left": pl, "right": pr}

        seq += 1
        left_name = "left_%06d.png" % seq
        right_name = "right_%06d.png" % seq
        imwrite_unicode(out_dir / left_name, left)
        imwrite_unicode(out_dir / right_name, right)
        frames_meta.append({
            "src": src.name, "left": left_name, "right": right_name,
            "disparity": d, "dy_px": round(dy, 3), "photometric": photo,
        })
        print("[%d/%d] %s -> d=%d dy=%.2f (%dx%d)"
              % (i + 1, n, src.name, d, dy, left.shape[1], left.shape[0]))

    manifest = {
        "global": {
            "input": str(in_dir), "n_input": n, "n_generated": seq,
            "n_skipped": n_skipped, "disparity": [d_lo, d_hi],
            "d_mode": d_mode, "dy_range": [dy_lo, dy_hi],
            "dy_mode": dy_mode, "photometric": bool(cfg["photometric"]),
            "seed": cfg["seed"],
            "note": "u_L-u_R=d>0; Z=fx*baseline/d; 右图含 +dy 垂直平移",
        },
        "frames": frames_meta,
    }
    with open(str(out_dir / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n[DONE] 生成 %d 对 (跳过 %d) -> %s" % (seq, n_skipped, str(out_dir)))
    print("       d∈[%d,%d] (%s)  dy∈[%g,%g] (%s, 全序列 dy=%.2f)"
          % (d_lo, d_hi, d_mode, dy_lo, dy_hi, dy_mode, dy_fixed))
    return 0


if __name__ == "__main__":
    sys.exit(main(CONFIG))
