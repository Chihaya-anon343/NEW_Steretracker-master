#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成训练数据集生成脚本 (3 轴旋转 + 全图特征点标注)
=====================================================
在参考 tests/scripts/generate_assets.py 的"图像参数 + 五状态分类"基础上:
  对目标图 (class0, data/big/img_1.png) 施加随机 3 轴姿态的透视投影
  (yaw 0-360° 全覆盖, pitch/roll 小角度, 鸟瞰视角), 随机尺度与任意位置
  (不被截断), 整图矩形粘贴到背景上;
  并把 class0/class1 特征点经同一单应变换投影, 为每一张图输出两套 txt
  (与 scripts/class0_points.txt 同格式, 兼容 PoseUtils::readCorners())。

核心几何: 目标为平面贴图, 其上所有点共享同一单应矩阵 H (K·[r1 r2 t]),
  因此缩放/旋转/平移时特征点自动同步变化 (需求 5)。

生成目录:
    <out>/
    ├── synthetic_tiny/     # State 1 远 (class0 面积 ≤800)
    ├── synthetic_bc/       # State 2 中 (面积 801~40000)
    ├── synthetic_akaze/    # State 3 中近 (面积 >40000)
    ├── synthetic_dual/     # State 4 近 (面积 ≥490000, 画布 1280x960)
    ├── mono_tiny/ mono_bc/ mono_akaze/ mono_dual/   # 拷贝双目左图 + 对应 txt
    └── manifest.json       # 每张图路径 + 场景 + 姿态参数 (供训练加载)

每帧文件 (以 synthetic_tiny 为例):
    left_000.png / right_000.png
    left_000_class0.txt / left_000_class1.txt
    right_000_class0.txt / right_000_class1.txt
    ... (mono 场景拷贝 left_*.png 与 left_* 两个 txt)

用法:
    python scripts/generate_synthetic_dataset.py [--out tests/data/fixtures_rich] \
        [--n 50] [--scale-jitter 0.15] [--pitch-max 10] [--roll-max 10] [--seed 0]

依赖:
    numpy, opencv-python
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

try:
    import cv2
except ImportError as e:
    print("[ERROR] 需要 opencv-python: pip install opencv-python numpy", file=sys.stderr)
    raise e

# ============================================================================
# 输入配置 (硬编码, 与 scripts/annotate_points.py / generate_assets.py 对齐)
# ============================================================================
TARGET_IMG = r"data/big/img_1.png"                  # 目标图 (即 class0)
BACKGROUND_IMG = r"data/small/gj06_image_0317.jpg"  # 背景图
CLASS0_POINTS = r"scripts/class0_points.txt"        # class0 特征点 (10 点)
CLASS1_POINTS = r"scripts/class1_points.txt"        # class1 特征点 (10 点)

# ============================================================================
# 常量 (复用 generate_assets.py 的图像参数与分类)
# ============================================================================
FOCAL_LEN = 1000.0            # 针孔模型焦距 (px); 尺度由 f/tz 决定
MARGIN = 2.0                  # 目标边缘与画布边距 (px), 保证不被截断

# 各状态 class0 目标短边目标尺寸 (参考 generate_assets.py STATE_SIZES + dual 720)
STATE_SIZES = {"tiny": 20, "bc": 150, "akaze": 210, "dual": 720}

# 各状态短边有效范围: 保证面积落在该 State 的判定区间, 且不大于画布可容纳上限。
#   tiny   面积 ≤800          → 短边 ≤28.3
#   bc     面积 801~40000      → 短边 28.3~200
#   akaze  面积 >40000         → 短边 >200; 上限 330 保证 640x480 画布内任意偏航不截断
#   dual   面积 ≥490000        → 短边 ≥700; 大目标偏航受限 (拒绝采样 + 卡迪纳角兜底)
SIZE_BANDS = {
    "tiny":  (1.0, 28.0),
    "bc":    (29.0, 199.0),
    "akaze": (201.0, 330.0),
    "dual":  (700.0, 760.0),
}

# 双目视差 (px, 参考 generate_assets.py DISP)
DISP = {"tiny": 4, "bc": 8, "akaze": 16, "dual": 20}
# 各状态画布尺寸 (参考 generate_assets.py IMG_W/IMG_H 与 DUAL_W/DUAL_H)
CANVAS = {"tiny": (640, 480), "bc": (640, 480),
          "akaze": (640, 480), "dual": (1280, 960)}

# 特征点解析正则 (与 C++ PoseUtils::readCorners 同构)
_POINT_RE = re.compile(r"Corner_\d+:\s*([-\d.]+),\s*([-\d.]+)")


# ----------------------------------------------------------------------------
# 输入加载
# ----------------------------------------------------------------------------
def load_points(path):
    """解析 'Corner_N: X, Y' 特征点文件, 返回 [(x, y), ...]。"""
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _POINT_RE.search(line)
            if m:
                pts.append((float(m.group(1)), float(m.group(2))))
    if not pts:
        raise ValueError(f"未解析到任何特征点: {path}")
    return pts


# ----------------------------------------------------------------------------
# 平面单应性 (homography)
# ----------------------------------------------------------------------------
def rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """R = Rz(yaw) · Rx(pitch) · Ry(roll)。

    yaw:   绕光轴 (Z) 旋转 = 目标平面内旋转 (鸟瞰主旋转, 0-360°)
    pitch: 绕 X 轴俯仰 (前后透视), roll: 绕 Y 轴滚转 (左右透视), 均为小角度
    """
    yaw, pitch, roll = map(np.deg2rad, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return Rz @ Rx @ Ry


def make_homography(canvas_size, target_size, size, yaw_deg, pitch_deg, roll_deg,
                    center, f, disp=0.0):
    """构造目标像素坐标 → 画布像素坐标的 3x3 单应矩阵。

    模型: 目标平面 Z=0, 世界坐标 X=px-Wt/2, Y=py-Ht/2 (绕目标中心旋转)。
      H = K · [r1 r2 t] · T_center
    其中 r1,r2 为 R 的前两列, t 使目标中心投影到 center, 尺度由 tz=f*short/S 控制。
    disp>0 时右图再整体右移 disp px (双目视差, 与 generate_assets.py 语义一致)。
    """
    Wc, Hc = canvas_size
    Wt, Ht = target_size
    cx_img, cy_img = Wc / 2.0, Hc / 2.0
    f = float(f)
    short = min(Wt, Ht)
    tz = f * short / float(size)          # 深度: 使目标短边视尺寸 ≈ size

    R = rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    K = np.array([[f, 0.0, cx_img],
                  [0.0, f, cy_img],
                  [0.0, 0.0, 1.0]])
    cx0, cy0 = center
    tx = (cx0 - cx_img) * tz / f           # 目标中心 → (cx0, cy0)
    ty = (cy0 - cy_img) * tz / f

    H = np.column_stack([R[:, 0], R[:, 1], np.array([tx, ty, tz])])
    H = K @ H

    # 目标像素坐标中心化 (绕目标中心旋转)
    T = np.array([[1.0, 0.0, -Wt / 2.0],
                  [0.0, 1.0, -Ht / 2.0],
                  [0.0, 0.0, 1.0]])
    H = H @ T

    if abs(disp) > 1e-9:                   # 双目视差: 图像平面右移 disp
        Td = np.array([[1.0, 0.0, disp],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
        H = Td @ H
    return H


def project_points(H, pts):
    """把目标像素点集 [(px,py),...] 经 H 投影到画布坐标 [(u,v),...]。"""
    out = []
    for (px, py) in pts:
        p = H @ np.array([px, py, 1.0])
        out.append((float(p[0] / p[2]), float(p[1] / p[2])))
    return out


def projected_aabb(H, target_size):
    """目标源图 4 角经 H 投影后的轴对齐包围盒 (x0, y0, x1, y1)。"""
    Wt, Ht = target_size
    xs, ys = [], []
    for (px, py) in [(0, 0), (Wt, 0), (0, Ht), (Wt, Ht)]:
        p = H @ np.array([px, py, 1.0])
        xs.append(p[0] / p[2])
        ys.append(p[1] / p[2])
    return min(xs), min(ys), max(xs), max(ys)


# ----------------------------------------------------------------------------
# 姿态采样 (不截断保证)
# ----------------------------------------------------------------------------
def feasible_center(canvas_size, target_size, size, yaw, pitch, roll, disp, f,
                    margin, rng):
    """返回一个使目标(含右图视差位移)完整落在画布内的中心; 不可行则 None。"""
    Wc, Hc = canvas_size
    cx_img, cy_img = Wc / 2.0, Hc / 2.0
    # 参考中心 = 画布中心; AABB 随中心刚性平移
    H0 = make_homography(canvas_size, target_size, size, yaw, pitch, roll,
                         (cx_img, cy_img), f, 0.0)
    Hr = make_homography(canvas_size, target_size, size, yaw, pitch, roll,
                         (cx_img, cy_img), f, float(disp))
    x0, y0, x1, y1 = projected_aabb(H0, target_size)
    xr0, yr0, xr1, yr1 = projected_aabb(Hr, target_size)
    x0 = min(x0, xr0); y0 = min(y0, yr0)
    x1 = max(x1, xr1); y1 = max(y1, yr1)

    lo_x = margin - x0 + cx_img
    hi_x = Wc - margin - x1 + cx_img
    lo_y = margin - y0 + cy_img
    hi_y = Hc - margin - y1 + cy_img
    if lo_x > hi_x or lo_y > hi_y:
        return None
    return (float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y)))


def sample_pose(canvas_size, target_size, scene, jitter, pitch_max, roll_max,
                disp, f, rng, margin=MARGIN, max_tries=100):
    """随机采样 (size, yaw, pitch, roll, center), 保证目标不被截断。

    yaw 全程均匀 0-360°; 若大目标在特定偏航不可放 (如 dual 45°), 拒绝采样重试,
    多次失败后回退到卡迪纳角 (0/90/180/270) + 最小尺寸, 保证必有一解。
    """
    center_size = STATE_SIZES[scene]
    lo, hi = SIZE_BANDS[scene]
    for _ in range(max_tries):
        size = float(np.clip(center_size * rng.uniform(1.0 - jitter, 1.0 + jitter),
                             lo, hi))
        yaw = float(rng.uniform(0.0, 360.0))
        pitch = float(rng.uniform(-pitch_max, pitch_max))
        roll = float(rng.uniform(-roll_max, roll_max))
        c = feasible_center(canvas_size, target_size, size, yaw, pitch, roll,
                            disp, f, margin, rng)
        if c is not None:
            return size, yaw, pitch, roll, c
    # 兜底: 最小尺寸 + 卡迪纳角
    size = lo
    for yaw in (0.0, 90.0, 180.0, 270.0):
        c = feasible_center(canvas_size, target_size, size, yaw, 0.0, 0.0,
                            disp, f, margin, rng)
        if c is not None:
            return size, yaw, 0.0, 0.0, c
    raise RuntimeError(f"scene={scene} size={size} 在画布 {canvas_size} 上无法不截断放置")


# ----------------------------------------------------------------------------
# 渲染与写盘
# ----------------------------------------------------------------------------
def render(bg, target, H):
    """把目标图经 H 透视投影后 opaque 覆盖到背景上 (整图矩形粘贴)。"""
    img = bg.copy()
    Wc, Hc = img.shape[1], img.shape[0]
    Ht, Wt = target.shape[:2]
    warped = cv2.warpPerspective(target, H, (Wc, Hc),
                                 flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((Ht, Wt), 255, np.uint8), H, (Wc, Hc),
                               flags=cv2.INTER_NEAREST, borderValue=0)
    img[mask > 0] = warped[mask > 0]
    return img


def write_points(path, pts, label, image_size):
    """按 scripts/class0_points.txt 格式写特征点文件 (readCorners 兼容)。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {label} Feature Points (synthetic)\n")
        f.write("# Format: Corner_Index: X, Y\n")
        f.write(f"# Image Size: {int(image_size[0])}x{int(image_size[1])}\n")
        for i, (x, y) in enumerate(pts):
            f.write(f"Corner_{i}: {x:.2f}, {y:.2f}\n")


# ============================================================================
# CLI
# ============================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="生成 3 轴旋转 + 全图特征点标注的合成训练数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default="tests/data/fixtures_rich",
                    help="输出根目录")
    ap.add_argument("--n", type=int, default=50, help="每场景帧数")
    ap.add_argument("--scale-jitter", type=float, default=0.15,
                    help="各状态短边尺寸抖动比例 (会在该状态有效面积带内截断)")
    ap.add_argument("--pitch-max", type=float, default=10.0,
                    help="俯仰角最大值 (度, 均匀 ±)")
    ap.add_argument("--roll-max", type=float, default=10.0,
                    help="滚转角最大值 (度, 均匀 ±)")
    ap.add_argument("--seed", type=int, default=None, help="随机种子 (复现用)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)

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
    manifest = {
        "version": 1,
        "note": "每张合成图的 class0/class1 特征点 txt + 姿态参数 (透视单应生成)",
        "n_class0_points": len(class0),
        "n_class1_points": len(class1),
        "images": [],
    }

    for scene in STATE_SIZES:
        canvas = CANVAS[scene]
        disp = DISP[scene]
        bg = cv2.resize(bg_src, canvas, interpolation=cv2.INTER_AREA)
        out_dir = os.path.join(args.out, f"synthetic_{scene}")
        mono_dir = os.path.join(args.out, f"mono_{scene}")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(mono_dir, exist_ok=True)

        for i in range(args.n):
            size, yaw, pitch, roll, center = sample_pose(
                canvas, target_size, scene, args.scale_jitter,
                args.pitch_max, args.roll_max, disp, FOCAL_LEN, rng)
            Hl = make_homography(canvas, target_size, size, yaw, pitch, roll,
                                 center, FOCAL_LEN, 0.0)
            Hr = make_homography(canvas, target_size, size, yaw, pitch, roll,
                                 center, FOCAL_LEN, float(disp))

            left_img = render(bg, target, Hl)
            right_img = render(bg, target, Hr)

            base = f"{i:03d}"
            rel = f"synthetic_{scene}"
            pose = {"size": round(size, 2), "yaw_deg": round(yaw, 2),
                    "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2),
                    "center": [round(center[0], 2), round(center[1], 2)]}

            cv2.imwrite(os.path.join(out_dir, f"left_{base}.png"), left_img)
            cv2.imwrite(os.path.join(out_dir, f"right_{base}.png"), right_img)
            c0l = project_points(Hl, class0)
            c1l = project_points(Hl, class1)
            c0r = project_points(Hr, class0)
            c1r = project_points(Hr, class1)
            write_points(os.path.join(out_dir, f"left_{base}_class0.txt"),
                         c0l, "Class 0", canvas)
            write_points(os.path.join(out_dir, f"left_{base}_class1.txt"),
                         c1l, "Class 1", canvas)
            write_points(os.path.join(out_dir, f"right_{base}_class0.txt"),
                         c0r, "Class 0", canvas)
            write_points(os.path.join(out_dir, f"right_{base}_class1.txt"),
                         c1r, "Class 1", canvas)

            manifest["images"].extend([
                {"path": f"{rel}/left_{base}.png", "scene": scene,
                 "mode": "stereo_left", "frame": i, "pose": pose,
                 "class0_txt": f"{rel}/left_{base}_class0.txt",
                 "class1_txt": f"{rel}/left_{base}_class1.txt"},
                {"path": f"{rel}/right_{base}.png", "scene": scene,
                 "mode": "stereo_right", "frame": i, "pose": pose,
                 "class0_txt": f"{rel}/right_{base}_class0.txt",
                 "class1_txt": f"{rel}/right_{base}_class1.txt"},
                {"path": f"mono_{scene}/left_{base}.png", "scene": scene,
                 "mode": "mono", "frame": i, "pose": pose,
                 "class0_txt": f"mono_{scene}/left_{base}_class0.txt",
                 "class1_txt": f"mono_{scene}/left_{base}_class1.txt"},
            ])

        # mono 场景 = 拷贝双目左图 + 左图两个 txt
        for f in os.listdir(out_dir):
            if f.startswith("left_"):
                shutil.copy(os.path.join(out_dir, f), os.path.join(mono_dir, f))

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 摘要
    print(f"[OK] 合成数据集生成完毕 -> {os.path.abspath(args.out)}")
    for scene in STATE_SIZES:
        d = os.path.join(args.out, f"synthetic_{scene}")
        n = len([f for f in os.listdir(d) if f.endswith(".png")])
        print(f"  synthetic_{scene}: {n} 张 (每张带 class0/class1 txt)")
    print(f"  mono_*: {args.n} 张/场景 (左图拷贝)")
    print(f"  manifest.json: {len(manifest['images'])} 条记录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
