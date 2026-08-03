#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成测试资产生成脚本
=====================
为 Steretracker 集成测试生成合成图像与 ROI 数据。

生成目录结构:
    tests/data/fixtures/
    ├── synthetic_tiny/          # TinyTarget 场景 (State 1, 面积 <= 800 px²)
    │   ├── left_000.png / right_000.png
    │   └── ...
    ├── synthetic_bc/            # BinaryCorner 场景 (State 2, 801~40000 px²)
    │   ├── left_000.png / right_000.png
    │   └── ...
    ├── synthetic_akaze/         # AKAZE_GPNP 场景 (State 3, >40000 px² 纹理丰富)
    │   ├── left_000.png / right_000.png
    │   └── ...
    ├── synthetic_dual/          # Dual-ROI 场景 (State 4, class0+class1, >=490000 px²)
    │   ├── left_000.png / right_000.png
    │   └── ...
    ├── mono_tiny/  mono_bc/  mono_akaze/   # 单目仅左图
    │   └── left_000.png ...
    └── manual_roi.json          # Debug 模式手动 ROI

用法:
    python tests/scripts/generate_assets.py [--out tests/data/fixtures]

依赖:
    numpy, opencv-python

说明:
    - 所有合成的目标均为深色背景上的白色目标
    - 右图为左图的水平位移副本 (视差 = disparity_px, 右移)
    - 目标中心纹理 (class 1 语义) 通过内部高反差矩形模拟
    - 生成的图像仅用作流水线冒烟/回归测试, 不追求真实相机成像
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

# ============================================================================
# 常量
# ============================================================================
IMG_W, IMG_H = 640, 480          # 图像尺寸
CENTER_X, CENTER_Y = IMG_W // 2, IMG_H // 2

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------

def make_canvas():
    """深色背景画布。"""
    return np.full((IMG_H, IMG_W, 3), 28, dtype=np.uint8)


def render_vertices(img, pts, color=255, thickness=2):
    """按顶点顺序画闭合多边形。pts: Nx2 int。"""
    pts = pts.astype(np.int32).reshape(-1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def fill_polygon(img, pts, color=255):
    """填充多边形。"""
    pts = pts.astype(np.int32).reshape(-1, 2)
    cv2.fillPoly(img, [pts], color=color)


def centroid(pts):
    return pts.mean(axis=0)


def star_10(side=140, center=None):
    """生成 10 角点星形多边形顶点 (BinaryCorner 场景, 类似 10 角点标识板)。

    顶点: 0=90°, 1=126°, ... 交替 10 个点, 两个半径交替。
    """
    center = np.array(center if center is not None else [0, 0], dtype=np.float64)
    outer = side / 2.0
    inner = outer * 0.55
    angles = np.deg2rad(np.linspace(90.0, 90.0 - 360.0, 11)[:-1])
    pts = []
    for i, a in enumerate(angles):
        r = outer if i % 2 == 0 else inner
        pts.append(center + r * np.array([np.cos(a), np.sin(a)]))
    return np.array(pts, dtype=np.float64)


def regular_ngon(n, radius, center=None, rot_deg=0.0):
    """正 n 边形顶点, 顶点 0 位于正上方, 顺时针。"""
    center = np.array(center if center is not None else [0, 0], dtype=np.float64)
    angles = np.deg2rad(rot_deg + 90.0 - np.arange(n) * 360.0 / n)
    return np.vstack([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])


def checkerboard(side=120, cells=8):
    """生成 side×side 棋盘格 (纹理丰富, AKAZE 场景)。"""
    cell = max(2, side // cells)
    board = np.zeros((side, side), dtype=np.uint8)
    for i in range(cells):
        for j in range(cells):
            if (i + j) % 2 == 0:
                board[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = 255
    return board


# ----------------------------------------------------------------------------
# 场景生成器
# ----------------------------------------------------------------------------

def gen_tiny_scene(frame_index, out_dir):
    """State 1: 远距离微小矩形目标 (TinyTarget)。

    目标: 白色实心方块, 面积 <= 800 px² (边长 <= ~28px)。
    视差: 4px。
    """
    side = 20
    x0 = CENTER_X + (frame_index % 5) * 2
    y0 = CENTER_Y - side // 2
    disp = 4

    for tag in ("left", "right"):
        img = make_canvas()
        x = x0 + (disp if tag == "right" else 0)
        cv2.rectangle(img, (x, y0), (x + side, y0 + side), 255, -1)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)


def gen_bc_scene(frame_index, out_dir):
    """State 2: 中等尺寸 10 角点星形目标 (BinaryCorner)。

    目标外接圆半径 ~70px → 面积 ~8000 px², 落在 801~40000 区间。
    视差: 8px。
    """
    side = 150
    disp = 8

    for tag in ("left", "right"):
        img = make_canvas()
        x0 = CENTER_X + (frame_index % 3) * 1.5
        pts = star_10(side=side, center=(x0 + (disp if tag == "right" else 0), CENTER_Y))
        fill_polygon(img, pts, 255)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)


def gen_akaze_scene(frame_index, out_dir):
    """State 3: 大尺寸纹理丰富目标 (AKAZE_GPNP)。

    目标: 背景上的棋盘格矩形 (~200x200 → 40000 px², 加白色边框确保面积 > 40000)。
    视差: 16px。
    """
    board_side = 180
    disp = 16

    for tag in ("left", "right"):
        img = make_canvas()
        x0 = CENTER_X - board_side // 2 + (frame_index % 3) * 2
        x0 += disp if tag == "right" else 0
        y0 = CENTER_Y - board_side // 2
        # 白色边框把棋盘格围起来, 增大整体面积 > 40000
        cv2.rectangle(img, (x0 - 8, y0 - 8), (x0 + board_side + 8, y0 + board_side + 8), 255, 6)
        board = checkerboard(board_side, cells=10)
        img[y0:y0 + board_side, x0:x0 + board_side] = board.reshape(board_side, board_side, 1)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)


def gen_dual_scene(frame_index, out_dir):
    """State 4: 大目标同时可见整体 (class0) 与中心 (class1)。

    - class0: 半个屏幕大的白色矩形外框 (~500x500, 面积 >= 490000)
    - class1: 中心白色实心方块 (~120x120), 语义: 目标中心
    视差: 20px。
    """
    outer = 520                      # 外框尺寸 (>490000 px² 含框面积)
    inner_box = 120
    disp = 20

    # 外框最左 x 保证完全在画布内
    x0 = max(8, IMG_W - outer - 8)
    y0 = max(8, IMG_H - outer - 8)

    for tag in ("left", "right"):
        img = make_canvas()
        dx = disp if tag == "right" else 0
        # class0 整体: 白底外框
        cv2.rectangle(img, (x0 + dx, y0), (x0 + outer + dx, y0 + outer), 255, 14)
        # class1 中心: 大区域白色(与背景对比强, 模拟中心纹理目标)
        cx = x0 + outer // 2 + dx
        cy = y0 + outer // 2
        cv2.rectangle(img, (cx - inner_box // 2, cy - inner_box // 2),
                      (cx + inner_box // 2, cy + inner_box // 2), 255, -1)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)


def gen_mono_scene(frame_index, src_dir, out_dir):
    """单目场景: 复用双目场景的左图, 仅复制 left_*.png。"""
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.startswith("left_"):
            src = os.path.join(src_dir, f)
            dst = os.path.join(out_dir, f)
            import shutil
            shutil.copy(src, dst)


def write_roi_json(out_dir):
    """Debug 模式手动 ROI (与 synthetic_bc 场景匹配, 覆盖 10 角点星形)。"""
    roi = {
        "image_size": {"width": IMG_W, "height": IMG_H},
        "left": {"x": 230, "y": 170, "width": 180, "height": 180},
        "right": {"x": 238, "y": 170, "width": 180, "height": 180},
        "scene": "synthetic_bc",
        "note": "覆盖 10 角点星形目标; 左图原点 (230,170), 右图因视差偏移"
    }
    with open(os.path.join(out_dir, "manual_roi.json"), "w", encoding="utf-8") as f:
        json.dump(roi, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="生成 Steretracker 集成测试合成资产")
    ap.add_argument("--out", default="tests/data/fixtures", help="输出目录")
    args = ap.parse_args()

    out_root = args.out
    scenes = {
        "synthetic_tiny": gen_tiny_scene,
        "synthetic_bc": gen_bc_scene,
        "synthetic_akaze": gen_akaze_scene,
        "synthetic_dual": gen_dual_scene,
    }

    n_frames = 3
    for name, fn in scenes.items():
        scene_dir = os.path.join(out_root, name)
        os.makedirs(scene_dir, exist_ok=True)
        for i in range(n_frames):
            fn(i, scene_dir)
        # 单目仅左图
        gen_mono_scene(None, scene_dir, os.path.join(out_root, f"mono_{name.replace('synthetic_', '')}"))

    write_roi_json(out_root)

    # 摘要
    print(f"[OK] 资产生成完毕 -> {os.path.abspath(out_root)}")
    total = 0
    for root, _dirs, files in os.walk(out_root):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            total += len(pngs)
            print(f"  {os.path.relpath(root, out_root)}: {len(pngs)} 张")
    print(f"  共 {total} 张图片 + manual_roi.json")


if __name__ == "__main__":
    main()