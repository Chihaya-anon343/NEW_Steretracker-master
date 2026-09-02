#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋盘格位姿解算脚本 (PnP, 批量)
==============================
基于已标定的相机内参, 遍历输入文件夹下所有图片, 对每张执行:
  findChessboardCorners → solvePnP(IPPE) → 平移向量 T (mm),
并把 T 写在可视化图 (*_pose.png) 的左上角 (角点 + 坐标轴叠加)。

坐标系约定 (与项目 C++ 端一致):
  - 相机坐标系: Z 前向, X 右, Y 下, 单位 mm
  - T = 棋盘格坐标系原点 (第一个检测到的内角点) 在相机系下的坐标
  - 棋盘格坐标系: Z=0 平面, 原点为检测到的第一个内角点

内参来源: 优先读取 calibrate_camera.py 输出的 INTRINSICS_FILE;
          文件不存在时使用下方硬编码兜底值 (占位! 务必先用标定结果覆盖)。

用法:
    1. 把待解算的棋盘格图片放入 IMAGE_DIR
    2. python scripts/solve_chessboard_pose.py
    3. 每张图生成同目录同名 *_pose.png (左上角标注 T)

依赖:
    numpy, opencv-python
"""

import glob
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
# 输入配置 (硬编码)
# ============================================================================
IMAGE_DIR = r"data/calib"                        # 待解算图片目录
IMAGE_GLOB = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
INTRINSICS_FILE = r"scripts/camera_intrinsics.json"

# 棋盘格规格: 必须与 calibrate_camera.py 一致 (7格x10格 → 内角点 9x6, 单格 18mm)
BOARD_CORNERS = (9, 6)   # 内角点数 (cols, rows)
SQUARE_SIZE_MM = 18.0

# 内参兜底值 (占位! camera_intrinsics.json 不存在时使用, 请用标定输出覆盖)
K_FALLBACK = np.array([
    [1000.0, 0.0, 640.0],
    [0.0, 1000.0, 512.0],
    [0.0, 0.0, 1.0],
])
DIST_FALLBACK = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

AXIS_LENGTH_MM = SQUARE_SIZE_MM * 3.0   # 可视化坐标轴长度
VIS_SUFFIX = "_pose.png"                # 可视化输出后缀 (重跑时自动跳过)

SUBPIX_WIN = (11, 11)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

# 左上角 T 文本样式 (双描边: 黑底 + 绿字)
TXT_POS = (12, 30)
TXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TXT_SCALE = 0.8
TXT_THICK = 2


def build_object_points():
    """棋盘格 object points (Z=0 平面, mm), 与 calibrate_camera.py 完全一致。"""
    cols, rows = BOARD_CORNERS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * SQUARE_SIZE_MM


def find_corners(gray):
    """找棋盘格内角点并亚像素精化; 失败返回 None。"""
    ok, corners = cv2.findChessboardCorners(gray, BOARD_CORNERS, FIND_FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)


def load_intrinsics():
    """优先读标定 JSON, 否则用硬编码兜底值。"""
    if os.path.isfile(INTRINSICS_FILE):
        with open(INTRINSICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if tuple(data.get("board_corners", [])) != BOARD_CORNERS or \
                abs(float(data.get("square_size_mm", 0.0)) - SQUARE_SIZE_MM) > 1e-6:
            print("[WARN] JSON 中的棋盘格规格与脚本硬编码不一致, 请检查!", file=sys.stderr)
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data["dist"], dtype=np.float64).reshape(1, -1)
        print(f"[INFO] 内参来自 {INTRINSICS_FILE} "
              f"(标定 RMS {data.get('rms_px', -1):.4f} px, {data.get('num_images', '?')} 张图)")
        return K, dist
    print(f"[WARN] 未找到 {INTRINSICS_FILE}, 使用硬编码兜底内参 (解算结果可能不准!)",
          file=sys.stderr)
    return K_FALLBACK.copy(), DIST_FALLBACK.copy().reshape(1, -1)


def process_image(path, K, dist, objp):
    """解算单张图: 检测角点 → PnP → 可视化 (左上角写 T)。成功返回 True。"""
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] 读取失败, 跳过: {path}")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = find_corners(gray)
    if corners is None:
        print(f"[WARN] 未检测到 {BOARD_CORNERS[0]}x{BOARD_CORNERS[1]} 内角点, 跳过: {path}")
        return False

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        print(f"[WARN] solvePnP(IPPE) 求解失败, 跳过: {path}")
        return False

    t = tvec.reshape(3)

    # 重投影 RMS (解算质量指标)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = np.linalg.norm(corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))

    # 可视化: 角点 + 坐标轴 + 左上角 T
    vis = img.copy()
    cv2.drawChessboardCorners(vis, BOARD_CORNERS, corners, True)
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, AXIS_LENGTH_MM)
    txt = f"T = ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) mm"
    cv2.putText(vis, txt, TXT_POS, TXT_FONT, TXT_SCALE, (0, 0, 0),
                TXT_THICK + 2, cv2.LINE_AA)
    cv2.putText(vis, txt, TXT_POS, TXT_FONT, TXT_SCALE, (0, 255, 0),
                TXT_THICK, cv2.LINE_AA)

    root, _ = os.path.splitext(path)
    vis_path = root + VIS_SUFFIX
    cv2.imwrite(vis_path, vis)

    print(f"[OK] {os.path.basename(path)}: RMS {rms:.3f} px, "
          f"T = ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) mm, "
          f"|T| = {np.linalg.norm(t):.2f} mm")
    return True


def main():
    K, dist = load_intrinsics()
    objp = build_object_points()

    paths = []
    for pat in IMAGE_GLOB:
        paths.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))
    paths = sorted(p for p in paths if not p.endswith(VIS_SUFFIX))
    if not paths:
        print(f"[ERROR] {IMAGE_DIR} 下没有图片, 请先放入待解算的棋盘格图片", file=sys.stderr)
        return 1

    print(f"[INFO] 待解算图片 {len(paths)} 张 (目录 {IMAGE_DIR}), "
          f"棋盘格 {BOARD_CORNERS[0]}x{BOARD_CORNERS[1]} 内角点, 单格 {SQUARE_SIZE_MM} mm\n")

    n_ok = 0
    for p in paths:
        if process_image(p, K, dist, objp):
            n_ok += 1

    print(f"\n===== 批量解算完成: 成功 {n_ok}/{len(paths)} 张 =====")
    print(f"[OK] 可视化 (左上角标注 T) 已保存到 {IMAGE_DIR}/*{VIS_SUFFIX}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
