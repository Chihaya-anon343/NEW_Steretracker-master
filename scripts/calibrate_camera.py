#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机内参标定脚本 (棋盘格)
=========================
读取多张棋盘格图片, 用 cv2.calibrateCamera 标定相机内参 K 与畸变系数 dist。
结果打印为可复制格式, 并保存 JSON (供 solve_chessboard_pose.py 自动读取)。

用法:
    1. 把棋盘格标定图放入 IMAGE_DIR (默认 data/calib/, 建议 10~20 张
       不同角度/距离/位置的图片, 标定板应覆盖画面各区域)
    2. python scripts/calibrate_camera.py

依赖:
    numpy, opencv-python (标定图可用 scripts/camera_capture.py 拍摄)
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
IMAGE_DIR = r"data/calib"                        # 标定图目录
IMAGE_GLOB = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
OUTPUT_FILE = r"scripts/camera_intrinsics.json"  # 标定结果输出

# 棋盘格规格: 7格 x 10格 → 内角点 (10-1) x (7-1) = 9 x 6
# findChessboardCorners 与图片中棋盘的旋转方向无关, (cols, rows) 任一顺序均可
BOARD_CORNERS = (9, 6)   # 内角点数 (cols, rows)
SQUARE_SIZE_MM = 18.0    # 单格物理边长 (mm), 1.8cm

# 角点检测 / 精化参数
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_WIN = (11, 11)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

MIN_IMAGES = 3        # 低于此值无法标定 (建议 >= 10)
RMS_WARN_PX = 0.5     # 整体 RMS 高于此值提示标定质量欠佳


def build_object_points():
    """构造棋盘格 object points (Z=0 平面, 单位 mm)。

    点序必须与 findChessboardCorners 返回顺序一一对应: 检测结果按
    "rows 行 x cols 列" 行优先展开, 故 mgrid 首段取 cols (行内步进方向)。
    """
    cols, rows = BOARD_CORNERS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * SQUARE_SIZE_MM


def find_corners(gray):
    """在灰度图上找棋盘格内角点并亚像素精化; 失败返回 None。"""
    ok, corners = cv2.findChessboardCorners(gray, BOARD_CORNERS, FIND_FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)


def view_rms(objp, corners, K, dist, rvec, tvec):
    """单张图重投影 RMS (px)。"""
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = np.linalg.norm(corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    paths = []
    for pat in IMAGE_GLOB:
        paths.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))
    paths = sorted(paths)
    if not paths:
        print(f"[ERROR] {IMAGE_DIR} 下没有图片, 请先采集棋盘格标定图", file=sys.stderr)
        return 1

    objp = build_object_points()
    objpoints, imgpoints, used_paths, failed = [], [], [], []
    image_size = None

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] 读取失败, 跳过: {p}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            print(f"[WARN] 图像尺寸不一致 ({gray.shape[1]}x{gray.shape[0]}), 跳过: {p}")
            continue
        corners = find_corners(gray)
        if corners is None:
            failed.append(p)
            print(f"[WARN] 未检测到 {BOARD_CORNERS[0]}x{BOARD_CORNERS[1]} 内角点, 跳过: {p}")
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used_paths.append(p)

    n = len(objpoints)
    if n < MIN_IMAGES:
        print(f"[ERROR] 有效标定图仅 {n} 张 (<{MIN_IMAGES}), 无法标定; "
              f"建议 10~20 张不同姿态的图片", file=sys.stderr)
        return 1

    print(f"[INFO] 有效标定图 {n}/{len(paths)} 张, 图像尺寸 {image_size[0]}x{image_size[1]}, "
          f"棋盘格 {BOARD_CORNERS[0]}x{BOARD_CORNERS[1]} 内角点, 单格 {SQUARE_SIZE_MM} mm")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    print("\n===== 标定结果 =====")
    print(f"整体重投影 RMS: {rms:.4f} px")
    for p, obs, rvec, tvec in zip(used_paths, imgpoints, rvecs, tvecs):
        print(f"  [{view_rms(objp, obs, K, dist, rvec, tvec):7.4f} px] {os.path.basename(p)}")
    if rms > RMS_WARN_PX:
        print(f"[WARN] RMS > {RMS_WARN_PX}px, 建议检查图片清晰度/棋盘格规格是否正确后重拍")

    print(f"\nK =\n{K}")
    print(f"dist (k1 k2 p1 p2 k3) = {dist.ravel()}")

    print("\n----- 复制到 solve_chessboard_pose.py 的硬编码兜底区 -----")
    print("K_FALLBACK = np.array([")
    for row in K:
        print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],")
    print("])")
    print(f"DIST_FALLBACK = np.array({dist.ravel().tolist()})")

    d = os.path.dirname(OUTPUT_FILE)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "image_size": list(image_size),
            "rms_px": float(rms),
            "num_images": n,
            "board_corners": list(BOARD_CORNERS),
            "square_size_mm": SQUARE_SIZE_MM,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 标定结果已保存: {OUTPUT_FILE}")

    if failed:
        print(f"[INFO] {len(failed)} 张图未检测到棋盘格被跳过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
