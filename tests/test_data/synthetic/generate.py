#!/usr/bin/env python3
"""
Steretracker 合成测试图像生成脚本

生成三类测试图像及其配套 Ground Truth 数据：
  - TinyTarget     : 200×200 px 灰度图，含 50×50 正方形板，多角度 / 多对比度 / 带遮挡
  - BinaryCorner   : 300×300 px 灰度图，含 10 角点多边形板，基于 data/NewMuBan(reordered)/0_degrees.txt
  - AKAZE          : 640×480 px 灰度纹理图，基于 Perlin 噪声 / 随机纹理 + 模板 ROI

输出目录结构：
  tests/test_data/synthetic/
    ├── tinytarget/
    │   ├── tt_000.png / tt_000.json  (GT 角点)
    │   ├── tt_015.png / ...
    │   ├── tt_030.png / ...
    │   ├── tt_045.png / ...
    │   ├── tt_060.png / ...
    │   ├── tt_075.png / ...
    │   ├── tt_090.png / ...
    │   ├── tt_lowcontrast.png / ...
    │   └── tt_occluded.png / ...
    ├── binarycorner/
    │   ├── bc_000.png / bc_000.json
    │   ├── bc_015.png / ...
    │   ├── bc_030.png / ...
    │   ├── bc_045.png / ...
    │   ├── bc_060.png / ...
    │   └── bc_noisy.png / ...
    └── akaze/
        ├── ak_texture.png          (512×384 纹理板)
        ├── ak_template.png         (裁剪的模板区域)
        └── ak_gt.json              (模板区域坐标)
"""

import cv2
import numpy as np
import json
import os
import math
import random

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "test_data", "synthetic")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "data", "NewMuBan(reordered)")

os.makedirs(os.path.join(OUTPUT_DIR, "tinytarget"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "binarycorner"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "akaze"), exist_ok=True)

print(f"输出目录: {OUTPUT_DIR}")
print(f"模板目录: {TEMPLATE_DIR}")


# ============================================================
# 辅助函数
# ============================================================

def get_rotated_rect_corners(center, size, angle_deg):
    """获取 RotatedRect 的 4 个角点（顺序: TL, TR, BR, BL）"""
    rect = (center, size, angle_deg)
    box = cv2.boxPoints(rect)
    # 转换为标准顺序: 按 (x+y) 排序...
    box = sorted(box, key=lambda p: p[0] + p[1])
    tl = box[0]
    br = box[3]
    mid = sorted(box[1:3], key=lambda p: p[0] - p[1], reverse=True)
    tr = mid[0]
    bl = mid[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rotate_points(points, angle_deg, center):
    """绕指定中心旋转点集"""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    translated = points - center
    rotated = translated @ R.T
    return rotated + center


def save_image_and_gt(filepath, img, gt_data):
    """保存图像和 GT JSON"""
    cv2.imwrite(filepath, img)
    json_path = filepath.replace(".png", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return json_path


# ============================================================
# 1. TinyTarget 合成图像生成
# ============================================================

def generate_tinytarget():
    """生成 TinyTarget 测试集"""
    out_dir = os.path.join(OUTPUT_DIR, "tinytarget")
    print("\n[1/3] 生成 TinyTarget 测试图像...")

    IMG_W, IMG_H = 200, 200
    BOARD_SIZE = (50, 50)          # 正方形板 (width, height) px
    BOARD_CENTER = (IMG_W//2, IMG_H//2)  # 图像中心

    rotations = [0, 15, 30, 45, 60, 75, 90]

    for rot_deg in rotations:
        # 标准对比度
        img = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
        corners = get_rotated_rect_corners(BOARD_CENTER, BOARD_SIZE, rot_deg)
        cv2.drawContours(img, [np.int32(corners)], -1, 220, -1)

        fpath = os.path.join(out_dir, f"tt_{rot_deg:03d}.png")
        save_image_and_gt(fpath, img, {
            "type": "tinytarget",
            "image_size": [IMG_W, IMG_H],
            "board_center": list(BOARD_CENTER),
            "board_size": list(BOARD_SIZE),
            "rotation_deg": rot_deg,
            "gt_corners_all": corners.astype(float),
            "background_gray": 128,
            "target_gray": 220
        })
        print(f"  -> tt_{rot_deg:03d}.png (rotation={rot_deg}°)")

    # 低对比度
    img_low = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    corners_low = get_rotated_rect_corners(BOARD_CENTER, BOARD_SIZE, 0)
    cv2.drawContours(img_low, [np.int32(corners_low)], -1, 158, -1)  # 差 30
    fpath = os.path.join(out_dir, "tt_lowcontrast.png")
    save_image_and_gt(fpath, img_low, {
        "type": "tinytarget", "image_size": [IMG_W, IMG_H],
        "board_center": list(BOARD_CENTER), "board_size": list(BOARD_SIZE),
        "rotation_deg": 0, "gt_corners_all": corners_low.astype(float),
        "background_gray": 128, "target_gray": 158, "contrast": "low"
    })
    print("  -> tt_lowcontrast.png (contrast=30)")

    # 部分遮挡（右下 25%）
    img_occ = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    corners_occ = get_rotated_rect_corners(BOARD_CENTER, BOARD_SIZE, 0)
    cv2.drawContours(img_occ, [np.int32(corners_occ)], -1, 220, -1)
    # 右下角覆盖为背景色
    img_occ[150:200, 150:200] = 128
    fpath = os.path.join(out_dir, "tt_occluded.png")
    save_image_and_gt(fpath, img_occ, {
        "type": "tinytarget", "image_size": [IMG_W, IMG_H],
        "board_center": list(BOARD_CENTER), "board_size": list(BOARD_SIZE),
        "rotation_deg": 0, "gt_corners_all": corners_occ.astype(float),
        "occlusion": "bottom_right_25pct"
    })
    print("  -> tt_occluded.png (25% bottom-right occlusion)")

    print(f"  TinyTarget 完成，共 {len(rotations) + 2} 张图像")


# ============================================================
# 2. BinaryCorner 合成图像生成
# ============================================================

def load_0degree_corners():
    """从 data/NewMuBan(reordered)/0_degrees.txt 加载 0° 角点坐标"""
    templ_path = os.path.join(TEMPLATE_DIR, "0_degrees.txt")
    corners = []
    with open(templ_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            corners.append([x, y])
    return np.array(corners, dtype=np.float32)


def generate_binarycorner():
    """生成 BinaryCorner 测试集"""
    out_dir = os.path.join(OUTPUT_DIR, "binarycorner")
    print("\n[2/3] 生成 BinaryCorner 测试图像...")

    # 加载 0° 模板角点
    try:
        tmpl_corners = load_0degree_corners()
        print(f"  加载 0_degrees.txt: {len(tmpl_corners)} 个角点")
    except Exception as e:
        print(f"  ⚠ 无法加载模板角点: {e}")
        print("  将使用默认正 10 边形")
        n = 10
        r = 50
        angles = np.linspace(-np.pi/2, 3*np.pi/2, n, endpoint=False)
        tmpl_corners = np.column_stack([r * np.cos(angles) + 100, r * np.sin(angles) + 100]).astype(np.float32)

    # 计算模板包围盒并居中
    min_xy = tmpl_corners.min(axis=0)
    max_xy = tmpl_corners.max(axis=0)
    tmpl_size = max_xy - min_xy
    tmpl_center = (min_xy + max_xy) / 2.0

    # 目标图像参数
    IMG_W, IMG_H = 300, 300
    BOARD_CENTER = (IMG_W // 2, IMG_H // 2)

    # 将模板角点居中并缩放到约 100×100 像素区域
    scale_factor = 100.0 / max(tmpl_size[0], tmpl_size[1])
    normalized_corners = (tmpl_corners - tmpl_center) * scale_factor + np.array(BOARD_CENTER)

    rotations = [0, 15, 30, 45, 60]

    for rot_deg in rotations:
        img = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)

        # 旋转角点
        rotated_corners = rotate_points(normalized_corners, rot_deg, np.array(BOARD_CENTER))

        # 绘制填充多边形
        cv2.fillPoly(img, [np.int32(rotated_corners)], 230)

        fpath = os.path.join(out_dir, f"bc_{rot_deg:03d}.png")
        save_image_and_gt(fpath, img, {
            "type": "binarycorner",
            "image_size": [IMG_W, IMG_H],
            "board_center": list(BOARD_CENTER),
            "rotation_deg": rot_deg,
            "gt_corners": rotated_corners.astype(float),
            "num_corners": len(rotated_corners),
            "background_gray": 128,
            "target_gray": 230
        })
        print(f"  -> bc_{rot_deg:03d}.png (rotation={rot_deg}°, {len(rotated_corners)} corners)")

    # 噪声版（附加随机小斑点）
    img_noisy = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    rotated_corners_0 = rotate_points(normalized_corners, 0, np.array(BOARD_CENTER))
    cv2.fillPoly(img_noisy, [np.int32(rotated_corners_0)], 230)
    # 添加 3 个随机小斑点作为噪声
    for _ in range(3):
        rx, ry = np.random.randint(20, IMG_W-20), np.random.randint(20, IMG_H-20)
        cv2.circle(img_noisy, (rx, ry), np.random.randint(5, 15), 200, -1)
    fpath = os.path.join(out_dir, "bc_noisy.png")
    save_image_and_gt(fpath, img_noisy, {
        "type": "binarycorner",
        "image_size": [IMG_W, IMG_H],
        "board_center": list(BOARD_CENTER),
        "rotation_deg": 0,
        "gt_corners": rotated_corners_0.astype(float),
        "num_corners": len(rotated_corners_0),
        "noise": "3_random_blobs"
    })
    print("  -> bc_noisy.png (with 3 random noise blobs)")

    print(f"  BinaryCorner 完成，共 {len(rotations) + 1} 张图像")


# ============================================================
# 3. AKAZE 纹理合成图像生成
# ============================================================

def simple_perlin_noise(size, scale=4, seed=42):
    """简化的类 Perlin 噪声纹理生成（使用多尺度高斯噪声叠加）"""
    np.random.seed(seed)
    w, h = size
    noise = np.zeros((h, w), dtype=np.float32)
    for octave, freq in enumerate([4, 8, 16, 32, 64]):
        # 生成低频噪声
        coarse_h, coarse_w = max(1, h // freq), max(1, w // freq)
        coarse = np.random.randn(coarse_h, coarse_w).astype(np.float32)
        # 上采样并平滑
        fine = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        fine = cv2.GaussianBlur(fine, (max(3, freq//2)*2+1, max(3, freq//2)*2+1), 0)
        amplitude = 1.0 / (2 ** octave)
        noise += fine * amplitude
    # 归一化到 [0, 255]
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)


def generate_akaze():
    """生成 AKAZE 纹理测试集"""
    out_dir = os.path.join(OUTPUT_DIR, "akaze")
    print("\n[3/3] 生成 AKAZE 纹理测试图像...")

    IMG_W, IMG_H = 512, 384
    TPL_W, TPL_H = 200, 150        # 模板区域
    TPL_OFFSET_X, TPL_OFFSET_Y = 50, 60  # 模板左上角在图像中的位置

    # 生成随机纹理
    texture = simple_perlin_noise((IMG_W, IMG_H), scale=6, seed=42)

    # 可选：叠加棋盘格增加结构化特征
    checker_y, checker_x = np.mgrid[0:IMG_H, 0:IMG_W]
    checker = ((checker_x // 30 + checker_y // 30) % 2).astype(np.uint8) * 40
    texture = np.clip(texture.astype(np.int16) + checker.astype(np.int16), 0, 255).astype(np.uint8)

    # 使用高斯模糊使纹理更自然
    texture = cv2.GaussianBlur(texture, (3, 3), 0.5)

    fpath_full = os.path.join(out_dir, "ak_texture.png")
    cv2.imwrite(fpath_full, texture)
    print(f"  -> ak_texture.png ({IMG_W}×{IMG_H}, grayscale)")

    # 裁剪模板区域
    template_roi = texture[TPL_OFFSET_Y:TPL_OFFSET_Y+TPL_H, TPL_OFFSET_X:TPL_OFFSET_X+TPL_W]
    fpath_tpl = os.path.join(out_dir, "ak_template.png")
    cv2.imwrite(fpath_tpl, template_roi)
    print(f"  -> ak_template.png ({TPL_W}×{TPL_H}, template ROI)")

    # 保存 GT
    gt_data = {
        "type": "akaze_texture",
        "full_image_size": [IMG_W, IMG_H],
        "template_roi": {
            "x": TPL_OFFSET_X,
            "y": TPL_OFFSET_Y,
            "width": TPL_W,
            "height": TPL_H
        },
        "physical_size_mm": {"width": 200.0, "height": 150.0}
    }
    with open(os.path.join(out_dir, "ak_gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)
    print("  -> ak_gt.json (GT metadata)")

    print("  AKAZE 完成，共 2 张图像")


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("Steretracker 合成测试图像生成器")
    print("=" * 60)

    generate_tinytarget()
    generate_binarycorner()
    generate_akaze()

    print("\n" + "=" * 60)
    print("全部生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()