#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成测试资产生成脚本
=====================
思路: 输入为 背景图 + 目标图(即 class0) + class1 坐标大小。
  1) class1 从目标图中按坐标裁剪;
  2) 目标图本身即 class0;
  3) 按测试所需大小等比缩放后, 布置到背景图中心区域。

生成目录结构:
    tests/data/fixtures/
    ├── synthetic_tiny/          # State 1 (TinyTarget, class0 短边 20px, 面积 <= 800 px²)
    ├── synthetic_bc/            # State 2 (BinaryCorner, class0 短边 150px, 801~40000 px²)
    ├── synthetic_akaze/         # State 3 (AKAZE, class0 短边 210px, >40000 px²)
    ├── synthetic_dual/          # State 4 (class0 外框 720x720 + class1 120x120, >=490000 px²)
    ├── synthetic_bc_class1/     # State 5 (仅 class1, 短边 120px, 801~40000 px²)
    ├── synthetic_akaze_class1/  # State 6 (仅 class1, 短边 200px, >40000 px²)
    ├── mono_tiny/ mono_bc/ mono_akaze/           # 单目仅左图 (class0-only 场景)
    ├── mono_bc_class1/ mono_akaze_class1/        # 单目仅左图 (class1-only 场景)
    ├── manual_roi.json          # Debug 模式手动 ROI (对齐 synthetic_bc 实际 bbox)
    └── rois.json                # 每帧每图 class0 / class1 ROI 表 (供后续读取)

用法:
    python tests/scripts/generate_assets.py [--out tests/data/fixtures]

输入配置:
    两个图片地址 与 class1 坐标大小 在脚本顶部"输入配置"区硬编码,
    运行前请修改为你的实际路径与坐标:
      TARGET_IMG       正视目标图路径 (即 class0)
      BACKGROUND_IMG   背景图路径 (自动 resize 到画布尺寸)
      CLASS1_RECT      class1 在 target.png 中的像素矩形 (x, y, width, height);
                       无 class1 时设为 None (dual/class1 场景跳过 class1)

依赖:
    numpy, opencv-python
"""

import argparse
import os
import shutil
import sys
import json
import numpy as np

try:
    import cv2
except ImportError as e:
    print("[ERROR] 需要 opencv-python: pip install opencv-python numpy", file=sys.stderr)
    raise e

# ============================================================================
# 输入配置 (硬编码, 运行前请修改为你的实际图片路径与 class1 坐标)
# ============================================================================
# 正视目标图路径 (即 class0)
TARGET_IMG = r"data/big/img_1.png"
# 背景图路径 (自动 resize 到画布尺寸)
BACKGROUND_IMG = r"data/small/gj06_image_0317.jpg"
# class1 在 target.png 中的像素矩形 (x, y, width, height); 无 class1 时设为 None
CLASS1_RECT = {"x": 359, "y": 417, "width": 60, "height": 60}

# ============================================================================
# 常量
# ============================================================================
IMG_W, IMG_H = 640, 480                        # 常规场景画布 (tiny / bc / akaze)
DUAL_W, DUAL_H = 1280, 960                     # dual 场景画布
CENTER_X, CENTER_Y = IMG_W // 2, IMG_H // 2

# 各 State 的 class0 短边目标尺寸与视差 (与 README §6 一致)
STATE_SIZES = {
    "tiny": 20,    # <= 800 px²
    "bc": 150,     # 801 ~ 40000 px²
    "akaze": 210,  # > 40000 px²
}
# 各 class1-only 场景的 class1 短边目标尺寸 (同分档; 可通过 CLASS1_ONLY_SIZES 覆盖)
CLASS1_ONLY_SIZES = {
    "bc": 120,     # 801 ~ 40000 px² (150 全部被 class1 遮蔽会增加误遮蔽风险, 故略小)
    "akaze": 200,  # > 40000 px²
}
DISP = {"tiny": 4, "bc": 8, "akaze": 16, "dual": 20}
N_FRAMES = 3


# ----------------------------------------------------------------------------
# 输入加载 (从脚本顶部硬编码配置读取)
# ----------------------------------------------------------------------------
def load_inputs():
    """读取脚本顶部硬编码的 TARGET_IMG / BACKGROUND_IMG / CLASS1_RECT。
    图片缺失或无法读取时直接报错。"""
    for label, p in (("TARGET_IMG", TARGET_IMG), ("BACKGROUND_IMG", BACKGROUND_IMG)):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"缺少输入图片: {label} = {p}\n"
                "请在脚本顶部'输入配置'区修改为你的实际图片路径。"
            )
    target = cv2.imread(TARGET_IMG, cv2.IMREAD_UNCHANGED)
    background = cv2.imread(BACKGROUND_IMG)
    if target is None:
        raise IOError(f"无法读取目标图: {TARGET_IMG}")
    if background is None:
        raise IOError(f"无法读取背景图: {BACKGROUND_IMG}")
    return target, background, CLASS1_RECT


def resize_bg(background, w, h):
    """背景图自适应缩放到画布尺寸。"""
    return cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)


def scale_paste(img, patch, center, size):
    """将 patch (class0 目标图或 class1 裁剪块) 等比缩放 (短边 = size) 后
    以 center 为中心粘贴到 img 画布。支持 3/4 通道 (4 通道做 alpha 混合)。
    返回实际粘贴的 bbox (x, y, w, h); 完全在画布外时返回 None。"""
    ph, pw = patch.shape[:2]
    if ph < 1 or pw < 1:
        return None
    scale = float(size) / float(min(ph, pw))
    nw = max(1, int(round(pw * scale)))
    nh = max(1, int(round(ph * scale)))
    resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = int(round(center[0] - nw / 2.0))
    y0 = int(round(center[1] - nh / 2.0))
    ih, iw = img.shape[:2]

    ex0, ey0 = max(0, x0), max(0, y0)
    ex1, ey1 = min(iw, x0 + nw), min(ih, y0 + nh)
    if ex1 <= ex0 or ey1 <= ey0:
        return None

    sx, sy = ex0 - x0, ey0 - y0
    src = resized[sy:sy + (ey1 - ey0), sx:sx + (ex1 - ex0)]
    roi = img[ey0:ey1, ex0:ex1]

    if src.ndim == 3 and src.shape[2] == 4:
        alpha = src[:, :, 3:4].astype(np.float32) / 255.0
        rgb = src[:, :, :3].astype(np.float32)
        img[ey0:ey1, ex0:ex1] = (rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    else:
        img[ey0:ey1, ex0:ex1] = src[:, :, :3] if src.ndim == 3 else src
    return (ex0, ey0, ex1 - ex0, ey1 - ey0)


def rect_or_none(bbox):
    """bbox (x,y,w,h) → 同构 dict; None → None。"""
    if not bbox:
        return None
    return {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}


# ----------------------------------------------------------------------------
# 场景生成
# ----------------------------------------------------------------------------
def gen_class0_scene(out_dir, target, background, scene, frame_index):
    """State 1/2/3: class0 (目标图) 按 STATE_SIZES 缩放到背景中心,
    右图水平右移 disp。返回 (左 bbox, 右 bbox)。"""
    size, disp = STATE_SIZES[scene], DISP[scene]
    bg = resize_bg(background, IMG_W, IMG_H)
    left0 = right0 = None
    for tag in ("left", "right"):
        img = bg.copy()
        cx = CENTER_X + (frame_index % 3) * 2
        if tag == "right":
            cx += disp
        bbox = scale_paste(img, target, (cx, CENTER_Y), size)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)
        if tag == "left":
            left0 = bbox
        else:
            right0 = bbox
    return left0, right0


def gen_class1_scene(out_dir, target, background, class1_rect, scene, frame_index):
    """State 5/6: 仅 class1 (从目标图 CLASS1_RECT 裁剪) 缩放到背景中心,
    不粘贴整张 class0 目标图。返回 (左 bbox, 右 bbox)。"""
    size, disp = CLASS1_ONLY_SIZES[scene], DISP[scene]
    th, tw = target.shape[:2]
    px = min(max(int(class1_rect.get("x", 0)), 0), max(tw - 1, 0))
    py = min(max(int(class1_rect.get("y", 0)), 0), max(th - 1, 0))
    pw = min(int(class1_rect.get("width", 0)), tw - px)
    ph = min(int(class1_rect.get("height", 0)), th - py)
    if pw <= 0 or ph <= 0:
        raise ValueError(f"class1 裁剪区域无效: rect={class1_rect}, target={tw}x{th}")
    patch = target[py:py + ph, px:px + pw]

    bg = resize_bg(background, IMG_W, IMG_H)
    left0 = right0 = None
    for tag in ("left", "right"):
        img = bg.copy()
        cx = CENTER_X + (frame_index % 3) * 2
        if tag == "right":
            cx += disp
        bbox = scale_paste(img, patch, (cx, CENTER_Y), size)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)
        if tag == "left":
            left0 = bbox
        else:
            right0 = bbox
    return left0, right0


def gen_dual_scene(out_dir, target, background, class1_rect, frame_index):
    """State 4: 画布 1280x960; class0 (目标图) 缩放至 720x720 框 (面积 >= 490000),
    中心叠加 class1 (从目标图 class1_rect 裁剪, resize 120x120)。
    返回 ((class0_left, class0_right), (class1_left, class1_right))。"""
    outer, inner, disp = 720, 120, DISP["dual"]
    bg = resize_bg(background, DUAL_W, DUAL_H)
    # 最左 x 保证左右图均不越界: ox + disp + outer <= DUAL_W
    ox = max(8, DUAL_W - outer - disp - 8)
    oy = max(8, DUAL_H - outer - 8)

    c0_left = c0_right = None
    c1_left = c1_right = None
    for tag in ("left", "right"):
        img = bg.copy()
        cx = ox + (disp if tag == "right" else 0) + outer // 2
        cy = oy + outer // 2
        # class0: 缩放至 720 框 (画白框圈定 bbox)
        c0_bbox = scale_paste(img, target, (cx, cy), outer)
        cv2.rectangle(img, (cx - outer // 2, cy - outer // 2),
                      (cx + outer // 2, cy + outer // 2), 255, 14)
        # class1: 从目标图裁剪后缩小, 粘贴到 class0 中心
        c1_bbox = None
        if class1_rect:
            th, tw = target.shape[:2]
            px = min(max(int(class1_rect.get("x", 0)), 0), max(tw - 1, 0))
            py = min(max(int(class1_rect.get("y", 0)), 0), max(th - 1, 0))
            pw = min(int(class1_rect.get("width", 0)), tw - px)
            ph = min(int(class1_rect.get("height", 0)), th - py)
            if pw > 0 and ph > 0:
                patch = target[py:py + ph, px:px + pw]
                c1_bbox = scale_paste(img, patch, (cx, cy), inner)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{frame_index:03d}.png"), img)
        if tag == "left":
            c0_left, c1_left = c0_bbox, c1_bbox
        else:
            c0_right, c1_right = c0_bbox, c1_bbox
    return (c0_left, c0_right), (c1_left, c1_right)


def gen_mono_scene(out_dir, src_dir):
    """单目场景: 复制双目场景的左图 left_*.png。"""
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.startswith("left_"):
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))


def write_roi_json(out_dir, bc_left0, bc_right0):
    """Debug 模式手动 ROI (对齐 synthetic_bc 实际 bbox)。"""
    if not bc_left0 or not bc_right0:
        bc_left0 = bc_right0 = (245, 165, 150, 150)
    roi = {
        "image_size": {"width": IMG_W, "height": IMG_H},
        "left": {"x": bc_left0[0], "y": bc_left0[1],
                 "width": bc_left0[2], "height": bc_left0[3]},
        "right": {"x": bc_right0[0], "y": bc_right0[1],
                  "width": bc_right0[2], "height": bc_right0[3]},
        "scene": "synthetic_bc",
        "note": "覆盖合成 bc 目标 (class0 短边 150px); 右图较左图水平偏移 8px",
    }
    with open(os.path.join(out_dir, "manual_roi.json"), "w", encoding="utf-8") as f:
        json.dump(roi, f, ensure_ascii=False, indent=2)


def write_rois_json(out_root, roi_data):
    """生成 rois.json: 每场景每帧左右图的 class0 / class1 ROI 表。

    roi_data 结构:
        {
          "<scene>": {
              "image_size": {"width": W, "height": H},
              "frames": [
                  {"frame": i,
                   "left":  {"class0": rect|None, "class1": rect|None},
                   "right": {"class0": rect|None, "class1": rect|None}},
                  ...
              ]
          }
        }
    """
    doc = {
        "version": 2,
        "note": "每场景每帧左右图的 class0/class1 ROI (像素坐标, 左上角原点); "
                "class0/class1 为 None 表示该帧该图不存在对应类别",
        "scenes": roi_data,
    }
    with open(os.path.join(out_root, "rois.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="生成 Steretracker 集成测试合成资产 (基于输入图)")
    ap.add_argument("--out", default="tests/data/fixtures", help="输出目录")
    args = ap.parse_args()

    target, background, class1_rect = load_inputs()
    out_root = args.out

    bc_left0 = bc_right0 = None
    # 记录单目场景对应的双目父场景, 用于复用左帧 ROI
    mono_parents = {
        "mono_tiny": "synthetic_tiny",
        "mono_bc": "synthetic_bc",
        "mono_akaze": "synthetic_akaze",
        "mono_bc_class1": "synthetic_bc_class1",
        "mono_akaze_class1": "synthetic_akaze_class1",
    }
    roi_data = {}

    # --- 双目: 各 scene 每帧收集 bbox ---
    stereo_bbox = {}   # scene -> {frame: {"left": (c0, c1), "right": (c0, c1)}}

    for i in range(N_FRAMES):
        # State 1: TinyTarget (class0 短边 20px), 视差 4px
        scene = "synthetic_tiny"
        os.makedirs(os.path.join(out_root, scene), exist_ok=True)
        stereo_bbox.setdefault(scene, {})
        l0, r0 = gen_class0_scene(os.path.join(out_root, scene), target, background, "tiny", i)
        stereo_bbox[scene][i] = {"left": (l0, None), "right": (r0, None)}

        # State 2: BinaryCorner (class0 短边 150px), 视差 8px
        scene = "synthetic_bc"
        os.makedirs(os.path.join(out_root, scene), exist_ok=True)
        stereo_bbox.setdefault(scene, {})
        l0, r0 = gen_class0_scene(os.path.join(out_root, scene), target, background, "bc", i)
        stereo_bbox[scene][i] = {"left": (l0, None), "right": (r0, None)}
        if i == 0:
            bc_left0, bc_right0 = l0, r0

        # State 3: AKAZE (class0 短边 210px, >40000 px²), 视差 16px
        scene = "synthetic_akaze"
        os.makedirs(os.path.join(out_root, scene), exist_ok=True)
        stereo_bbox.setdefault(scene, {})
        l0, r0 = gen_class0_scene(os.path.join(out_root, scene), target, background, "akaze", i)
        stereo_bbox[scene][i] = {"left": (l0, None), "right": (r0, None)}

        # State 5: 仅 class1 BinaryCorner (短边 120px), 视差 8px
        scene = "synthetic_bc_class1"
        os.makedirs(os.path.join(out_root, scene), exist_ok=True)
        stereo_bbox.setdefault(scene, {})
        l1, r1 = gen_class1_scene(os.path.join(out_root, scene), target, background,
                                  class1_rect, "bc", i)
        stereo_bbox[scene][i] = {"left": (None, l1), "right": (None, r1)}

        # State 6: 仅 class1 AKAZE (短边 200px, >40000 px²), 视差 16px
        scene = "synthetic_akaze_class1"
        os.makedirs(os.path.join(out_root, scene), exist_ok=True)
        stereo_bbox.setdefault(scene, {})
        l1, r1 = gen_class1_scene(os.path.join(out_root, scene), target, background,
                                  class1_rect, "akaze", i)
        stereo_bbox[scene][i] = {"left": (None, l1), "right": (None, r1)}

        # State 4: Dual-ROI (class0 720x720 + class1 120x120), 视差 20px
        scene = "synthetic_dual"
        os.makedirs(os.path.join(out_root, scene), exist_ok=True)
        stereo_bbox.setdefault(scene, {})
        (c0_l, c0_r), (c1_l, c1_r) = gen_dual_scene(
            os.path.join(out_root, scene), target, background, class1_rect, i)
        stereo_bbox[scene][i] = {
            "left": (c0_l, c1_l), "right": (c0_r, c1_r),
        }

    # --- 构建各场景 frames ROI 表 ---
    for scene, frames_map in stereo_bbox.items():
        frames = []
        for i in sorted(frames_map):
            left_c0, left_c1 = frames_map[i]["left"]
            right_c0, right_c1 = frames_map[i]["right"]
            frames.append({
                "frame": i,
                "left": {"class0": rect_or_none(left_c0),
                         "class1": rect_or_none(left_c1)},
                "right": {"class0": rect_or_none(right_c0),
                          "class1": rect_or_none(right_c1)},
            })
        is_dual = (scene == "synthetic_dual")
        w, h = (DUAL_W, DUAL_H) if is_dual else (IMG_W, IMG_H)
        roi_data[scene] = {"image_size": {"width": w, "height": h}, "frames": frames}

    # --- 单目仅左图 (复制左图; ROI 复用双目父场景左帧) ---
    for mono, parent in mono_parents.items():
        gen_mono_scene(os.path.join(out_root, mono), os.path.join(out_root, parent))
        parent_frames = roi_data[parent]["frames"]
        frames = []
        for fr in parent_frames:
            frames.append({
                "frame": fr["frame"],
                "left": fr["left"],
                "right": {"class0": None, "class1": None},
            })
        roi_data[mono] = {
            "image_size": roi_data[parent]["image_size"],
            "frames": frames,
        }

    write_roi_json(out_root, bc_left0, bc_right0)
    write_rois_json(out_root, roi_data)

    # 摘要
    print(f"[OK] 资产生成完毕 -> {os.path.abspath(out_root)}")
    total = 0
    for root, _dirs, files in os.walk(out_root):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            total += len(pngs)
            print(f"  {os.path.relpath(root, out_root)}: {len(pngs)} 张")
    print(f"  共 {total} 张图片 + manual_roi.json + rois.json")
    print(f"  输入: TARGET_IMG={TARGET_IMG}")
    print(f"        BACKGROUND_IMG={BACKGROUND_IMG}")
    print(f"        CLASS1_RECT={CLASS1_RECT}")


if __name__ == "__main__":
    main()