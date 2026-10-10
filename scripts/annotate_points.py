#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式特征点标注脚本 (带缩放/平移)
====================
功能:
  - 展示一张目标图像 (路径在 [配置区] 硬编码)
  - 鼠标左键依次点击 2×N 个特征点:
      前 N 个 = class 0 (目标整体) 特征点
      后 N 个 = class 1 (中心) 特征点
  - 按现有 data/NewMuBan(reordered)/0_degrees.txt 的文件格式,
    将两类点分别写入两个 txt (与 PoseUtils::readCorners() 兼容)

输出格式:
  # Class 0 Feature Points (manual annotation)
  # Image: data/clean_target2.png
  # Format: Corner_Index: X, Y
  # Image Size: 753x756
  Corner_0: 123.00, 456.00
  ...

交互按键:
  左键          添加一个点 (按当前阶段归入 class 0 / class 1)
  右键 或 u     撤销最后一个点 (跨 class 回退)
  滚轮          以光标为中心缩放 (精确落点: 放大后点击)
  中键拖拽      平移视图
  + / -         以窗口中心缩放
  f             复位视图 (适配窗口大小)
  q / ESC       取消, 不写文件
  集满 2×N 个点后自动写盘退出

依赖:
    numpy, opencv-python

用法:
    python scripts/annotate_points.py
    (输入/输出路径在下方 [配置区] 硬编码, 换图直接改常量)
"""

import os
import sys

try:
    import cv2
    import numpy as np
except ImportError as e:
    print("[ERROR] 需要 opencv-python numpy: pip install opencv-python numpy", file=sys.stderr)
    raise e

# ============================================================================
# [配置区] 输入/输出路径硬编码 —— 换图只改这里
# ============================================================================
IMAGE_PATH = "data/clean_target2.png"   # 要标注的模板图
OUT_DIR    = "scripts"                  # 输出目录 (自动创建)
OUT0_NAME  = "class0_points.txt"        # class 0 特征点输出文件
OUT1_NAME  = "class1_points.txt"        # class 1 特征点输出文件
NUM_PER_CLASS = 10                      # 每类点数 (共 2×N 个)
WIN_SIZE   = (1200, 800)                # 显示窗口 (宽, 高)

# class 0 / class 1 的标记颜色 (BGR)
_COLOR_CLASS0 = (0, 0, 255)   # 红 = 目标整体
_COLOR_CLASS1 = (0, 255, 0)   # 绿 = 目标中心

_ZOOM_STEP   = 1.25          # 滚轮/键盘单次缩放倍率
_ZOOM_MIN    = 0.1
_ZOOM_MAX    = 16.0


def _write_points(path, points, image_path, shape, label):
    """按 0_degrees.txt 格式写坐标文件。

    仅行 `Corner_N: X, Y` 会被 C++ 端 PoseUtils::readCorners() 解析,
    表头 `#` 注释行可自由调整。
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {label} Feature Points (manual annotation)\n")
        f.write(f"# Image: {image_path}\n")
        f.write("# Format: Corner_Index: X, Y\n")
        f.write(f"# Image Size: {shape[1]}x{shape[0]}\n")
        for i, (x, y) in enumerate(points):
            f.write("Corner_{}: {:.2f}, {:.2f}\n".format(i, float(x), float(y)))


class Annotator:
    """单窗口交互式特征点采集器 (视口缩放/平移)。

    视图模型: 图像坐标 (ix, iy) ↔ 显示坐标 (dx, dy)
        dx = (ix - ox) * s ;  ix = dx / s + ox
    即 (ox, oy) 为视口左上角对应的图像坐标, s 为缩放倍率。
    所有点记录一律存图像坐标 (float), 写盘精度 0.01 px。
    """

    def __init__(self, image_path, num_per_class=10, win_size=WIN_SIZE):
        self.image_path = image_path
        self.base = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.base is None:
            raise IOError(f"无法读取图像: {image_path}")
        self.num_per_class = num_per_class
        self.win_w, self.win_h = win_size
        self.points = []          # [(x, y), ...] 图像坐标
        self.finished = False

        # 视口状态
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self.cursor = None        # 最近光标位置 (显示坐标)
        self._pan_last = None     # 中键拖拽上一位置
        self._fit_view()

    # ------------------------------------------------------------ 视图控制
    def _fit_view(self):
        """复位: 整图适配窗口并居中。"""
        h, w = self.base.shape[:2]
        self.scale = min(1.0, self.win_w / w, self.win_h / h)
        self.ox = (w - self.win_w / self.scale) / 2.0
        self.oy = (h - self.win_h / self.scale) / 2.0

    def _zoom(self, factor, cx, cy):
        """以显示坐标 (cx, cy) 为锚点缩放 factor 倍。"""
        ix = cx / self.scale + self.ox
        iy = cy / self.scale + self.oy
        s = self.scale * factor
        self.scale = max(_ZOOM_MIN, min(_ZOOM_MAX, s))
        self.ox = ix - cx / self.scale
        self.oy = iy - cy / self.scale

    def _img_to_disp(self, ix, iy):
        return ((ix - self.ox) * self.scale,
                (iy - self.oy) * self.scale)

    # ------------------------------------------------------------------ 绘制
    def _redraw(self):
        s = self.scale
        M = np.float32([[s, 0.0, -self.ox * s],
                        [0.0, s, -self.oy * s]])
        display = cv2.warpAffine(self.base, M, (self.win_w, self.win_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(30, 30, 30))

        # 光标十字线 (辅助精确落点)
        if self.cursor is not None:
            cx, cy = self.cursor
            cv2.line(display, (cx, 0), (cx, self.win_h - 1), (80, 80, 80), 1)
            cv2.line(display, (0, cy), (self.win_w - 1, cy), (80, 80, 80), 1)

        # 已标点: 小圈 + 小号数字 (显示空间绘制, 不随缩放变大)
        for i, (ix, iy) in enumerate(self.points):
            cls = 0 if i < self.num_per_class else 1
            color = _COLOR_CLASS0 if cls == 0 else _COLOR_CLASS1
            idx = i % self.num_per_class          # 各 class 内序号 0..N-1
            dx, dy = self._img_to_disp(ix, iy)
            if -20 <= dx <= self.win_w + 20 and -20 <= dy <= self.win_h + 20:
                p = (int(round(dx)), int(round(dy)))
                cv2.circle(display, p, 3, color, 1)
                cv2.putText(display, str(idx), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 顶部状态提示
        total = len(self.points)
        if total < self.num_per_class:
            text = f"Class 0: click point {total}/{self.num_per_class}"
            color = _COLOR_CLASS0
        elif total < 2 * self.num_per_class:
            text = f"Class 1: click point {total - self.num_per_class}/{self.num_per_class}"
            color = _COLOR_CLASS1
        else:
            text = "Complete! Writing files..."
            color = (255, 255, 255)
        cv2.putText(display, text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(display,
                    f"zoom={self.scale:.2f}x  wheel=zoom  mid-drag=pan  "
                    f"L=add  R/u=undo  f=fit  q/ESC=cancel",
                    (10, self.win_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)
        return display

    # ------------------------------------------------------------------ 回调
    def _on_mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN and not self.finished:
            if len(self.points) < 2 * self.num_per_class:
                ix = x / self.scale + self.ox
                iy = y / self.scale + self.oy
                h, w = self.base.shape[:2]
                if 0.0 <= ix < w and 0.0 <= iy < h:
                    self.points.append((ix, iy))
                    if len(self.points) == 2 * self.num_per_class:
                        self.finished = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.finished = False   # 撤销后脱离完成态, 允许继续补点
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = _ZOOM_STEP if flags > 0 else 1.0 / _ZOOM_STEP
            self._zoom(factor, x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._pan_last = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_MBUTTONDOWN):
            if self._pan_last is not None:
                dx = x - self._pan_last[0]
                dy = y - self._pan_last[1]
                self.ox -= dx / self.scale
                self.oy -= dy / self.scale
                self._pan_last = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            self._pan_last = None

    # ------------------------------------------------------------------ 主流程
    def run(self, out_dir, out0, out1, window_name="Annotate Points"):
        os.makedirs(out_dir, exist_ok=True)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._on_mouse)

        print(f"[INFO] 展示图像: {os.path.abspath(self.image_path)}")
        print(f"[INFO] 每类采集 {self.num_per_class} 个点, 共 "
              f"{2 * self.num_per_class} 个")
        print("[INFO] 滚轮缩放(光标锚点) / 中键拖拽平移 / 左键添加 / "
              "右键或u撤销 / f复位视图 / q/ESC 取消")
        print("[INFO] 建议: 滚轮放大到角点清晰后落点, 精度更高")
        try:
            while not self.finished:
                cv2.imshow(window_name, self._redraw())
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):          # q / ESC 取消
                    print("[INFO] 已取消, 不写文件")
                    return False
                elif key == ord("u"):              # u 撤销
                    if self.points:
                        self.points.pop()
                        self.finished = False
                elif key in (ord("f"), ord("F")):  # 复位视图
                    self._fit_view()
                elif key in (ord("+"), ord("=")):  # 以窗口中心放大
                    self._zoom(_ZOOM_STEP, self.win_w / 2, self.win_h / 2)
                elif key in (ord("-"), ord("_")):  # 以窗口中心缩小
                    self._zoom(1.0 / _ZOOM_STEP, self.win_w / 2, self.win_h / 2)
        except KeyboardInterrupt:
            print("\n[INFO] 已中断, 不写文件")
            return False
        finally:
            cv2.destroyAllWindows()

        path0 = os.path.join(out_dir, out0)
        path1 = os.path.join(out_dir, out1)
        _write_points(path0, self.points[:self.num_per_class],
                      self.image_path, self.base.shape, "Class 0")
        _write_points(path1, self.points[self.num_per_class:],
                      self.image_path, self.base.shape, "Class 1")
        print(f"[OK] class 0 ({len(self.points[:self.num_per_class])} 点) -> "
              f"{os.path.abspath(path0)}")
        print(f"[OK] class 1 ({len(self.points[self.num_per_class:])} 点) -> "
              f"{os.path.abspath(path1)}")
        return True


# ============================================================================
# 入口 (路径见文件头 [配置区])
# ============================================================================
def main():
    if NUM_PER_CLASS < 1:
        print("[ERROR] NUM_PER_CLASS 必须 >= 1", file=sys.stderr)
        return 1
    try:
        annotator = Annotator(IMAGE_PATH, num_per_class=NUM_PER_CLASS,
                              win_size=WIN_SIZE)
    except IOError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    ok = annotator.run(OUT_DIR, OUT0_NAME, OUT1_NAME)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
