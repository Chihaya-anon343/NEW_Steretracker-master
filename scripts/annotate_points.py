#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式特征点标注脚本
====================
功能:
  - 展示一张目标图像 (默认 data/big/img_1.png)
  - 鼠标左键依次点击 2×N 个特征点:
      前 N 个 = class 0 (目标整体) 特征点
      后 N 个 = class 1 (中心) 特征点
  - 按现有 data/NewMuBan(reordered)/0_degrees.txt 的文件格式,
    将两类点分别写入两个 txt (与 PoseUtils::readCorners() 兼容)

输出格式:
  # Class 0 Feature Points (manual annotation)
  # Image: data/big/img_1.png
  # Format: Corner_Index: X, Y
  # Image Size: 798x786
  Corner_0: 123.00, 456.00
  ...

交互按键:
  左键          添加一个点 (按当前阶段归入 class 0 / class 1)
  右键 或 u     撤销最后一个点 (跨 class 回退)
  q / ESC       取消, 不写文件
  集满 2×N 个点后自动写盘退出

依赖:
    numpy, opencv-python

用法:
    python scripts/annotate_points.py
    python scripts/annotate_points.py --image data/big/img_1.png \
        --out-dir scripts --out0 class0_points.txt --out1 class1_points.txt
"""

import argparse
import os
import sys

try:
    import cv2
except ImportError as e:
    print("[ERROR] 需要 opencv-python: pip install opencv-python numpy", file=sys.stderr)
    raise e

# class 0 / class 1 的标记颜色 (BGR)
_COLOR_CLASS0 = (0, 0, 255)   # 红 = 目标整体
_COLOR_CLASS1 = (0, 255, 0)   # 绿 = 目标中心


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
    """单窗口交互式特征点采集器。

    state.points: [(x, y), ...] 追加式记录; 前 N 个归 class 0, 后 N 个归 class 1。
    """

    def __init__(self, image_path, num_per_class=10):
        self.image_path = image_path
        self.base = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.base is None:
            raise IOError(f"无法读取图像: {image_path}")
        self.num_per_class = num_per_class
        self.points = []          # [(x, y), ...]
        self.finished = False

    # ------------------------------------------------------------------ 绘制
    def _redraw(self):
        display = self.base.copy()
        h, w = display.shape[:2]

        for i, (x, y) in enumerate(self.points):
            cls = 0 if i < self.num_per_class else 1
            color = _COLOR_CLASS0 if cls == 0 else _COLOR_CLASS1
            idx = i % self.num_per_class          # 各 class 内序号 0..N-1
            cv2.circle(display, (x, y), 6, color, 2)
            cv2.putText(display, str(idx), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, "L-click=add  R-click/u=undo  q/ESC=cancel",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        return display

    # ------------------------------------------------------------------ 回调
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.finished:
            if len(self.points) < 2 * self.num_per_class:
                self.points.append((x, y))
                if len(self.points) == 2 * self.num_per_class:
                    self.finished = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.finished = False   # 撤销后脱离完成态, 允许继续补点

    # ------------------------------------------------------------------ 主流程
    def run(self, out_dir, out0, out1, window_name="Annotate Points"):
        os.makedirs(out_dir, exist_ok=True)
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

        print(f"[INFO] 展示图像: {os.path.abspath(self.image_path)}")
        print(f"[INFO] 每类采集 {self.num_per_class} 个点, 共 "
              f"{2 * self.num_per_class} 个; 左键添加, 右键/u 撤销, q/ESC 取消")
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
# CLI
# ============================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="交互式标注目标图上的 class 0 / class 1 特征点, 按 "
                    "0_degrees.txt 格式输出两个 txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", default="data/big/img_1.png",
                    help="要标注的目标图像路径")
    ap.add_argument("--out-dir", default="scripts",
                    help="输出目录 (自动创建)")
    ap.add_argument("--out0", default="class0_points.txt",
                    help="class 0 特征点输出文件名")
    ap.add_argument("--out1", default="class1_points.txt",
                    help="class 1 特征点输出文件名")
    ap.add_argument("--num-per-class", type=int, default=10,
                    help="每类采集点数 (共 2×N 个)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.num_per_class < 1:
        print("[ERROR] --num-per-class 必须 >= 1", file=sys.stderr)
        return 1
    try:
        annotator = Annotator(args.image, num_per_class=args.num_per_class)
    except IOError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    ok = annotator.run(args.out_dir, args.out0, args.out1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
