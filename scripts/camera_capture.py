#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头采集与帧导出脚本
======================
功能:
  - 读取本机摄像头 (USB / 内置 webcam) 的实时画面
  - 提供 4 种使用方式: 预览 / 单帧抓拍 / 连拍 / 流式回调导出
  - 预留图片帧导出接口, 供后续项目 (如 C++ Steretracker 单目流水线)
    直接消费导出的帧序列

设计要点:
  - 面向对象: CameraCapture 封装全部摄像头操作, 可被 import 复用
  - 回调导出: start_streaming(callback) 把「导出去哪」完全交给调用方,
    内置 StreamingExporter 只是默认实现; 用户可自定义回调(存数据库/送网络/喂C++)
  - 帧命名规范: {prefix}_{index:04d}.png 与 C++ DirectoryStereoSource /
    SequenceSource 的 pattern "*.png" 兼容, 可直接作为 input_system.source_dir
  - 单文件零新依赖: 仅 numpy + opencv-python (项目已有)

依赖:
    numpy, opencv-python

用法:
    # 预览 (s=截图, q/ESC=退出)
    python scripts/camera_capture.py --preview

    # 单帧抓拍
    python scripts/camera_capture.py --capture -o data/camera_frames/snapshot.png

    # 连拍 10 张 (每 5 帧取 1 张)
    python scripts/camera_capture.py --burst 10 --interval 5 -o data/camera_frames/

    # 流式导出: 持续采集, 每 10 帧存一张, Ctrl+C 停止
    python scripts/camera_capture.py --stream --interval 10 -o data/camera_frames/

    # 指定摄像头索引与分辨率
    python scripts/camera_capture.py --preview --camera 1 --width 1920 --height 1080
"""

import argparse
import os
import sys
import time

try:
    import cv2
except ImportError as e:
    print("[ERROR] 需要 opencv-python: pip install opencv-python numpy", file=sys.stderr)
    raise e

# 扩展名 → OpenCV imwrite 编码 (png 无损, jpg 有损压缩质量)
_ENCODER = {
    ".png": [cv2.IMWRITE_PNG_COMPRESSION, 3],
    ".jpg": [cv2.IMWRITE_JPEG_QUALITY, 95],
    ".bmp": [],
}


def _write_frame(frame, path):
    """按扩展名选择编码方式写帧, 目录不存在时自动创建。返回写成功与否。"""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".png"
    params = _ENCODER.get(ext, _ENCODER[".png"])
    return bool(cv2.imwrite(path, frame, params))


class StreamingExporter:
    """内置帧导出回调 (可自定义替换)。

    以 {output_dir}/{prefix}_{index:04d}{ext} 命名逐帧写盘。
    命名与 C++ DirectoryStereoSource / SequenceSource 的 "*.png" 兼容。
    """

    def __init__(self, output_dir, prefix="frame", ext=".png"):
        self.output_dir = output_dir
        self.prefix = prefix
        self.ext = ext if ext.startswith(".") else "." + ext

    def __call__(self, frame, index):
        """回调接口: 接收 (frame, index), 返回保存的文件路径。"""
        path = os.path.join(self.output_dir, f"{self.prefix}_{index:04d}{self.ext}")
        if not _write_frame(frame, path):
            raise IOError(f"写入帧失败: {path}")
        return path


class CameraCapture:
    """摄像头采集封装。

    典型用法:
        cam = CameraCapture(camera_id=0, width=1280, height=720)
        if not cam.open():
            sys.exit("无法打开摄像头")
        cam.capture_single("data/snapshot.png")
        cam.close()
    """

    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._streaming = False
        self._stream_index = 0

    # ------------------------------------------------------------------ 基础
    def open(self):
        """打开摄像头并设置分辨率/FPS。成功返回 True。"""
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            print(f"[ERROR] 无法打开摄像头 index={self.camera_id} "
                  f"(可能被占用或不存在)", file=sys.stderr)
            return False
        # 设置参数; 某些摄像头不支持目标分辨率时会回退到其原生值
        if self.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 等几帧让摄像头完成自动曝光/白平衡
        for _ in range(5):
            self._cap.grab()
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] 摄像头已打开: index={self.camera_id} "
              f"实际分辨率 {w}x{h}, {f:.0f} fps")
        return True

    def close(self):
        """释放摄像头。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._streaming = False

    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    def get_resolution(self):
        """返回实际分辨率 (width, height)。"""
        if not self.is_opened():
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    # ------------------------------------------------------------------ 读取
    def read_frame(self):
        """读取一帧 BGR 图像; 失败返回 None。"""
        if not self.is_opened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    # ------------------------------------------------------------ 抓拍 / 连拍
    def capture_single(self, output_path):
        """单帧抓拍, 存到 output_path。成功返回 True。"""
        frame = self.read_frame()
        if frame is None:
            print(f"[ERROR] 读取帧失败, 未保存 {output_path}", file=sys.stderr)
            return False
        ok = _write_frame(frame, output_path)
        if ok:
            print(f"[OK] 已保存: {os.path.abspath(output_path)} "
                  f"({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"[ERROR] 写盘失败: {output_path}", file=sys.stderr)
        return ok

    def capture_burst(self, output_dir, count=10, interval=0):
        """连拍 count 张, 每两张之间跳过 interval 帧 (减少连续帧重复)。

        返回保存成功的文件路径列表。"""
        os.makedirs(output_dir, exist_ok=True)
        saved = []
        for i in range(count):
            frame = self.read_frame()
            if frame is None:
                print(f"[ERROR] 第 {i} 帧读取失败, 提前终止", file=sys.stderr)
                break
            path = os.path.join(output_dir, f"frame_{i:04d}.png")
            if _write_frame(frame, path):
                saved.append(path)
            # 跳帧: 丢弃 interval 帧以拉大时间间隔
            for _ in range(interval):
                self._cap.grab()
            time.sleep(0.01)
        print(f"[OK] 连拍完成: {len(saved)}/{count} 张 -> {os.path.abspath(output_dir)}")
        return saved

    # ------------------------------------------------------------ 流式回调
    def start_streaming(self, callback, interval_frames=1):
        """持续采集并回调。

        callback(frame: np.ndarray, index: int) -> None
          - 每 interval_frames 帧调用一次 callback
          - index 从 0 起单调递增
          - 回调抛异常 / 按 Ctrl+C 都会停止并抛出
        """
        if not self.is_opened():
            raise RuntimeError("摄像头未打开, 请先调用 open()")
        self._streaming = True
        self._stream_index = 0
        try:
            while self._streaming:
                frame = self.read_frame()
                if frame is None:
                    continue
                if self._stream_index % interval_frames == 0:
                    callback(frame, self._stream_index // interval_frames)
                self._stream_index += 1
        except KeyboardInterrupt:
            print("\n[INFO] 已停止流式采集 (Ctrl+C)")
            raise
        finally:
            self._streaming = False

    def stop_streaming(self):
        """请求停止流式采集 (start_streaming 的循环会在下一帧退出)。"""
        self._streaming = False

    # ------------------------------------------------------------------ 预览
    def preview(self, window_name="Camera Preview", save_dir="data/camera_frames"):
        """实时预览窗口。

        按键:
          s    截图保存到 save_dir
          q/ESC 退出预览
        """
        if not self.is_opened():
            print("[ERROR] 摄像头未打开, 无法预览", file=sys.stderr)
            return
        os.makedirs(save_dir, exist_ok=True)
        shot_count = 0
        print(f"[INFO] 预览中: 按 's' 保存截图到 {save_dir}, 按 'q'/ESC 退出")
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("[ERROR] 读取帧失败", file=sys.stderr)
                    break
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):           # q / ESC
                    break
                elif key == ord("s"):               # s 截图
                    path = os.path.join(save_dir, f"shot_{shot_count:04d}.png")
                    if _write_frame(frame, path):
                        shot_count += 1
                        print(f"[OK] 截图保存: {os.path.abspath(path)}")
        except KeyboardInterrupt:
            print("\n[INFO] 预览已退出")
        finally:
            cv2.destroyAllWindows()
        print(f"[INFO] 预览结束, 共保存 {shot_count} 张截图")


# ============================================================================
# CLI
# ============================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="读取本机摄像头并导出图片帧 (预留导出接口)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="摄像头索引 (0=第一个)")
    ap.add_argument("--width", type=int, default=1280, help="请求的帧宽 (px)")
    ap.add_argument("--height", type=int, default=720, help="请求的帧高 (px)")
    ap.add_argument("--fps", type=int, default=30, help="请求的帧率")
    ap.add_argument("-o", "--output", default="data/camera_frames",
                    help="输出目录 (capture/burst/stream) 或文件路径 (capture)")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="实时预览窗口")
    mode.add_argument("--capture", action="store_true", help="单帧抓拍")
    mode.add_argument("--burst", type=int, metavar="N", help="连拍 N 张")
    mode.add_argument("--stream", action="store_true", help="流式回调导出 (Ctrl+C 停止)")

    ap.add_argument("--interval", type=int, default=5,
                    help="burst/stream 模式: 每 interval 帧取一张")
    ap.add_argument("--prefix", default="frame", help="导出帧名前缀 (默认 frame)")
    ap.add_argument("--ext", default=".png", help="导出帧扩展名 (.png/.jpg/.bmp)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    cam = CameraCapture(camera_id=args.camera,
                        width=args.width, height=args.height, fps=args.fps)
    if not cam.open():
        sys.exit(1)

    try:
        if args.preview:
            cam.preview(save_dir=args.output)
        elif args.capture:
            out = args.output
            if os.path.isdir(out) or not os.path.splitext(out)[1]:
                out = os.path.join(out, "snapshot.png")
            cam.capture_single(out)
        elif args.burst is not None:
            cam.capture_burst(args.output, count=args.burst, interval=args.interval)
        elif args.stream:
            exporter = StreamingExporter(args.output, prefix=args.prefix, ext=args.ext)
            print(f"[INFO] 流式导出中: 每 {args.interval} 帧存一张到 "
                  f"{os.path.abspath(args.output)} (Ctrl+C 停止)")
            try:
                cam.start_streaming(callback=exporter, interval_frames=args.interval)
            except KeyboardInterrupt:
                pass  # start_streaming 内部已打印提示
            saved = sorted(f for f in os.listdir(args.output)
                           if f.startswith(args.prefix))
            print(f"[OK] 流式导出结束, 共 {len(saved)} 张")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
