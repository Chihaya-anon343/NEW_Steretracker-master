#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时预览 SSH 服务器上的 V4L2 相机，无需本机安装 OpenCV。

脚本让远端 ffmpeg 把相机帧编码为 MJPEG，经 OpenSSH 回传；本机只在
127.0.0.1 启动临时预览页，并在收到首帧后自动打开默认浏览器。

针对当前 Rockchip 相机的直接用法::

    python scripts/ssh_camera_preview.py

通用用法::

    python scripts/ssh_camera_preview.py user@host --device /dev/video0

退出方式：点击预览页中的“停止预览”，或回到终端按 Ctrl+C。
密码由系统 ssh 交互读取，不会由脚本接收或保存。
"""

from __future__ import annotations

import argparse
import html
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Sequence


def _iter_jpegs(pipe: BinaryIO, chunk_size: int = 16384) -> Iterator[bytes]:
    """从 ffmpeg image2pipe 字节流中增量拆分 JPEG 帧。"""
    soi = b"\xff\xd8"
    eoi = b"\xff\xd9"
    buffer = bytearray()
    while True:
        chunk = pipe.read(chunk_size)
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            start = buffer.find(soi)
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if start > 0:
                del buffer[:start]
            end = buffer.find(eoi, 2)
            if end < 0:
                break
            end += 2
            yield bytes(buffer[:end])
            del buffer[:end]


class FrameHub:
    """只保存最新帧；浏览器处理慢时丢旧帧，避免延迟不断累积。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Optional[bytes] = None
        self._sequence = 0
        self._stopped = False
        self._error: Optional[str] = None
        self.first_frame = threading.Event()

    def publish(self, frame: bytes) -> None:
        with self._condition:
            self._frame = frame
            self._sequence += 1
            self.first_frame.set()
            self._condition.notify_all()

    def finish(self, error: Optional[str] = None) -> None:
        with self._condition:
            self._stopped = True
            self._error = error
            self._condition.notify_all()

    def wait_after(self, sequence: int, timeout: float = 2.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence or self._stopped,
                timeout=timeout,
            )
            return self._sequence, self._frame, self._stopped, self._error

    @property
    def stopped(self) -> bool:
        with self._condition:
            return self._stopped

    @property
    def error(self) -> Optional[str]:
        with self._condition:
            return self._error


class RemoteCamera:
    def __init__(self, args: argparse.Namespace, hub: FrameHub) -> None:
        self.args = args
        self.hub = hub
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def _command(self) -> list[str]:
        ffmpeg = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "v4l2",
        ]
        if self.args.input_format:
            ffmpeg.extend(["-input_format", self.args.input_format])
        if self.args.width > 0 and self.args.height > 0:
            ffmpeg.extend(
                ["-video_size", f"{self.args.width}x{self.args.height}"]
            )
        # 当前 Rockchip ISP 不支持 VIDIOC_G/S_PARM，默认不传 -framerate。
        if self.args.fps > 0:
            ffmpeg.extend(["-framerate", str(self.args.fps)])
        ffmpeg.extend(
            [
                "-i",
                self.args.device,
                "-an",
                "-c:v",
                "mjpeg",
                "-q:v",
                str(self.args.jpeg_quality),
                "-flush_packets",
                "1",
                "-f",
                "image2pipe",
                "pipe:1",
            ]
        )
        remote = (
            "command -v ffmpeg >/dev/null 2>&1 || "
            "{ echo '[ERROR] 远端未安装 ffmpeg' >&2; exit 127; }; "
            "exec "
            + shlex.join(ffmpeg)
        )

        command = [self.args.ssh_command, "-T"]
        if self.args.ssh_port is not None:
            command.extend(["-p", str(self.args.ssh_port)])
        if self.args.identity is not None:
            command.extend(["-i", os.fspath(self.args.identity)])
        command.extend(
            [
                "-o",
                f"ConnectTimeout={self.args.connect_timeout}",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=3",
            ]
        )
        command.extend([self.args.host, remote])
        return command

    def start(self) -> None:
        self.process = subprocess.Popen(
            self._command(), stdout=subprocess.PIPE, bufsize=0
        )
        self.thread = threading.Thread(
            target=self._read_loop, name="ssh-camera-reader", daemon=True
        )
        self.thread.start()

    def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        frames = 0
        error: Optional[str] = None
        try:
            for jpeg in _iter_jpegs(self.process.stdout):
                if self.stop_event.is_set():
                    break
                frames += 1
                self.hub.publish(jpeg)
            code = self.process.wait()
            if code != 0 and not self.stop_event.is_set():
                error = f"SSH/ffmpeg 退出，状态码 {code}"
            elif frames == 0 and not self.stop_event.is_set():
                error = "视频流结束，但没有收到 JPEG 帧"
        except (OSError, ValueError) as exc:
            error = str(exc)
        finally:
            self.hub.finish(error)

    def stop(self) -> None:
        self.stop_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.thread is not None:
            self.thread.join(timeout=2)


def _preview_html(host: str, device: str, rotation: int) -> bytes:
    safe_title = html.escape(f"{host} · {device}")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SSH 相机实时预览</title>
<style>
html,body{{margin:0;height:100%;background:#111;color:#eee;font-family:system-ui,sans-serif;overflow:hidden}}
header{{height:48px;box-sizing:border-box;display:flex;align-items:center;gap:14px;padding:8px 14px;background:#202020}}
#title{{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
button,select{{padding:6px 10px;background:#383838;color:#fff;border:1px solid #666;border-radius:5px}}
#viewport{{height:calc(100% - 48px);display:flex;align-items:center;justify-content:center;overflow:hidden}}
#camera{{max-width:100%;max-height:100%;flex-shrink:0;object-fit:contain;transform:rotate({rotation}deg)}}
#status{{color:#7ee787}}
</style></head><body>
<header><span id="status">● 实时</span><span id="title">{safe_title}</span>
<label>旋转 <select id="rotate"><option>0</option><option>90</option><option>180</option><option>270</option></select></label>
<button id="stop">停止预览</button></header>
<div id="viewport"><img id="camera" src="/stream.mjpg" alt="等待相机画面"></div>
<script>
const image=document.getElementById('camera');
const rotate=document.getElementById('rotate'); rotate.value='{rotation}';
function fit(){{
  const sideways=Number(rotate.value)%180!==0;
  image.style.transform=`rotate(${{rotate.value}}deg)`;
  image.style.maxWidth=sideways?'calc(100vh - 48px)':'100vw';
  image.style.maxHeight=sideways?'100vw':'calc(100vh - 48px)';
}}
rotate.onchange=fit; fit();
image.onerror=()=>{{document.getElementById('status').textContent='● 流已断开';document.getElementById('status').style.color='#ff7b72';}};
document.getElementById('stop').onclick=async()=>{{await fetch('/shutdown',{{method:'POST'}});window.close();document.body.innerHTML='<p style="padding:20px">预览已停止，可以关闭此页面。</p>';}};
</script></body></html>"""
    return page.encode("utf-8")


def _make_handler(
    hub: FrameHub,
    stop_event: threading.Event,
    page: bytes,
):
    class PreviewHandler(BaseHTTPRequestHandler):
        server_version = "SshCameraPreview/1.0"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path in ("/", "/index.html"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store, no-cache")
                self.end_headers()
                sequence = -1
                try:
                    while not stop_event.is_set():
                        new_sequence, frame, stopped, _ = hub.wait_after(sequence)
                        if new_sequence == sequence:
                            if stopped:
                                break
                            continue
                        sequence = new_sequence
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        if stopped:
                            break
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            if self.path == "/healthz":
                body = b"ok\n" if hub.error is None else hub.error.encode("utf-8")
                self.send_response(
                    HTTPStatus.OK if hub.error is None else HTTPStatus.BAD_GATEWAY
                )
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/shutdown":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            stop_event.set()
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

    return PreviewHandler


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实时预览 SSH 服务器的 V4L2 相机（自动打开本机浏览器）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "host",
        nargs="?",
        default="cat@10.100.127.98",
        help="SSH 目标或 ~/.ssh/config 主机别名",
    )
    parser.add_argument("--device", default="/dev/video11", help="远端 V4L2 通道")
    parser.add_argument("--width", type=int, default=1280, help="请求画面宽度")
    parser.add_argument("--height", type=int, default=720, help="请求画面高度")
    parser.add_argument("--fps", type=float, default=0.0, help="0=设备原生帧率")
    parser.add_argument("--input-format", default="uyvy422", help="V4L2 像素格式")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        choices=range(2, 32),
        default=5,
        metavar="2..31",
        help="传输 JPEG 质量，越小画质越高",
    )
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=270, help="浏览器显示旋转角度")
    parser.add_argument("--http-port", type=int, default=0, help="本地预览端口；0=自动选择")
    parser.add_argument("--ssh-port", type=int, help="SSH 端口")
    parser.add_argument("--identity", type=Path, help="SSH 私钥")
    parser.add_argument("--ssh-command", default="ssh", help="OpenSSH 命令")
    parser.add_argument("--connect-timeout", type=int, default=10, help="SSH 连接超时秒数")
    parser.add_argument("--startup-timeout", type=int, default=120, help="等待首帧超时秒数")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器，仅打印 URL")
    args = parser.parse_args(argv)
    if args.host.startswith("-"):
        parser.error("host 不能以 '-' 开头")
    if (args.width > 0) != (args.height > 0):
        parser.error("--width 与 --height 必须同时为正，或同时 <=0")
    if args.startup_timeout <= 0 or args.connect_timeout <= 0:
        parser.error("超时秒数必须 >0")
    if not 0 <= args.http_port <= 65535:
        parser.error("--http-port 必须在 0..65535 之间")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if shutil.which(args.ssh_command) is None:
        print(f"[ERROR] 找不到 SSH 客户端: {args.ssh_command}", file=sys.stderr)
        return 127

    hub = FrameHub()
    stop_event = threading.Event()
    page = _preview_html(args.host, args.device, args.rotate)
    handler = _make_handler(hub, stop_event, page)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.http_port), handler)
    except OSError as exc:
        print(f"[ERROR] 本地预览端口绑定失败: {exc}", file=sys.stderr)
        return 1
    server.daemon_threads = True
    server_thread = threading.Thread(
        target=server.serve_forever, name="preview-http", daemon=True
    )
    server_thread.start()

    camera = RemoteCamera(args, hub)
    print(f"[INFO] 正在连接 {args.host}，请在 SSH 提示中输入密码……")
    try:
        camera.start()
        deadline = time.monotonic() + args.startup_timeout
        while not hub.first_frame.wait(timeout=0.1):
            if hub.stopped or time.monotonic() >= deadline:
                detail = hub.error or "等待首帧超时"
                print(f"[ERROR] {detail}", file=sys.stderr)
                return 1
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/"
        print(f"[OK] 已收到首帧，实时预览地址: {url}")
        if not args.no_browser:
            opened = webbrowser.open(url, new=1, autoraise=True)
            if not opened:
                print("[WARN] 无法自动打开浏览器，请手动访问上面的地址")
        print("[INFO] 点击页面中的“停止预览”，或在终端按 Ctrl+C 退出")
        while not stop_event.wait(0.25):
            if hub.stopped:
                print(f"[ERROR] {hub.error or '远端视频流已结束'}", file=sys.stderr)
                return 1
        return 0
    except OSError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] 用户终止预览")
        return 0
    finally:
        stop_event.set()
        camera.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
