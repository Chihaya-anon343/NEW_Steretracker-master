#!/usr/bin/env python3
"""AprilTag sidecar worker (pose_compare2).

与 C++ 主程序 (AprilTagExtractor) 的管道协议:
  1. 启动握手: stdin 首行 = 配置 JSON, 回复 "READY"
     {"class0": {"family": "tag36h11", "tag_ids": [0], "tag_size_mm": 200.0},
      "class1": {...}, "nthreads": 4}
  2. 每帧请求: 一行 JSON 头 + img_len 字节原始灰度图 (uint8, 行优先)
     {"class": 0, "w": W, "h": H, "fx": .., "fy": .., "cx": .., "cy": .., "img_len": N}
     (cx/cy 必须是 ROI 局部主点 = 全图主点 - ROI 偏移)
  3. 每帧回复: 一行 JSON
     {"ok": true, "n_tags": K, "tag": {"id": .., "hamming": .., "family": "..",
       "corners": [[x,y]x4], "center": [x,y], "decision_margin": ..,
       "R": [[..]x3], "t_mm": [x,y,z], "pose_err": ..}}
     多标签时按白名单过滤后取角点四边形面积最大者; 无标签时省略 "tag" 键
    (回包任何位置不可出现 JSON null —— C++ cv::FileStorage 不支持)。

约定: stdout 只走协议 (二进制缓冲, 逐行 JSON), 日志一律走 stderr。
"""

import json
import sys

import numpy as np
from pupil_apriltags import Detector


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_exact(n: int) -> bytes:
    """从 stdin 精确读取 n 字节 (管道 read 可能分片返回)。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin EOF")
        buf.extend(chunk)
    return bytes(buf)


def poly_area(corners) -> float:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    s = 0.0
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def main() -> int:
    # --- 握手: 读配置, 建各 class 的 Detector ---
    line = sys.stdin.buffer.readline()
    if not line:
        log("[apriltag_worker] 无启动配置, 退出")
        return 1
    cfg = json.loads(line.decode("utf-8"))
    nthreads = int(cfg.get("nthreads", 4))

    detectors = {}
    for cls in ("class0", "class1"):
        c = cfg.get(cls)
        if not c or not c.get("family"):
            continue
        tag_ids = c.get("tag_ids") or []
        detectors[cls] = {
            "det": Detector(families=c["family"], nthreads=nthreads),
            "tag_ids": set(int(t) for t in tag_ids),
            "tag_size_m": float(c["tag_size_mm"]) / 1000.0,
        }
        log(f"[apriltag_worker] {cls}: family={c['family']} "
            f"tag_ids={sorted(set(tag_ids)) or 'ANY'} "
            f"tag_size={c['tag_size_mm']}mm")

    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()

    # --- 帧循环 ---
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        try:
            req = json.loads(line.decode("utf-8"))
            cls = f"class{int(req['class'])}"
            if cls not in detectors:
                raise ValueError(f"{cls} 未配置标签模板")

            n = int(req["img_len"])
            data = read_exact(n)
            img = np.frombuffer(data, dtype=np.uint8).reshape(int(req["h"]), int(req["w"]))

            d = detectors[cls]
            results = d["det"].detect(
                img,
                estimate_tag_pose=True,
                camera_params=(float(req["fx"]), float(req["fy"]),
                               float(req["cx"]), float(req["cy"])),
                tag_size=d["tag_size_m"],
            )
            if d["tag_ids"]:
                results = [r for r in results if r.tag_id in d["tag_ids"]]

            # 择优: 四边形面积最大 (近距离最稳), 平手取 hamming 低者
            best = None
            best_key = None
            for r in results:
                key = (poly_area(r.corners), -r.hamming)
                if best is None or key > best_key:
                    best, best_key = r, key

            # 注意: 回包不可含 JSON null —— C++ 端 cv::FileStorage 不支持 null, 遇到即抛解析异常。
            # 故无标签时省略 "tag" 键; 位姿缺失时省略对应键 (C++ 侧按空节点降级该帧)。
            tag = None
            if best is not None:
                corners = [[float(p[0]), float(p[1])] for p in best.corners]
                tag = {
                    "id": int(best.tag_id),
                    "hamming": int(best.hamming),
                    "family": str(best.tag_family),
                    "corners": corners,
                    "center": [float(best.center[0]), float(best.center[1])],
                    "decision_margin": float(best.decision_margin),
                }
                # pose_t 单位与 tag_size 一致 (米) → mm
                if best.pose_R is not None:
                    tag["R"] = best.pose_R.tolist()
                if best.pose_t is not None:
                    tag["t_mm"] = [float(v) * 1000.0 for v in np.asarray(best.pose_t).flatten()]
                if best.pose_err is not None:
                    tag["pose_err"] = float(best.pose_err)
            reply = {"ok": True, "n_tags": len(results)}
            if tag is not None:
                reply["tag"] = tag
        except Exception as e:  # 协议/检测异常不杀 worker, 回错误让主程序降级该帧
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        sys.stdout.buffer.write(json.dumps(reply).encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
