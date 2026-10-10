#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obb_best.onnx -> obb_best.rknn 转换脚本 (RK3588, INT8 量化)。

在板子上运行 (项目根目录):
    cd ~/tmp/NEW_Steretracker-master_COPY && python3 scripts/convert_obb_rknn.py

配方 (与 yolov8n_p2.rknn 喂入方式对齐):
  - mean_values=[[0,0,0]], std_values=[[255,255,255]]  归一化 baked 进模型
    -> NPU 输入为量化 int8, C++ 端直喂原始 uint8 RGB (RknnYoloDetector input_is_uint8_ 路径)
  - target_platform='rk3588'
  - do_quantization=True, 校准集 = tests/data/approach_seq/*.png (自动生成绝对路径列表)

可覆盖: python3 scripts/convert_obb_rknn.py <onnx> <rknn> <calib_dir>
"""

import sys
from pathlib import Path

from rknn.api import RKNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ONNX_PATH = PROJECT_ROOT / 'yolo_onnx' / 'obb_best.onnx'
RKNN_PATH = PROJECT_ROOT / 'yolo_onnx' / 'obb_best.rknn'
CALIB_DIR = PROJECT_ROOT / 'tests' / 'data' / 'approach_seq'
CALIB_TXT = PROJECT_ROOT / 'scripts' / 'obb_calib_list.txt'
TARGET_PLATFORM = 'rk3588'


def build_calib_list(calib_dir: Path, out_txt: Path):
    imgs = sorted(p for p in calib_dir.glob('*.png')
                  if not p.name.endswith(('_pose.png', '_vis.png')))
    imgs += sorted(calib_dir.glob('*.jpg'))
    if len(imgs) < 20:
        raise SystemExit(f'[ERROR] 校准图太少 ({len(imgs)} < 20): {calib_dir}')
    out_txt.write_text('\n'.join(str(p) for p in imgs) + '\n', encoding='utf-8')
    print(f'[INFO] 校准集: {len(imgs)} 张 -> {out_txt}')
    return imgs


def main():
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ONNX_PATH
    rknn_path = Path(sys.argv[2]) if len(sys.argv) > 2 else RKNN_PATH
    calib_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else CALIB_DIR

    if not onnx_path.exists():
        raise SystemExit(f'[ERROR] ONNX 不存在: {onnx_path}')

    build_calib_list(calib_dir, CALIB_TXT)

    rknn = RKNN(verbose=True)

    # disable_rules: toolkit2 2.3.0 图优化器在 OBB 角度分支 (Cos/Sin→Concat) 上
    # 的 reduce_reshape_op_around_concat 融合规则有 bug (KeyError), 官方提示禁用绕过
    print(f'[INFO] config: mean=0 std=255 target={TARGET_PLATFORM} '
          f'disable_rules=[reduce_reshape_op_around_concat]')
    ret = rknn.config(mean_values=[[0, 0, 0]],
                      std_values=[[255, 255, 255]],
                      target_platform=TARGET_PLATFORM,
                      disable_rules=['reduce_reshape_op_around_concat'])
    if ret != 0:
        raise SystemExit(f'[ERROR] config 失败: {ret}')

    print(f'[INFO] load_onnx: {onnx_path}')
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit(f'[ERROR] load_onnx 失败: {ret}')

    print('[INFO] build: INT8 量化 (校准中, 板上 CPU 较慢请耐心等待)...')
    ret = rknn.build(do_quantization=True, dataset=str(CALIB_TXT))
    if ret != 0:
        raise SystemExit(f'[ERROR] build 失败: {ret}')

    print(f'[INFO] export_rknn: {rknn_path}')
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise SystemExit(f'[ERROR] export_rknn 失败: {ret}')

    rknn.release()
    size_mb = rknn_path.stat().st_size / 1024 / 1024
    print(f'[DONE] {rknn_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
