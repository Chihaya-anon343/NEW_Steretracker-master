"""OBB ONNX 模型 INT8 动态量化 (离线, 一次性).

用法 (Windows 主机, 需 pip install onnxruntime):
    python scripts/quantize_obb_model.py                          # 默认 obb_best.onnx -> obb_best_int8.onnx
    python scripts/quantize_obb_model.py --input A.onnx --output B.onnx
    python scripts/quantize_obb_model.py --ops Conv               # 只量化 Conv (不量化 MatMul)

说明:
    - 动态量化: 权重 INT8 (per-channel), 激活按运行时量化; CPU 上卷积有 VNNI 加速时
      通常 1.5~2.5x 提速; 对 GPU 无收益 (CUDA EP 不走 QLinearConv)。
    - OBB 角度/回归输出对量化敏感, 量化后务必在 val/approach 集上回归对比
      (检出数 / conf / 模板分 / PnP 重投影), 见 obb_pipeline 双跑对比流程。
    - 若动态量化提速不足 (<1.3x), 下一步是带校准数据的静态量化 (QuantizedDataReader)。
"""

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    ap = argparse.ArgumentParser(description="OBB ONNX INT8 动态量化")
    ap.add_argument("--input", default="yolo_onnx/obb_best.onnx")
    ap.add_argument("--output", default="yolo_onnx/obb_best_int8.onnx")
    ap.add_argument("--ops", default="Conv,MatMul",
                    help="量化算子类型, 逗号分隔 (默认 Conv,MatMul)")
    ap.add_argument("--per-channel", type=int, default=1, help="权重 per-channel 量化 (1=开)")
    ap.add_argument("--weight-type", default="quint8", choices=["qint8", "quint8"],
                    help="权重类型: quint8=u8u8 (无 VNNI 的 CPU 可跑, 通用); "
                         "qint8=u8s8 (需 AVX512-VNNI, 否则 ConvInteger 无实现直接报错)")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise SystemExit(f"输入模型不存在: {src}")

    ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    wtype = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    quantize_dynamic(
        str(src), str(dst),
        weight_type=wtype,
        op_types_to_quantize=ops,
        per_channel=bool(args.per_channel),
        reduce_range=False,
    )
    mb = 1e6
    print(f"量化完成: {src.name} ({src.stat().st_size / mb:.1f} MB) "
          f"-> {dst.name} ({dst.stat().st_size / mb:.1f} MB), "
          f"算子: {ops}, weight={args.weight_type}, per_channel={bool(args.per_channel)}")


if __name__ == "__main__":
    main()
