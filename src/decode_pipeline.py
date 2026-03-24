"""
ColorRay **收端（解码）流水线** — 与 ``frame_pipeline`` 中的发端逻辑对称、分层说明如下。

## 发端已完成部分（encoder，仍在 ``frame_pipeline``）

- 文件切分、线格式（CRC16、tribit 尾字节对齐）、互质填充、帧头写入、``drawer`` 出图；
- 多帧目录、RAID 条带、PNG→MP4（OpenCV）。

## 收端流水线阶段（本模块对外语义）

1. **图像 → 137×137 矩阵**  
   ``image_to_matrix``：``test`` / ``test_robust`` / ``test_robust_inner`` / ``test_robust_mean`` / ``normal``（见该函数文档）。MP4 可用 ``--run-mode auto`` 逐帧自动尝试前四种直至 CRC 通过。

2. **帧头**  
   ``ReadHeaderFromGrid`` → ``PayloadLen``、Mode、组号等（与发端 ``WriteHeaderToGrid`` 对应）。

3. **矩阵 → tribit 序列**  
   ``matrix_to_colors``（与发端 ``colors_to_matrix`` 同模板 ``generate_frame``）。

4. **tribit → 定长线缓冲**  
   ``frame_pipeline._tribits_to_bytes_exact``（经本模块转调，见下方 API）。

5. **按帧头截断 + CRC16**  
   ``decode_png_to_payload``；RAID/MP4 路径上先用 ``decode_png_to_wire_payload`` 再条带重组、后去 CRC。

6. **多帧拼接 / RAID / 视频**  
   ``decode_frames_to_file``、``decode_mp4_to_file``；RAID 需 ``original_size.txt`` 与同目录帧命名。

## 实现策略

解码核心实现暂 **复用** ``frame_pipeline`` 内函数（单一真源、避免双份 tribit/CRC 逻辑漂移）。
本文件提供 **收端专用聚合与 CLI**（``python run_decode.py``），便于与发端文件分工；后续若拆分公共子模块，可再把 tribit/CRC 抽到 ``common`` 而不改对外 API。

CLI 子命令与 ``python run.py decode|decode-file|from-mp4|extract-mp4`` 等价。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# 与 frame_pipeline 一致，保证从仓库根或 src 运行均可
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.header import FrameHeader
from frame_pipeline import (
    decode_frames_to_file,
    decode_mp4_to_file,
    decode_png_to_payload,
    decode_png_to_wire_payload,
    extract_mp4_to_png_dir,
)

__all__ = [
    "decode_png_to_payload",
    "decode_png_to_wire_payload",
    "decode_frames_to_file",
    "decode_mp4_to_file",
    "extract_mp4_to_png_dir",
    "main",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ColorRay 收端：PNG / 帧目录 / MP4 → 还原文件（与 run.py 中 decode* 子命令等价）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_png = sub.add_parser("png", help="单帧 PNG → 用户负载（校验 CRC16）")
    p_png.add_argument("--input", "-i", required=True)
    p_png.add_argument("--output", "-o", required=True)
    p_png.add_argument(
        "--run-mode",
        default="test",
        choices=(
            "test",
            "test_robust",
            "test_robust_inner",
            "test_robust_mean",
            "normal",
        ),
    )

    p_dir = sub.add_parser("frames", help="多帧 PNG 目录 → 单个文件（含 RAID 目录规则）")
    p_dir.add_argument("--input", "-i", required=True)
    p_dir.add_argument("--output", "-o", required=True)
    p_dir.add_argument(
        "--run-mode",
        default="test",
        choices=(
            "test",
            "test_robust",
            "test_robust_inner",
            "test_robust_mean",
            "normal",
        ),
    )

    p_mp4 = sub.add_parser("mp4", help="MP4 按帧解码 → 单个文件（RAID 需同目录 original_size.txt）")
    p_mp4.add_argument("--input", "-i", required=True)
    p_mp4.add_argument("--output", "-o", required=True)
    p_mp4.add_argument(
        "--run-mode",
        default="test",
        choices=(
            "auto",
            "test",
            "test_robust",
            "test_robust_inner",
            "test_robust_mean",
            "normal",
        ),
        help="auto：平铺视频下逐帧尝试多种采样直至 CRC 通过",
    )

    p_ex = sub.add_parser("extract-mp4", help="MP4 → PNG 帧目录（调试）")
    p_ex.add_argument("--input", "-i", required=True)
    p_ex.add_argument("--output", "-o", required=True)

    args = parser.parse_args()

    if args.cmd == "png":
        payload, hdr = decode_png_to_payload(args.input, run_mode=args.run_mode)
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Path(args.output).write_bytes(payload)
        print(
            f"decode_pipeline png ok bytes={len(payload)} -> {os.path.abspath(args.output)} "
            f"PayloadLen={hdr.PayloadLen} Mode={hdr.Mode}"
        )
        return

    if args.cmd == "frames":
        nb = decode_frames_to_file(args.input, args.output, run_mode=args.run_mode)
        print(f"decode_pipeline frames ok bytes={nb} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "mp4":
        nb = decode_mp4_to_file(args.input, args.output, run_mode=args.run_mode)
        print(f"decode_pipeline mp4 ok bytes={nb} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "extract-mp4":
        n = extract_mp4_to_png_dir(args.input, args.output)
        print(f"decode_pipeline extract-mp4 ok frames={n} -> {os.path.abspath(args.output)}")
        return


# 类型别名供其它模块引用
DecodedPayload = Tuple[bytes, FrameHeader]

if __name__ == "__main__":
    main()
