"""
ColorRay **发端（编码）主入口**：帧头写入/读出网格，串联 generate_frame、colors_to_matrix、drawer、
image_to_matrix、matrix_to_colors 与 CRC16 线格式负载。

收端流水线说明与专用 CLI 见 ``decode_pipeline.py`` / 仓库根 ``run_decode.py``（子命令：
png / frames / mp4 / extract-mp4）。
--------------------------
1. 对要传输的文件编码，得到多帧图，再合成 MP4（发端本机）。
2. 在屏幕上播放该 MP4，用录屏软件录制（或数字拷贝同一 MP4 文件，见下）。

  # 发端：本地或「上传后放到某路径」的任意文件 → MP4
  python run.py file-to-video -i D:\\path\\你的文件.zip -o colorray_out.mp4 --fps 2
  # 同上（子命令别名）
  python run.py encode-file-to-mp4 -i 要传.bin -o colorray_out.mp4 --fps 2

  # 收端：屏录或拷贝得到的视频 → 还原文件
  python run.py from-mp4 -i 录屏.mp4 -o 还原.bin

若编码时使用了 ``--raid``，发端会在 MP4 同目录写出 ``original_size.txt``；收端解码 RAID
视频时，请把该文件与录屏视频放在**同一目录**后再执行 ``from-mp4``。

PayloadLen（16bit）：0xFFFF 表示满帧 DataSize；否则为线格式有效字节数（含 2 字节 CRC16）。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

# 保证可导入 common、sender、receiver（本文件位于 src/）
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.Config import DataBlocks, DataSize
from common.CorrectionLevel import RaidLevel
from common.crc16 import crc16
from common.header import FrameHeader, ReadHeaderFromGrid, WriteHeaderToGrid
from common.Raid import Raid5Decode, Raid5Encode, Raid6Decode, Raid6Encode
from receiver.decoder.image_to_matrix import image_to_matrix
from receiver.decoder.matrix_to_colors import matrix_to_colors
from sender.generator.colors_to_matrix import colors_to_matrix
from sender.generator.drawer import drawer
from sender.generator.frame_gen import generate_frame

# 16bit 全 1：与 header.PayloadLen 一致，表示「线格式有效长度 = Config.DataSize」
FULL_FRAME_PAYLOAD_MARKER = 0xFFFF

# FrameHeader.Mode：0 无 RAID；1=RAID5(9+1)；2=RAID5(4+1)；3=RAID6(3+2)。与 CorrectionLevel.RaidLevel 对应。
_RAID_FRAME_RE = re.compile(r"^frame_(\d+)_(\d+)\.png$", re.I)


def _import_cv2():
    """MP4 读写依赖 OpenCV；PNG 编解码路径不导入本模块。"""
    try:
        import cv2
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "视频/MP4 功能需要 OpenCV，请安装: pip install opencv-python\n"
            "（仅 PNG 编解码可不装 opencv，例如 encode、decode、roundtrip。）"
        ) from e
    return cv2


def _write_mp4_frame_bgr_to_png_for_decode(bgr_frame, path: str) -> None:
    """VideoCapture 为 BGR；解码链用 PIL 读 RGB。与 drawer 出图一致，避免 cv2.imwrite 通道约定差异。"""
    cv2 = _import_cv2()
    from PIL import Image

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


# 传给 image_to_matrix；不含 auto（auto 仅在 decode_mp4_to_file 内逐帧尝试下列子策略）
_CRC_RUN_MODES = ("test", "test_robust", "test_robust_inner", "test_robust_mean", "normal")
# MP4 有损编码时逐帧按序尝试，直至 CRC16 通过（仅 Mode=0）
_MP4_AUTO_STRATEGIES: tuple[str, ...] = (
    "test",
    "test_robust",
    "test_robust_inner",
    "test_robust_mean",
)

CRC16_LEN = 2
# tribit 共 DataBlocks×3 位；若 (DataBlocks*3)%8!=0，末字节仅部分位进图，须留 1 字节作尾填充，
# 使 user+CRC 两字节完全落在「整字节进图」的前缀里。
_TRIBIT_REM_BITS = (DataBlocks * 3) % 8
_TRIBIT_TAIL_PAD_BYTES = 1 if _TRIBIT_REM_BITS else 0
MAX_USER_BYTES_PER_FRAME = DataSize - CRC16_LEN - _TRIBIT_TAIL_PAD_BYTES


def _mask_unused_high_bits_last_byte(buf: bytes) -> bytes:
    """保证与 _bytes_to_tribits_exact / _tribits_to_bytes_exact 一致，避免最后一字节高半字节「幽灵位」。"""
    if len(buf) != DataSize:
        raise ValueError(f"缓冲长度须为 DataSize={DataSize}")
    if _TRIBIT_REM_BITS == 0:
        return buf
    b = bytearray(buf)
    b[-1] &= (1 << _TRIBIT_REM_BITS) - 1
    return bytes(b)


def _effective_from_hdr(hdr: FrameHeader) -> int:
    if hdr.PayloadLen == FULL_FRAME_PAYLOAD_MARKER:
        return DataSize
    return min(hdr.PayloadLen, DataSize)


def _wire_pad_user_with_crc(user: bytes) -> tuple[bytes, int]:
    if len(user) > MAX_USER_BYTES_PER_FRAME:
        raise ValueError(
            f"单帧用户数据 {len(user)} 字节，超过上限 {MAX_USER_BYTES_PER_FRAME} "
            f"（CRC {CRC16_LEN} 字节"
            + (f" + tribit 尾填充 {_TRIBIT_TAIL_PAD_BYTES} 字节" if _TRIBIT_TAIL_PAD_BYTES else "")
            + "）"
        )
    wire = user + crc16(user).to_bytes(CRC16_LEN, "big")
    if len(wire) > DataSize - _TRIBIT_TAIL_PAD_BYTES:
        raise ValueError("user+CRC 过长，与 tribit/字节对齐约束冲突")
    eff = len(wire)
    padded = _mask_unused_high_bits_last_byte(wire.ljust(DataSize, b"\x00")[:DataSize])
    return padded, eff


def _strip_and_verify_crc(wire: bytes) -> bytes:
    if len(wire) < CRC16_LEN:
        raise ValueError("帧数据过短，无法包含 CRC16")
    got = int.from_bytes(wire[-CRC16_LEN:], "big")
    body = wire[:-CRC16_LEN]
    if crc16(body) != got:
        raise ValueError("CRC16-CCITT 校验失败")
    return body


def _wire_ljust_datasize(wire: bytes) -> bytes:
    return wire.ljust(DataSize, b"\x00")[:DataSize]


def _user_from_recovered_raid_block(blk: bytes, hdr: FrameHeader | None) -> bytes:
    """RAID 恢复出的数据盘块（DataSize）→ 去 CRC 的用户数据；缺帧恢复时 hdr 可为 None（按满帧）。"""
    eff = _effective_from_hdr(hdr) if hdr is not None else DataSize
    return _strip_and_verify_crc(blk[:eff])


def _encode_padded_frame_to_png(
    padded: bytes,
    effective: int,
    out_path: str,
    *,
    mode: int,
    group_id: int,
    in_group_id: int,
    is_last: bool,
) -> str:
    if len(padded) != DataSize:
        raise ValueError(f"padded 长度须为 DataSize={DataSize}")
    padded = _mask_unused_high_bits_last_byte(padded)
    color_list = _bytes_to_tribits_exact(padded, DataBlocks)
    grid, need_border = generate_frame()
    filled = colors_to_matrix(grid, color_list)
    hdr = FrameHeader(
        mode,
        group_id & 0xFF,
        in_group_id & 0x0F,
        _payload_len_for_header(effective),
        is_last,
    )
    WriteHeaderToGrid(filled, hdr)
    out_path = os.path.abspath(out_path)
    directory, filename = os.path.dirname(out_path), os.path.basename(out_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return drawer(filled, need_border, filename, directory or ".")


def _bits_low_first(data: bytes) -> list[int]:
    bits: list[int] = []
    for b in data:
        for bit_pos in range(8):
            bits.append((b >> bit_pos) & 1)
    return bits


def _bytes_to_tribits_exact(buf: bytes, tribit_count: int) -> list[tuple[int, int, int]]:
    """与 bytes_to_colors 相同位序，但严格取 tribit_count*3 位（不补字节到 3 倍数）。"""
    need_bits = tribit_count * 3
    need_bytes = (need_bits + 7) // 8
    raw = buf.ljust(need_bytes, b"\x00")[:need_bytes]
    bits = _bits_low_first(raw)[:need_bits]
    colors: list[tuple[int, int, int]] = []
    for i in range(0, need_bits, 3):
        g = bits[i : i + 3]
        while len(g) < 3:
            g.append(0)
        colors.append((g[0] * 255, g[1] * 255, g[2] * 255))
    return colors


def _tribits_to_bytes_exact(colors: list[tuple[int, int, int]]) -> bytes:
    """与 colors_to_bytes 相同规则，输出 ceil(len(colors)*3/8) 字节。"""
    bits: list[int] = []
    for (r, g, b) in colors:
        bits.append(1 if r >= 128 else 0)
        bits.append(1 if g >= 128 else 0)
        bits.append(1 if b >= 128 else 0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        v = sum(bit * (2**j) for j, bit in enumerate(chunk))
        out.append(v)
    return bytes(out)


def _payload_len_for_header(effective_bytes: int) -> int:
    if effective_bytes == DataSize:
        return FULL_FRAME_PAYLOAD_MARKER
    if effective_bytes > 0xFFFE:
        raise ValueError(
            f"有效负载长度 {effective_bytes} 超过 0xFFFE，无法写入 16bit 帧头（0xFFFF 保留为满帧标记）"
        )
    return effective_bytes


def encode_frame_to_png(
    payload: bytes,
    out_path: str,
    *,
    mode: int = 0,
    group_id: int = 0,
    in_group_id: int = 0,
    is_last: bool = True,
) -> str:
    """
    将用户数据编码为单帧 PNG：追加 CRC16-CCITT（大端），再零填充至 DataSize 后写入网格。
    单帧用户数据至多 MAX_USER_BYTES_PER_FRAME（见 tribit 位对齐与 CRC 尾字节约束）。
    """
    padded, eff = _wire_pad_user_with_crc(payload)
    return _encode_padded_frame_to_png(
        padded,
        eff,
        out_path,
        mode=mode,
        group_id=group_id,
        in_group_id=in_group_id,
        is_last=is_last,
    )


def encode_file_to_frames(
    src_path: str,
    out_dir: str,
    *,
    mode: int = 0,
) -> int:
    """
    将文件按每帧至多 MAX_USER_BYTES_PER_FRAME 字节切分，每帧自动加 CRC16 后编码为 PNG。
    输出文件名 frame_000000.png … 按字典序即播放/解码顺序。
    帧头：GroupId = 块序号 // 16，InGroupId = 块序号 % 16（各 8/4 位，超 4096 块时 GroupId 会回绕，解码以文件名为准）。
    """
    src_path = os.path.abspath(src_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    data = Path(src_path).read_bytes()
    n = len(data)
    if n == 0:
        raise ValueError("空文件")
    nframes = (n + MAX_USER_BYTES_PER_FRAME - 1) // MAX_USER_BYTES_PER_FRAME
    for i in range(nframes):
        chunk = data[
            i * MAX_USER_BYTES_PER_FRAME : (i + 1) * MAX_USER_BYTES_PER_FRAME
        ]
        gid = (i // 16) & 0xFF
        ing = i % 16
        is_last = i == nframes - 1
        out_png = os.path.join(out_dir, f"frame_{i:06d}.png")
        encode_frame_to_png(
            chunk,
            out_png,
            mode=mode,
            group_id=gid,
            in_group_id=ing,
            is_last=is_last,
        )
    return nframes


def _list_frame_pngs(directory: str) -> list[str]:
    directory = os.path.abspath(directory)
    names = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".png")
    )
    return [os.path.join(directory, f) for f in names]


def _raid_level_to_header_mode(level: RaidLevel) -> int:
    m = {
        RaidLevel.LEVEL1_10: 1,
        RaidLevel.LEVEL2_20: 2,
        RaidLevel.LEVEL3_40: 3,
    }
    if level not in m:
        raise ValueError("raid_level 须为 LEVEL1_10 / LEVEL2_20 / LEVEL3_40")
    return m[level]


def _header_mode_to_k_and_kind(hdr_mode: int) -> tuple[str, int]:
    if hdr_mode == 1:
        return "raid5", 9
    if hdr_mode == 2:
        return "raid5", 4
    if hdr_mode == 3:
        return "raid6", 3
    raise ValueError(f"非 RAID 帧头 Mode={hdr_mode}")


def encode_file_to_frames_raid(
    src_path: str,
    out_dir: str,
    raid_level: RaidLevel,
) -> int:
    """
    按 RAID 条带分组：每组 k 个「线格式」DataSize 块（用户段补满至 MAX_USER 字节再加 CRC）
    经 Raid5Encode/Raid6Encode 后输出 k+1 或 k+2 帧。校验盘为不透明整块，按满帧写入。
    文件名 frame_{组号:05d}_{盘号:02d}.png；写入 original_size.txt 供还原截断。
    """
    hdr_mode = _raid_level_to_header_mode(raid_level)
    kind, k_data = _header_mode_to_k_and_kind(hdr_mode)
    src_path = os.path.abspath(src_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    data = Path(src_path).read_bytes()
    if not data:
        raise ValueError("空文件")
    orig_len = len(data)
    Path(out_dir, "original_size.txt").write_text(str(orig_len), encoding="ascii")

    unit_user = k_data * MAX_USER_BYTES_PER_FRAME
    total_padded = ((len(data) + unit_user - 1) // unit_user) * unit_user
    data = data.ljust(total_padded, b"\x00")
    user_chunks = [
        data[i : i + MAX_USER_BYTES_PER_FRAME]
        for i in range(0, len(data), MAX_USER_BYTES_PER_FRAME)
    ]
    n_groups = len(user_chunks) // k_data
    total_png = 0
    for g in range(n_groups):
        group_users = user_chunks[g * k_data : (g + 1) * k_data]
        wires_effs = [_wire_pad_user_with_crc(u) for u in group_users]
        data_disks = [[we[0]] for we in wires_effs]
        if kind == "raid5":
            enc = Raid5Encode(data_disks)
        else:
            enc = Raid6Encode(data_disks)
        nd = len(enc)
        for d in range(nd):
            block = enc[d][0]
            if len(block) != DataSize:
                raise RuntimeError(f"RAID 输出块长 {len(block)} != DataSize")
            fname = f"frame_{g:05d}_{d:02d}.png"
            fp = os.path.join(out_dir, fname)
            is_last = g == n_groups - 1 and d == nd - 1
            if d < k_data:
                padded, eff = wires_effs[d]
                _encode_padded_frame_to_png(
                    padded,
                    eff,
                    fp,
                    mode=hdr_mode,
                    group_id=g & 0xFF,
                    in_group_id=d & 0x0F,
                    is_last=is_last,
                )
            else:
                _encode_padded_frame_to_png(
                    block,
                    DataSize,
                    fp,
                    mode=hdr_mode,
                    group_id=g & 0xFF,
                    in_group_id=d & 0x0F,
                    is_last=is_last,
                )
            total_png += 1
    return total_png


def _decode_file_from_raid_frames(
    in_dir: str,
    out_path: str,
    *,
    run_mode: str,
) -> int:
    in_dir = os.path.abspath(in_dir)
    manifest = Path(in_dir) / "original_size.txt"
    orig_len = int(manifest.read_text(encoding="ascii").strip())
    paths = [
        p
        for p in _list_frame_pngs(in_dir)
        if _RAID_FRAME_RE.match(os.path.basename(p))
    ]
    if not paths:
        raise ValueError("未找到 frame_GGGGG_DD.png 形式的 RAID 帧")

    def _sort_key(p: str) -> tuple[int, int]:
        m = _RAID_FRAME_RE.match(os.path.basename(p))
        assert m
        return int(m.group(1)), int(m.group(2))

    paths.sort(key=_sort_key)
    hdr_mode: int | None = None
    groups: dict[int, dict[int, tuple[bytes, FrameHeader]]] = {}
    for p in paths:
        wire, hdr = decode_png_to_wire_payload(p, run_mode=run_mode)
        if hdr.Mode not in (1, 2, 3):
            raise ValueError(f"RAID 目录中帧 Mode={hdr.Mode} 非法")
        if hdr_mode is None:
            hdr_mode = hdr.Mode
        elif hdr.Mode != hdr_mode:
            raise ValueError("组内帧头 Mode 不一致")
        m = _RAID_FRAME_RE.match(os.path.basename(p))
        assert m
        g, d = int(m.group(1)), int(m.group(2))
        groups.setdefault(g, {})[d] = (wire, hdr)

    assert hdr_mode is not None
    kind, k_data = _header_mode_to_k_and_kind(hdr_mode)
    n_disks = k_data + (1 if kind == "raid5" else 2)
    out_chunks: list[bytes] = []
    for g in sorted(groups.keys()):
        sm = groups[g]
        slot: list[tuple[bytes, FrameHeader] | None] = [sm.get(d) for d in range(n_disks)]
        disks: list[list[bytes | None]] = []
        for d in range(n_disks):
            t = slot[d]
            if t is None:
                disks.append([None])
            else:
                w, _h = t
                disks.append([_wire_ljust_datasize(w)])
        if kind == "raid5":
            rec = Raid5Decode(disks)
        else:
            rec = Raid6Decode(disks)
        for di in range(k_data):
            blk = rec[di][0]
            if blk is None:
                raise ValueError(f"RAID 组 {g} 数据盘 {di} 无法恢复（丢帧过多或损坏）")
            tdi = slot[di]
            hdr_di = tdi[1] if tdi is not None else None
            out_chunks.append(_user_from_recovered_raid_block(blk, hdr_di))
    raw = b"".join(out_chunks)[:orig_len]
    out_path = os.path.abspath(out_path)
    od = os.path.dirname(out_path)
    if od:
        os.makedirs(od, exist_ok=True)
    Path(out_path).write_bytes(raw)
    return len(raw)


def decode_frames_to_file(
    in_dir: str,
    out_path: str,
    *,
    run_mode: str = "test",
) -> int:
    """
    读取目录内所有 .png（按文件名排序），按帧头有效长度拼接，写入单个文件。
    若存在 original_size.txt 且含 frame_*_*.png，则按 RAID 组解码。
    返回写入的总字节数。
    """
    in_abs = os.path.abspath(in_dir)
    manifest = Path(in_abs) / "original_size.txt"
    raid_named = [
        p
        for p in _list_frame_pngs(in_abs)
        if _RAID_FRAME_RE.match(os.path.basename(p))
    ]
    if manifest.is_file():
        if not raid_named:
            raise ValueError("存在 original_size.txt 但未找到 RAID 帧（frame_组号_盘号.png）")
        return _decode_file_from_raid_frames(in_abs, out_path, run_mode=run_mode)

    paths = _list_frame_pngs(in_abs)
    if not paths:
        raise ValueError(f"目录中无 PNG: {in_dir}")
    parts: list[bytes] = []
    for p in paths:
        payload, hdr = decode_png_to_payload(p, run_mode=run_mode)
        parts.append(payload)
    out_data = b"".join(parts)
    out_path = os.path.abspath(out_path)
    od = os.path.dirname(out_path)
    if od:
        os.makedirs(od, exist_ok=True)
    Path(out_path).write_bytes(out_data)
    return len(out_data)


def write_mp4_from_png_dir(
    frames_dir: str,
    out_mp4: str,
    *,
    fps: float = 2.0,
    fourcc: str = "mp4v",
) -> int:
    """
    按文件名字典序读取目录内全部 .png，写入 MP4（BGR，与 cv2.imread 一致）。
    """
    cv2 = _import_cv2()
    paths = _list_frame_pngs(os.path.abspath(frames_dir))
    if not paths:
        raise ValueError(f"目录中无 PNG: {frames_dir}")
    out_mp4 = os.path.abspath(out_mp4)
    od = os.path.dirname(out_mp4)
    if od:
        os.makedirs(od, exist_ok=True)
    first = cv2.imread(paths[0])
    if first is None:
        raise ValueError(f"无法读取: {paths[0]}")
    h, w = first.shape[:2]
    if len(fourcc) != 4:
        raise ValueError("fourcc 须为 4 字符，如 mp4v")
    cc = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(out_mp4, cc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 无法打开: {out_mp4}（可换 fourcc 或检查 OpenCV 编解码）")
    try:
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                raise ValueError(f"无法读取: {p}")
            if img.shape[0] != h or img.shape[1] != w:
                raise ValueError(f"帧尺寸不一致: {p}")
            writer.write(img)
    finally:
        writer.release()
    return len(paths)


def extract_mp4_to_png_dir(mp4_path: str, out_dir: str) -> int:
    """将 MP4 逐帧导出为 frame_000000.png（调试用）。"""
    cv2 = _import_cv2()
    mp4_path = os.path.abspath(mp4_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {mp4_path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fp = os.path.join(out_dir, f"frame_{n:06d}.png")
            _write_mp4_frame_bgr_to_png_for_decode(frame, fp)
            n += 1
    finally:
        cap.release()
    return n


def decode_mp4_to_file(
    mp4_path: str,
    out_path: str,
    *,
    run_mode: str = "test",
) -> int:
    """
    按播放顺序逐帧解码。Mode=0：直接拼接各帧负载；Mode=1..3：RAID，须在与 mp4 同目录存在 original_size.txt。
    run_mode=auto：仅 Mode=0；逐帧在 test / test_robust / test_robust_inner / test_robust_mean 中尝试直至 CRC 通过。
    """
    cv2 = _import_cv2()
    mp4_path = os.path.abspath(mp4_path)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {mp4_path}")
    fps: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="colorray_f_") as td:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fp = os.path.join(td, f"f_{idx:06d}.png")
                _write_mp4_frame_bgr_to_png_for_decode(frame, fp)
                fps.append(fp)
                idx += 1
            if not fps:
                raise ValueError("视频中未读到帧")

            # 须在临时目录仍存在时读完所有 PNG（此前在 with 外解码会导致文件已删）
            if run_mode == "auto":
                return _decode_mp4_mode0_auto_strategies(fps, out_path)

            rows: list[tuple[bytes, FrameHeader]] = []
            for fp in fps:
                wire, hdr = decode_png_to_wire_payload(fp, run_mode=run_mode)
                rows.append((wire, hdr))

            hdr0 = rows[0][1]
            if hdr0.Mode == 0:
                parts: list[bytes] = []
                for fi, (w, _) in enumerate(rows):
                    try:
                        parts.append(_strip_and_verify_crc(w))
                    except ValueError as e:
                        raise ValueError(f"第 {fi} 帧（0 起）CRC16 校验失败") from e
                raw = b"".join(parts)
                out_path_abs = os.path.abspath(out_path)
                od = os.path.dirname(out_path_abs)
                if od:
                    os.makedirs(od, exist_ok=True)
                Path(out_path_abs).write_bytes(raw)
                return len(raw)

            mp4_dir = os.path.dirname(mp4_path)
            manifest = Path(mp4_dir) / "original_size.txt"
            if not manifest.is_file():
                raise ValueError(
                    "RAID 视频解码需要与 mp4 同目录下的 original_size.txt（encode-file-to-mp4 已复制）"
                )
            orig_len = int(manifest.read_text(encoding="ascii").strip())
            hdr_mode = hdr0.Mode
            for _w, hdr in rows:
                if hdr.Mode != hdr_mode:
                    raise ValueError("各帧 Mode 不一致")
            kind, k_data = _header_mode_to_k_and_kind(hdr_mode)
            n_disks = k_data + (1 if kind == "raid5" else 2)
            if len(rows) % n_disks != 0:
                raise ValueError(
                    f"总帧数 {len(rows)} 不能整除每 RAID 组盘数 {n_disks}"
                )
            out_chunks: list[bytes] = []
            for g in range(0, len(rows), n_disks):
                slot = rows[g : g + n_disks]
                disks: list[list[bytes | None]] = [
                    [_wire_ljust_datasize(slot[d][0])] for d in range(n_disks)
                ]
                if kind == "raid5":
                    rec = Raid5Decode(disks)
                else:
                    rec = Raid6Decode(disks)
                for di in range(k_data):
                    blk = rec[di][0]
                    if blk is None:
                        raise ValueError(f"RAID 组 {g // n_disks} 数据盘 {di} 无法恢复")
                    _w, hdr_di = slot[di]
                    out_chunks.append(_user_from_recovered_raid_block(blk, hdr_di))
            raw = b"".join(out_chunks)[:orig_len]
            out_path_abs = os.path.abspath(out_path)
            od = os.path.dirname(out_path_abs)
            if od:
                os.makedirs(od, exist_ok=True)
            Path(out_path_abs).write_bytes(raw)
            return len(raw)
    finally:
        cap.release()


def encode_file_to_mp4(
    src_path: str,
    out_mp4: str,
    *,
    fps: float = 2.0,
    raid_level: RaidLevel | None = None,
) -> int:
    """
    先编码为临时 PNG 序列再写 MP4。RAID 时把 original_size.txt 复制到 mp4 所在目录供解码使用。
    """
    out_mp4 = os.path.abspath(out_mp4)
    dest_dir = os.path.dirname(out_mp4) or "."
    os.makedirs(dest_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colorray_mp4_") as td:
        if raid_level is None:
            encode_file_to_frames(src_path, td, mode=0)
        else:
            encode_file_to_frames_raid(src_path, td, raid_level)
            shutil.copy2(
                Path(td) / "original_size.txt",
                Path(dest_dir) / "original_size.txt",
            )
        n = write_mp4_from_png_dir(td, out_mp4, fps=fps)
    return n


def decode_png_to_wire_payload(
    image_path: str,
    *,
    run_mode: str = "test",
) -> Tuple[bytes, FrameHeader]:
    """
    解码「线格式」：用户 + CRC +（逻辑上）填充，不校验 CRC。
    RAID 重组须用本函数取各帧字节，再在条带层去 CRC。
    """
    mat = image_to_matrix(image_path, run_mode=run_mode)
    if not mat:
        raise ValueError("image_to_matrix 返回空矩阵")

    hdr = ReadHeaderFromGrid(mat)
    if hdr is None:
        raise ValueError("帧头解析失败（同步字节或 CRC8 不匹配）")

    template, _ = generate_frame()
    colors = matrix_to_colors(template, mat)
    raw = _tribits_to_bytes_exact(colors)
    if len(raw) != DataSize:
        raise ValueError(f"解码得到 {len(raw)} 字节，期望 DataSize={DataSize}")
    eff = _effective_from_hdr(hdr)
    return raw[:eff], hdr


def decode_png_to_payload(
    image_path: str,
    *,
    run_mode: str = "test",
) -> Tuple[bytes, FrameHeader]:
    """
    解码用户负载并校验 CRC16；返回已去掉末尾 2 字节 CRC 的数据。

    :param run_mode: 传给 image_to_matrix；由本仓库 drawer 生成的 PNG 一般用 test（137 格满幅）。
    """
    wire, hdr = decode_png_to_wire_payload(image_path, run_mode=run_mode)
    return _strip_and_verify_crc(wire), hdr


def decode_png_to_payload_first_matching(
    image_path: str, strategies: tuple[str, ...]
) -> Tuple[bytes, FrameHeader, str]:
    """按序尝试 strategies，返回首个 CRC16 通过的策略名与用户负载。"""
    last: BaseException | None = None
    for m in strategies:
        try:
            payload, hdr = decode_png_to_payload(image_path, run_mode=m)
            return payload, hdr, m
        except ValueError as e:
            last = e
    raise ValueError(
        f"下列采样策略均无法通过 CRC16: {', '.join(strategies)}"
    ) from last


def _decode_mp4_mode0_auto_strategies(fps: list[str], out_path: str) -> int:
    parts: list[bytes] = []
    hdr0: FrameHeader | None = None
    for fi, fp in enumerate(fps):
        try:
            payload, hdr, _m = decode_png_to_payload_first_matching(
                fp, _MP4_AUTO_STRATEGIES
            )
        except ValueError as e:
            raise ValueError(f"第 {fi} 帧（0 起）auto 仍失败") from e
        if hdr0 is None:
            hdr0 = hdr
        elif hdr.Mode != hdr0.Mode:
            raise ValueError(f"第 {fi} 帧 Mode={hdr.Mode} 与首帧 Mode={hdr0.Mode} 不一致")
        parts.append(payload)
    if hdr0 is None:
        raise ValueError("内部错误：无帧")
    if hdr0.Mode != 0:
        raise ValueError(
            "run_mode=auto 仅支持 Mode=0（平铺）视频；RAID 请指定固定 --run-mode"
        )
    raw = b"".join(parts)
    out_path = os.path.abspath(out_path)
    od = os.path.dirname(out_path)
    if od:
        os.makedirs(od, exist_ok=True)
    Path(out_path).write_bytes(raw)
    return len(raw)


def print_workflow_guide() -> None:
    """在终端打印端到端流程说明（与模块文档一致，便于复制命令）。"""
    print(
        """ColorRay 端到端流程
==================

一、发端（生成可播放的码流视频）
  整文件直接变 MP4（内部会生成临时 PNG 序列再封装）：
    python run.py file-to-video -i <原文件路径> -o colorray.mp4 --fps 2
    python run.py encode-file-to-mp4 -i <原文件路径> -o colorray.mp4 --fps 2

  若需 RAID（可选）：
    python run.py file-to-video -i <原文件> -o colorray.mp4 --fps 2 --raid level2_20
  同目录会多一个 original_size.txt，收端须与视频放在一起再解码。

  分步：先 PNG 目录再合成视频：
    python run.py encode-file -i <原文件> -o frames/
    python run.py to-mp4 -i frames/ -o colorray.mp4 --fps 2

二、屏录或传文件
  • 屏录：全屏或尽量让 1644×1644 码区居中、占满；录屏帧率建议与上面 --fps 一致，
    播放器关闭「运动模糊/插帧」，避免一帧码图在录像里重复多帧（否则会解码错位）。
  • 不传屏：直接把 colorray.mp4 拷到收端解码（最稳）。

三、收端（从视频还原文件）
    python run.py from-mp4 -i <录屏或拷贝的.mp4> -o <还原文件>
  默认 --run-mode test；单帧失败可试 test_robust_inner（4:2:0 色度）或 ``--run-mode auto``（逐帧多策略）。

四、自检
    python run.py roundtrip-mp4 --fps 2
  验证「文件→MP4→再读 MP4→文件」在本机是否一致。

其它子命令：python run.py -h
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ColorRay：文件↔图像↔视频编解码（含屏录收端 workflow）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "workflow",
        help="打印「编码→视频→屏录/传文件→解码」步骤与示例命令",
    )

    p_enc = sub.add_parser("encode", help="二进制 -> PNG")
    p_enc.add_argument("--input", "-i", required=True, help="输入文件路径")
    p_enc.add_argument("--output", "-o", required=True, help="输出 PNG 路径")
    p_enc.add_argument("--mode", type=int, default=0)
    p_enc.add_argument("--group-id", type=int, default=0)
    p_enc.add_argument("--in-group-id", type=int, default=0)
    p_enc.add_argument("--not-last", action="store_true", help="非文件最后一帧")

    p_dec = sub.add_parser("decode", help="PNG -> 二进制")
    p_dec.add_argument("--input", "-i", required=True, help="输入 PNG 路径")
    p_dec.add_argument("--output", "-o", required=True, help="输出文件路径")
    p_dec.add_argument(
        "--run-mode",
        default="test",
        choices=_CRC_RUN_MODES,
        help="image_to_matrix 模式；本仓库生成的图通常用 test",
    )

    p_rt = sub.add_parser("roundtrip", help="内存往返自测（不写盘）")
    p_rt.add_argument("--run-mode", default="test", choices=_CRC_RUN_MODES)

    p_ef = sub.add_parser("encode-file", help="整文件 -> 多帧 PNG 目录")
    p_ef.add_argument("--input", "-i", required=True, help="输入文件")
    p_ef.add_argument("--output", "-o", required=True, help="输出目录（将创建）")
    p_ef.add_argument("--mode", type=int, default=0)

    p_df = sub.add_parser("decode-file", help="多帧 PNG 目录 -> 单个文件")
    p_df.add_argument("--input", "-i", required=True, help="含 frame_*.png 的目录")
    p_df.add_argument("--output", "-o", required=True, help="输出文件路径")
    p_df.add_argument(
        "--run-mode",
        default="test",
        choices=_CRC_RUN_MODES,
    )

    p_rtf = sub.add_parser("roundtrip-file", help="多帧整文件往返自测（临时目录）")
    p_rtf.add_argument(
        "--run-mode",
        default="test",
        choices=_CRC_RUN_MODES,
    )

    p_efr = sub.add_parser("encode-file-raid", help="整文件 -> RAID 分组多帧 PNG")
    p_efr.add_argument("--input", "-i", required=True)
    p_efr.add_argument("--output", "-o", required=True)
    p_efr.add_argument(
        "--raid",
        required=True,
        choices=("level1_10", "level2_20", "level3_40"),
        help="LEVEL1_10=RAID5 9+1；LEVEL2_20=RAID5 4+1；LEVEL3_40=RAID6 3+2（须 pyjerasure）",
    )

    p_rtfr = sub.add_parser("roundtrip-file-raid", help="RAID 多帧往返自测（默认 level2_20）")
    p_rtfr.add_argument(
        "--raid",
        default="level2_20",
        choices=("level1_10", "level2_20", "level3_40"),
    )
    p_rtfr.add_argument("--run-mode", default="test", choices=_CRC_RUN_MODES)

    p_tm = sub.add_parser("to-mp4", help="PNG 目录 -> MP4（按文件名排序）")
    p_tm.add_argument("--input-dir", "-i", required=True)
    p_tm.add_argument("--output", "-o", required=True)
    p_tm.add_argument("--fps", type=float, default=2.0)
    p_tm.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV 四字符编码，默认 mp4v",
    )

    p_em = sub.add_parser("extract-mp4", help="MP4 -> PNG 帧目录（调试）")
    p_em.add_argument("--input", "-i", required=True)
    p_em.add_argument("--output", "-o", required=True)

    p_fm = sub.add_parser(
        "from-mp4",
        help="MP4 → 还原文件（屏录/上传视频；RAID 时视频与 original_size.txt 同目录）",
    )
    p_fm.add_argument("--input", "-i", required=True)
    p_fm.add_argument("--output", "-o", required=True)
    p_fm.add_argument(
        "--run-mode",
        default="test",
        choices=("auto",) + _CRC_RUN_MODES,
        help="auto：逐帧在 test / test_robust / inner / mean 间尝试直至 CRC 通过（仅平铺 Mode=0）",
    )

    p_efm = sub.add_parser(
        "encode-file-to-mp4",
        help="整文件 -> 临时多帧 -> MP4；可选 --raid（并复制 original_size.txt 到 mp4 目录）",
    )
    p_efm.add_argument("--input", "-i", required=True)
    p_efm.add_argument("--output", "-o", required=True)
    p_efm.add_argument("--fps", type=float, default=2.0)
    p_efm.add_argument(
        "--raid",
        choices=("level1_10", "level2_20", "level3_40"),
        default=None,
        help="不传则平铺多帧（无 RAID）",
    )

    p_ftv = sub.add_parser(
        "file-to-video",
        help="任意文件 → MP4（与 encode-file-to-mp4 相同，便于上传文件后一键出视频）",
    )
    p_ftv.add_argument("--input", "-i", required=True, help="输入文件路径（任意类型、任意大小）")
    p_ftv.add_argument("--output", "-o", required=True, help="输出 .mp4 路径")
    p_ftv.add_argument("--fps", type=float, default=2.0)
    p_ftv.add_argument(
        "--raid",
        choices=("level1_10", "level2_20", "level3_40"),
        default=None,
        help="不传则平铺多帧（无 RAID）",
    )

    p_rtm = sub.add_parser("roundtrip-mp4", help="经 MP4 的整文件往返（默认平铺）")
    p_rtm.add_argument("--fps", type=float, default=2.0)
    p_rtm.add_argument(
        "--raid",
        choices=("level1_10", "level2_20", "level3_40"),
        default=None,
    )
    p_rtm.add_argument(
        "--run-mode",
        default="test",
        choices=("auto",) + _CRC_RUN_MODES,
    )

    args = parser.parse_args()

    if args.cmd == "workflow":
        print_workflow_guide()
        return

    if args.cmd == "encode":
        with open(args.input, "rb") as f:
            data = f.read()
        path = encode_frame_to_png(
            data,
            args.output,
            mode=args.mode,
            group_id=args.group_id,
            in_group_id=args.in_group_id,
            is_last=not args.not_last,
        )
        print(path)
        return

    if args.cmd == "decode":
        payload, hdr = decode_png_to_payload(args.input, run_mode=args.run_mode)
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(payload)
        print(
            f"写入 {args.output} 字节数={len(payload)} "
            f"Mode={hdr.Mode} GroupId={hdr.GroupId} InGroupId={hdr.InGroupId} "
            f"PayloadLenField={hdr.PayloadLen} IsLast={hdr.IsLast}"
        )
        return

    if args.cmd == "roundtrip":
        original = os.urandom(MAX_USER_BYTES_PER_FRAME)
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "t.png")
            encode_frame_to_png(original, png)
            back, hdr = decode_png_to_payload(png, run_mode=args.run_mode)
        assert back == original, "往返不一致"
        print(
            f"roundtrip ok MAX_USER={MAX_USER_BYTES_PER_FRAME} "
            f"hdr.PayloadLen={hdr.PayloadLen}"
        )
        return

    if args.cmd == "encode-file":
        nf = encode_file_to_frames(args.input, args.output, mode=args.mode)
        print(f"encode-file ok frames={nf} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "decode-file":
        nb = decode_frames_to_file(args.input, args.output, run_mode=args.run_mode)
        print(f"decode-file ok bytes={nb} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "roundtrip-file":
        original = bytes((i * 17 + 42) % 256 for i in range(DataSize * 3 + 123))
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.bin")
            Path(src).write_bytes(original)
            d = os.path.join(td, "frames")
            encode_file_to_frames(src, d, mode=0)
            out = os.path.join(td, "out.bin")
            decode_frames_to_file(d, out, run_mode=args.run_mode)
            back = Path(out).read_bytes()
        assert back == original, "roundtrip-file 不一致"
        print(f"roundtrip-file ok bytes={len(back)}")
        return

    if args.cmd == "encode-file-raid":
        rl = {
            "level1_10": RaidLevel.LEVEL1_10,
            "level2_20": RaidLevel.LEVEL2_20,
            "level3_40": RaidLevel.LEVEL3_40,
        }[args.raid]
        n = encode_file_to_frames_raid(args.input, args.output, rl)
        print(f"encode-file-raid ok frames={n} raid={args.raid} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "roundtrip-file-raid":
        rl = {
            "level1_10": RaidLevel.LEVEL1_10,
            "level2_20": RaidLevel.LEVEL2_20,
            "level3_40": RaidLevel.LEVEL3_40,
        }[args.raid]
        original = bytes((i * 19 + 11) % 256 for i in range(DataSize * 7 + 400))
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.bin")
            Path(src).write_bytes(original)
            d = os.path.join(td, "frames")
            encode_file_to_frames_raid(src, d, rl)
            out = os.path.join(td, "out.bin")
            decode_frames_to_file(d, out, run_mode=args.run_mode)
            back = Path(out).read_bytes()
        assert back == original, "roundtrip-file-raid 不一致"
        print(f"roundtrip-file-raid ok raid={args.raid} bytes={len(back)}")
        return

    if args.cmd == "to-mp4":
        n = write_mp4_from_png_dir(
            args.input_dir,
            args.output,
            fps=args.fps,
            fourcc=args.fourcc,
        )
        print(f"to-mp4 ok frames={n} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "extract-mp4":
        n = extract_mp4_to_png_dir(args.input, args.output)
        print(f"extract-mp4 ok frames={n} -> {os.path.abspath(args.output)}")
        return

    if args.cmd == "from-mp4":
        nb = decode_mp4_to_file(args.input, args.output, run_mode=args.run_mode)
        print(f"from-mp4 ok bytes={nb} -> {os.path.abspath(args.output)}")
        return

    if args.cmd in ("encode-file-to-mp4", "file-to-video"):
        rl_map = {
            "level1_10": RaidLevel.LEVEL1_10,
            "level2_20": RaidLevel.LEVEL2_20,
            "level3_40": RaidLevel.LEVEL3_40,
        }
        rl = rl_map[args.raid] if args.raid else None
        n = encode_file_to_mp4(args.input, args.output, fps=args.fps, raid_level=rl)
        label = "file-to-video" if args.cmd == "file-to-video" else "encode-file-to-mp4"
        print(
            f"{label} ok frames={n} raid={args.raid or 'none'} -> {os.path.abspath(args.output)}"
        )
        return

    if args.cmd == "roundtrip-mp4":
        rl_map = {
            "level1_10": RaidLevel.LEVEL1_10,
            "level2_20": RaidLevel.LEVEL2_20,
            "level3_40": RaidLevel.LEVEL3_40,
        }
        rl = rl_map[args.raid] if args.raid else None
        original = bytes((i * 23 + 5) % 256 for i in range(DataSize * 5 + 333))
        with tempfile.TemporaryDirectory(prefix="colorray_rtm_") as td:
            src = os.path.join(td, "src.bin")
            Path(src).write_bytes(original)
            mp4 = os.path.join(td, "t.mp4")
            encode_file_to_mp4(src, mp4, fps=args.fps, raid_level=rl)
            out = os.path.join(td, "out.bin")
            decode_mp4_to_file(mp4, out, run_mode=args.run_mode)
            back = Path(out).read_bytes()
        assert back == original, "roundtrip-mp4 不一致"
        print(f"roundtrip-mp4 ok raid={args.raid or 'none'} bytes={len(back)}")
        return


if __name__ == "__main__":
    main()
