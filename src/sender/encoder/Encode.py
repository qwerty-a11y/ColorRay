from common.CorrectionLevel import RaidLevel, RSLevel
from common.File import FileToBinary


def Encode(path: str, raid: RaidLevel, rs: RSLevel, out_frames_dir: str | None = None) -> int | None:
    """
    若提供 out_frames_dir：raid 为 NONE 时平铺多帧；否则按 RAID 分组编码（与 frame_pipeline 一致）。
    """
    if out_frames_dir is not None:
        if raid == RaidLevel.NONE:
            from frame_pipeline import encode_file_to_frames

            return encode_file_to_frames(path, out_frames_dir, mode=0)
        from frame_pipeline import encode_file_to_frames_raid

        return encode_file_to_frames_raid(path, out_frames_dir, raid)
    FileToBinary(path)
    return None
