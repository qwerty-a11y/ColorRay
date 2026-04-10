import math
import os
from typing import List, Tuple

from common import Config
from common.CorrectionLevel import RaidLevel,RSLevel
from common.File import FileToBinary
from common.Config import FrameDataSize, GroupDataSize, RSCorrectionBytes
from common.RSmodule import rs_encode_bytes, rs_decode_bytes
from common.Raid import Raid5Encode, Raid6Encode, Raid5Decode, Raid6Decode
from common.CRC16 import verify_crc16

def Encode(path:str, raid:RaidLevel, rs:RSLevel):
    binary = FileToBinary(path)
    print(f"原始文件大小: {len(binary)} 字节")
    
    # 【新增】构建固定长度的文件头信息 (文件名 + 文件大小)
    # 格式定义：
    # 1. 文件名长度 (2 字节，uint16, 大端序)
    # 2. 文件名数据 (254 字节，不足补 0，超长截断)
    # 3. 文件大小 (8 字节，uint64, 大端序)
    # 总长度 = 2 + 254 + 8 = 264 字节
    FILE_NAME_MAX_LEN = 254
    HEADER_TOTAL_LEN = 2 + FILE_NAME_MAX_LEN + 8
    
    file_name = os.path.basename(path).encode('utf-8')
    if len(file_name) > FILE_NAME_MAX_LEN:
        file_name = file_name[:FILE_NAME_MAX_LEN]
    
    # 填充文件名部分
    padded_file_name = file_name.ljust(FILE_NAME_MAX_LEN, b'\x00')
    
    # 打包文件名长度
    name_len_bytes = len(file_name).to_bytes(2, byteorder='big')
    
    # 打包文件大小 (原始二进制长度)
    file_size_bytes = len(binary).to_bytes(8, byteorder='big')
    
    # 组合文件头
    file_header = name_len_bytes + padded_file_name + file_size_bytes
    
    # 将文件头拼接到原始数据前
    binary = file_header + binary
    
    RSCorrectBytesPerGroup = 0
    FileGroupSize = 0
    match rs:
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = RSCorrectionBytes.LEVEL1_5.r
            FileGroupSize = GroupDataSize - RSCorrectBytesPerGroup * RSCorrectionBytes.LEVEL1_5.b
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = RSCorrectionBytes.LEVEL2_10.r
            FileGroupSize = GroupDataSize - RSCorrectBytesPerGroup * RSCorrectionBytes.LEVEL2_10.b
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = RSCorrectionBytes.LEVEL3_15.r
            FileGroupSize = GroupDataSize - RSCorrectBytesPerGroup * RSCorrectionBytes.LEVEL3_15.b
        case RSLevel.NONE:
            FileGroupSize = GroupDataSize
    

    # 根据 RaidLevel 决定第一维的元素个数 (行数)
    row_count = 0
    match raid:
        case RaidLevel.LEVEL1_10:
            row_count = 9
        case RaidLevel.LEVEL2_20:
            row_count = 4
        case RaidLevel.LEVEL3_40:
            row_count = 3
        case RaidLevel.NONE:
            row_count = 1
            
    # 计算总共需要多少个切片 (向上取整)
    total_chunks = math.ceil(len(binary) / FileGroupSize)
    
    # 计算需要的列数
    base_col_count = math.ceil(total_chunks / row_count) if row_count > 0 else 0
    
    # 【修改】确保第二维（列）元素个数是 8 的倍数，不足则向上取整
    col_count = ((base_col_count + 7) // 8) * 8
    print(f"打包数据大小：{col_count * row_count * FileGroupSize} 字节 (行数: {row_count}, 列数: {col_count}, 每块大小: {FileGroupSize} 字节)")
    
    # 初始化二维数组 [row_count][col_count]
    data_groups:List[List[bytes|None]] = [[None for _ in range(col_count)] for _ in range(row_count)]
    
    current_index = 0
    # 按列优先顺序填充：先填满第 0 列的行，再填第 1 列...
    # 对应需求：依次填充 [0][0], [1][0], [2][0]... 
    for col in range(col_count):
        for row in range(row_count):
            if current_index >= total_chunks:
                break
            
            start = current_index * int(FileGroupSize)
            end = start + int(FileGroupSize)
            chunk = binary[start:end]
            
            # 【修改】在此处立即填充至 FileGroupSize (针对最后一个可能不满的块)
            # 注意：此时还未添加 RS 校验码，所以目标是填充到 FileGroupSize
            if len(chunk) < int(FileGroupSize):
                padding_len = int(FileGroupSize) - len(chunk)
                chunk = chunk + os.urandom(padding_len)
            
            # 填充到对应位置
            data_groups[row][col] = chunk
            
            current_index += 1
            
        if current_index >= total_chunks:
            break

    # 【修改新增】在所有真实数据填充完成后，对剩余的 None 位置填充随机数据
    # 此时所有数据块长度均为 FileGroupSize
    for row in range(row_count):
        for col in range(col_count):
            if data_groups[row][col] is None:
                # 生成与 FileGroupSize 长度一致的随机数据
                data_groups[row][col] = os.urandom(int(FileGroupSize))

    # 【修改】先进行 Raid 编码
    # 此时数据块长度为 FileGroupSize，Raid 编码会生成新的校验块（长度也为 FileGroupSize）
    # 编码后，data_groups 的行数可能会增加（例如 RAID5 增加 1 行，RAID6 增加 2 行）
    match raid:
        case RaidLevel.LEVEL1_10 | RaidLevel.LEVEL2_20:
            # RAID 5 编码
            data_groups = Raid5Encode(data_groups) # type: ignore
            
        case RaidLevel.LEVEL3_40:
            # RAID 6 编码
            data_groups = Raid6Encode(data_groups) # type: ignore
            
        case RaidLevel.NONE:
            # 无需 RAID 编码，保持原状
            pass

    # 【修改】后进行 RS 编码
    # 此时对所有块（包括原始数据块和 Raid 生成的校验块）统一添加 RS 纠错码
    # 编码后长度将变为 GroupDataSize (FileGroupSize + RS 校验码)
    RSCorrectBytesPerGroup = 0
    match rs:
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = RSCorrectionBytes.LEVEL1_5.r
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = RSCorrectionBytes.LEVEL2_10.r
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = RSCorrectionBytes.LEVEL3_15.r
        case RSLevel.NONE:
            RSCorrectBytesPerGroup = 0

    if RSCorrectBytesPerGroup > 0:
        for row in range(len(data_groups)):
            for col in range(len(data_groups[0])):
                if data_groups[row][col] is not None:
                    data_groups[row][col] = rs_encode_bytes(data_groups[row][col], RSCorrectBytesPerGroup) # type: ignore
    

    return data_groups

def GroupToFrames(group: List[List[bytes]]) -> List[bytes]:
    """
    将二维数据组按行优先顺序展平成帧列表
    逻辑：先处理完第一行的所有数据块（每8个块组成一帧），再处理第二行，以此类推。
    
    :param group: 二维数据组 [行数][列数]，矩形矩阵
    :return: 帧列表，每个帧包含同一行的连续8个数据块的拼接
    """
    frames = []
    if not group or not group[0]:
        return frames
    
    rows = len(group)
    cols = len(group[0])
    
    block_index_global = 0
    
    # 【修改】外层循环遍历行，确保先把一行分完再分下一行
    for row in range(rows):
        row_blocks = []
        # 内层循环遍历当前行的所有列
        for col in range(cols):
            chunk = group[row][col]
            if chunk is not None:
                row_blocks.append(chunk)
                block_index_global += 1
        
        # 当前行的所有块收集完毕后，每8个一组组成帧
        for i in range(0, len(row_blocks), 8):
            frame = b''.join(row_blocks[i:i+8])
            frames.append(frame)
    return frames


def FramesToGroup(frames: List[bytes], raid: RaidLevel) -> List[List[bytes]]:
    """
    将帧列表恢复为二维数据组（GroupToFrames 的反向操作）
    逻辑：按行优先顺序，先从帧列表中取出属于第一行的所有帧，恢复第一行的数据块；
         再取出属于第二行的所有帧，恢复第二行的数据块，以此类推。
    """
    # 确定行数
    match raid:
        case RaidLevel.LEVEL1_10:
            row_count = 10
        case RaidLevel.LEVEL2_20:
            row_count = 5
        case RaidLevel.LEVEL3_40:
            row_count = 5
        case RaidLevel.NONE:
            row_count = 1
        case _:
            raise ValueError(f"未知的 RAID 等级: {raid}")
    
    if row_count == 0:
        return []
    if not frames:
        return [[]]
    
    total_frames = len(frames)
    if total_frames % row_count != 0:
        raise ValueError(f"帧数 {total_frames} 不能被行数 {row_count} 整除")
    
    # 计算每行有多少个帧
    frames_per_row = total_frames // row_count
    # 每行的列数 = 每行帧数 * 8
    cols_per_row = frames_per_row * 8
    
    # 初始化二维数组
    group: List[List[bytes]] = [[None for _ in range(cols_per_row)] for _ in range(row_count)]
    
    frame_idx = 0
    
    # 【修改】外层循环遍历行，确保先恢复完一行再恢复下一行
    for row in range(row_count):
        # 处理当前行的所有帧
        for f in range(frames_per_row):
            frame = frames[frame_idx]
            frame_idx += 1
            
            # 将帧拆分为8个块
            chunk_size = len(frame) // 8
            for i in range(8):
                start = i * chunk_size
                end = start + chunk_size if i < 7 else len(frame)
                chunk = frame[start:end]
                
                # 计算在当前行中的列索引
                col = f * 8 + i
                group[row][col] = chunk
    
    return group
