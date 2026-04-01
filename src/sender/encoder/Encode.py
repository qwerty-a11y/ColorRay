import math
import os
from typing import List, Tuple

from common.CorrectionLevel import RaidLevel,RSLevel
from common.File import FileToBinary
from common.Config import FrameDataSize, GroupDataSize
from common.RSmodule import rs_encode_bytes, rs_decode_bytes
from common.Raid import Raid5Encode, Raid6Encode, Raid5Decode, Raid6Decode
from common.CRC16 import verify_crc16

def Encode(path:str, raid:RaidLevel, rs:RSLevel):
    binary = FileToBinary(path)
    
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
            RSCorrectBytesPerGroup = 10
            FileGroupSize = GroupDataSize - 2 - RSCorrectBytesPerGroup * 4
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = 20
            FileGroupSize = GroupDataSize - 2 - RSCorrectBytesPerGroup * 4
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = 40
            FileGroupSize = GroupDataSize - 2 - RSCorrectBytesPerGroup * 3
        case RSLevel.NONE:
            RSPercent = 0
            FileGroupSize = GroupDataSize - 2
            
    

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
            
            # 填充到对应位置
            data_groups[row][col] = chunk
            
            current_index += 1
            
        if current_index >= total_chunks:
            break

    # 为每个数据块添加 CRC16 校验码和 RS 纠错码
    for row in range(row_count):
        for col in range(col_count):
            # 【修改】跳过仍为 None 的位置（这些将是随机填充块，不需要校验和纠错）
            if data_groups[row][col] is None:
                continue
                
            from common.CRC16 import add_crc16
            data_groups[row][col] = rs_encode_bytes(add_crc16(data_groups[row][col]), RSCorrectBytesPerGroup)

    # 【修改新增】在所有真实数据完成校验和编码后，再填充随机数据并统一长度
    for row in range(row_count):
        for col in range(col_count):
            if data_groups[row][col] is None:
                # 生成与 GroupDataSize 长度一致的随机数据，确保所有块长度相同
                data_groups[row][col] = os.urandom(int(GroupDataSize))
            else:
                # 【新增】对于已有数据块，如果长度不满 GroupDataSize，进行补全
                current_len = len(data_groups[row][col])
                if current_len < int(GroupDataSize):
                    # 使用 \x00 填充至标准长度
                    padding_len = int(GroupDataSize) - current_len
                    data_groups[row][col] = data_groups[row][col] + os.urandom(padding_len) 

    # 若 Raid 等级不为 NONE，调用对应的 Raid 编码函数
    match raid:
        case RaidLevel.LEVEL1_10 | RaidLevel.LEVEL2_20:
            # RAID 5 编码：输入为 List[List[bytes]]，输出为扩展后的磁盘列表
            # 注意：Raid5Encode 期望输入是 [磁盘 1 块列表，磁盘 2 块列表...]
            # 当前 data_groups 是 [行][列]，需要转置或调整视角以符合“磁盘”定义
            # 假设当前 data_groups 的每一行代表一个原始数据盘的数据块序列
            data_groups = Raid5Encode(data_groups)
            
        case RaidLevel.LEVEL3_40:
            # RAID 6 编码
            data_groups = Raid6Encode(data_groups)
            
        case RaidLevel.NONE:
            # 无需 RAID 编码，保持原状
            pass

    return data_groups

def GroupToFrames(group: List[List[bytes]]) -> List[bytes]:
    """
    将二维数据组按列序号优先、行序号其次的顺序展平成帧列表
    每 8 个数据块组成一个帧
    
    :param group: 二维数据组 [行数][列数]
    :return: 帧列表，每个帧包含 8 个数据块的拼接字节
    
    示例（3行×4列）:
    [[A, B, C, D],
     [E, F, G, H],
     [I, J, K, L]]
    
    展平顺序: A, E, I, B, F, J, C, G, K, D, H, L
    分帧结果: [A+E+I+B+F+J+C+G, K+D+H+L+...]
    """
    frames = []
    
    if not group or not group[0]:
        return frames
    
    rows = len(group)
    cols = len(group[0])
    
    # 按列优先遍历，按行遍历每列的数据块
    blocks = []
    for col in range(cols):
        for row in range(rows):
            if group[row][col] is not None:
                blocks.append(group[row][col])
    
    # 每 8 个块组成一个帧
    for i in range(0, len(blocks), 8):
        frame = b''
        for j in range(i, min(i + 8, len(blocks))):
            frame += blocks[j]
        frames.append(frame)
    
    return frames


def FramesToGroup(frames: List[bytes], raid: RaidLevel) -> List[List[bytes]]:
    """
    将帧列表恢复为二维数据组（GroupToFrames 的反向操作）
    
    :param frames: 帧列表，每个帧为一个块（GroupDataSize 字节）
    :param raid: RAID 等级，用于确定行数
    :return: 二维数据组 [行数][列数]
    
    工作流程：
    1. 每个帧即一个块，直接分为 8 份（对应原始的 8 行）
    2. 根据 RAID 等级将 8 份块按行数重新组织
    3. 最终形成二维数组
    """
    # 确定行数
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
        case _:
            raise ValueError(f"未知的 RAID 等级: {raid}")
    
    if row_count == 0:
        return []
    
    if not frames:
        return [[]]
    
    # 每个帧分为 8 份数据块
    blocks = []
    chunk_size = GroupDataSize // 8  # 帧被分成 8 份
    
    for frame in frames:
        for i in range(8):
            start = i * chunk_size
            end = start + chunk_size if i < 7 else len(frame)  # 最后一份包含剩余数据
            if start < len(frame):
                blocks.append(frame[start:end])
    
    # 根据块数和行数计算列数
    col_count = math.ceil(len(blocks) / row_count)
    
    # 初始化二维数组
    group: List[List[bytes]] = [[None for _ in range(col_count)] for _ in range(row_count)]
    
    # 按列优先顺序填充（与 GroupToFrames 的展平顺序相反）
    block_index = 0
    for col in range(col_count):
        for row in range(row_count):
            if block_index < len(blocks):
                group[row][col] = blocks[block_index]
                block_index += 1
    
    return group


