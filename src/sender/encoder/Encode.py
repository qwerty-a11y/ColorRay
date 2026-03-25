import math
import os
from typing import List

from common.CorrectionLevel import RaidLevel,RSLevel
from common.File import FileToBinary
from common.Config import FrameDataSize, GroupDataSize
from common.RSmodule import rs_encode_bytes, rs_decode_bytes
from common.Raid import Raid5Encode, Raid6Encode

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
    match rs:
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = 10
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = 20
        case RSLevel.LEVEL3_20:
            RSCorrectBytesPerGroup = 40
        case RSLevel.NONE:
            RSPercent = 0
            
    FileGroupSize = GroupDataSize - 2 - RSCorrectBytesPerGroup * 4
    
    # 若校验位部分为奇数，则使数据部分结果再减一
    rs_part = (GroupDataSize - 2) * RSPercent / 100
    if rs_part % 2 != 0:
        FileGroupSize -= 1

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

    # 【修改新增】在所有真实数据完成校验和编码后，再填充随机数据
    for row in range(row_count):
        for col in range(col_count):
            if data_groups[row][col] is None:
                # 生成与正常数据块长度一致的随机数据
                # 注意：这里直接生成原始长度的随机数，不加 CRC 和 RS，符合需求
                data_groups[row][col] = os.urandom(GroupDataSize)

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
    frames = []
    index = 0
    finish = False
    while True:
        for row in group:
            frame = b''
            for i in range(8):
                if i+index >= len(row):
                    finish = True
                    break
                if row[i+index] is not None:
                    frame += row[i+index]
            if finish:
                break
            frames.append(frame)
        if finish:
            break
        index+=8
    return frames