from typing import List

from common.CorrectionLevel import RaidLevel,RSLevel
from common.File import FileToBinary
from common.Config import FrameDataSize, GroupDataSize
from common.RSmodule import rs_encode_bytes, rs_decode_bytes
import math

def Encode(path:str, raid:RaidLevel, rs:RSLevel):
    binary = FileToBinary(path)
    
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
    
    # 初始化二维数组 [row_count][col_count]
    # 计算需要的列数
    col_count = math.ceil(total_chunks / row_count) if row_count > 0 else 0
    
    # 创建二维列表，初始化为空列表或 None
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

    



