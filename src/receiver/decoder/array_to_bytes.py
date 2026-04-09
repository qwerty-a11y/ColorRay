import math

import numpy as np

from common import Config
from receiver.decoder.colors_to_bytes import colors_to_bytes

# 注意解析时的基础结构矩阵、起始位置、步长必须与填充时完全一致，否则会导致解析失败（颜色数量不符或结构不匹配）

# 复用原有异常类，新增矩阵结构不一致异常
class MatrixHasNoneError(ValueError):
    """填充完成后矩阵仍有None的异常"""
    pass

class NoEmptyPositionError(ValueError):
    """无空余位置可填，但颜色列表未填充完成的异常"""
    pass

class ParseColorCountError(ValueError):
    """解析出的颜色数量与预期patch_size不符的异常"""
    pass

class MatrixStructureMismatchError(ValueError):
    """frame矩阵与填充矩阵的基础结构（非None位置）颜色不一致的异常"""
    pass

def numpy_to_int(nparray: np.ndarray, x: int, y: int) -> int:
    B = nparray[x*6,y*6,0]
    G = nparray[x*6,y*6,1]
    R = nparray[x*6,y*6,2]
    match (R, G, B):
        case (0,0,0):
            return 0
        case (0,0,255):
            return 1 
        case (0,255,0):
            return 2
        case (0,255,255):
            return 3
        case (255,0,0):
            return 4
        case (255,0,255):
            return 5
        case (255,255,0):
            return 6
        case (255,255,255):
            return 7
        case _:
            raise ValueError(f"无法识别的RGB值：{R},{G},{B}")


def array_to_bytes(
    frame_matrix: list[list[tuple[int,int,int]]],  # 原始 frame 矩阵（基础结构 +None）
    filled_matrix: np.ndarray,        # 填充后的矩阵（基础结构 + 填充色）
    patch_size: int = Config.FrameDataBlocks,                           # 预期提取的颜色列表长度
    start_pos: int = 0,                                     # 填充时的起始一维索引
    step: int = 68                                          # 填充时使用的互质步长
) -> bytes:
    
    N = Config.QRSize  # 矩阵固定边长
    patch_size = Config.FrameDataBlocks  # 固定提取长度
    
    total_pos = (N+4)*(N+4)  # 矩阵总位置数
    # 检查是否需要补白 (输入为 133x133 时)

    # 校验步长与N互质
    if math.gcd(step, N) != 1:
        raise ValueError(f"步长{step}与矩阵边长{N}不互质，请使用与{N}互质的步长（如68）")

    # ===================== 2. 反向解析核心逻辑（对比双矩阵） =====================
    color_indices = []  # 存储解析出的颜色索引列表 (0-7)
    current_pos = start_pos  # 从填充时的起始位置开始遍历

    # 遍历次数上限为总位置数（避免无限循环）
    for _ in range(total_pos):
        # 一维索引转二维坐标（行优先：pos = i*N + j → i=pos//N, j=pos%N）
        i = current_pos // (N+4)
        j = current_pos % (N+4)

        # 仅收集frame_matrix中为None的位置（填充位）的颜色
        # 注意：原代码中 frame_matrix[i+2][j+2] 暗示 matrix 有 padding，需确保索引不越界
        if frame_matrix[i][j] is None:
            # 检查是否已收集满预期数量，避免多余收集
            if len(color_indices) < patch_size:
                # 获取该位置的 RGB 并转换为索引 (0-7)
                idx_val = numpy_to_int(filled_matrix, i-2, j-2)
                color_indices.append(idx_val)
        
        # 按互质步长更新当前位置（与填充函数完全一致）
        current_pos = (current_pos + step) % total_pos

        # 提前终止：已收集满
        if len(color_indices) == patch_size:
            break

    # ===================== 3. 解析后校验 =====================
    if len(color_indices) != patch_size:
        raise ParseColorCountError(
            f"解析出的颜色数量为{len(color_indices)}，与预期的{patch_size}不符！\n"
            f"可能原因：1.步长/起始位置与填充时不一致；2.frame矩阵与填充矩阵不匹配；3.填充矩阵未填充满"
        )

    # 5. 转换为bytes并返回
    return colors_to_bytes(color_indices)
