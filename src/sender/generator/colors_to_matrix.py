import math
import numpy as np
import copy

# 使用颜色流数据填充矩阵（矩阵需要先通过frame_gen.py生成基础结构），不包含解码器

EMPTY_SPACR_NUM = 16844 #hyperparameter
PATCH_SIZE = 16844 * 8 // 3 #每张图片可存储的加密字节数 

class MatrixHasNoneError(ValueError):
    """填充完成后矩阵仍有None的异常"""
    pass

class NoEmptyPositionError(ValueError):
    """无空余位置可填，但颜色列表未填充完成的异常"""
    pass

def colors_to_matrix(
    matrix: list[list[tuple[int, int, int] | None]],
    color_list: list[tuple[int, int, int]],
    patch_size: int = 16844, #现版本方案为固定值16844
    start_pos: int = 0,  # 填充起始一维索引（默认从0开始）
    step: int = 68        # 互质步长，建议选择68
) -> list[list[tuple[int, int, int] | None]]:
    """
    按互质步长填充法，将颜色列表填入137×137矩阵的可填充位置（None）
    :param matrix: 137×137的矩阵（列表套列表），None=可填充，RGB元组=固定结构
    :param color_list: 待填充的颜色列表（每个元素为RGB元组，如(255,0,0)）
    :param patch_size: 颜色列表的预期长度（需与color_list实际长度一致）
    :param start_pos: 填充起始一维索引（默认0，对应矩阵(0,0)）
    :param step: 互质步长（需与137互质，默认2）
    :return: 填充完成的137×137矩阵
    :raises TypeError: 输入类型非法
    :raises ValueError: 矩阵尺寸错误、颜色列表长度不匹配、步长不互质等
    :raises MatrixHasNoneError: 填充后矩阵仍有None
    :raises NoEmptyPositionError: 无空余位置但颜色列表未填完
    """
    N = 137  # 矩阵固定边长
    patch_size = 16844 #现版本方案为固定值16844
    total_pos = N * N  # 矩阵总位置数

    matrix = copy.deepcopy(matrix)  # 避免修改原矩阵

    # ===================== 1. 输入合法性校验 =====================
    # 校验矩阵尺寸
    if len(matrix) != N or any(len(row) != N for row in matrix):
        raise ValueError(f"矩阵必须是{N}×{N}的列表套列表，当前矩阵行数={len(matrix)}，列数={[len(row) for row in matrix][0] if matrix else 0}")
    
    # 校验颜色列表长度
    if len(color_list) != patch_size:
        raise ValueError(f"颜色列表长度必须为{patch_size}，当前长度={len(color_list)}")
    
    # 校验颜色列表元素（必须是RGB元组）
    for idx, color in enumerate(color_list):
        if not isinstance(color, tuple) or len(color) != 3 or any(not isinstance(c, int) or c < 0 or c > 255 for c in color):
            raise TypeError(f"颜色列表第{idx}个元素必须是(0-255,0-255,0-255)的RGB元组，当前为{color}")
    
    # 校验矩阵元素（只能是None或RGB元组）
    for i in range(N):
        for j in range(N):
            val = matrix[i][j]
            if val is not None and (not isinstance(val, tuple) or len(val) != 3 or any(not isinstance(c, int) or c < 0 or c > 255 for c in val)):
                raise TypeError(f"矩阵[{i}][{j}]位置元素非法，只能是None或RGB元组，当前为{val}")
    
    # 校验步长与N互质
    if math.gcd(step, N) != 1:
        raise ValueError(f"步长{step}与矩阵边长{N}不互质，请选择如2、3、4等步长（{N}是质数，除1和{N}外都互质）")

    # ===================== 2. 互质步长填充核心逻辑 =====================
    color_idx = 0  # 颜色列表的当前填充索引
    current_pos = start_pos  # 当前遍历的一维索引

    # 遍历次数上限为总位置数（避免无限循环）
    for _ in range(total_pos):
        # 一维索引转二维坐标（行优先：pos = i*N + j → i=pos//N, j=pos%N）
        i = current_pos // N
        j = current_pos % N

        # 仅填充None位置，跳过固定结构
        if matrix[i][j] is None:
            # 检查是否还有颜色可填
            if color_idx >= patch_size:
                break  # 颜色已填完，退出遍历
            # 填入颜色
            matrix[i][j] = color_list[color_idx]
            color_idx += 1

        # 按互质步长更新当前位置（模总位置数实现循环）
        current_pos = (current_pos + step) % total_pos

    # ===================== 3. 填充后校验 =====================
    # 检查1：颜色列表未填完但无空余位置
    if color_idx < patch_size:
        raise NoEmptyPositionError(f"无空余位置可填，但颜色列表仅填充了{color_idx}/{patch_size}个元素")
    
    # 检查2：填充完成后矩阵仍有None
    has_none = any(None in row for row in matrix)
    if has_none:
        # 统计剩余None数量，方便排查
        none_count = sum(row.count(None) for row in matrix)
        raise MatrixHasNoneError(f"填充完成后矩阵仍有{none_count}个None位置未填充")

    return matrix