import math

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

def matrix_to_colors(
    frame_matrix: list[list[tuple[int, int, int] | None]],  # 原始frame矩阵（基础结构+None）
    filled_matrix: list[list[tuple[int, int, int]]],        # 填充后的矩阵（基础结构+填充色）
    patch_size: int = 16844,                                # 预期提取的颜色列表长度
    start_pos: int = 0,                                     # 填充时的起始一维索引
    step: int = 68                                          # 填充时使用的互质步长
) -> list[tuple[int, int, int]]:
    """
    对比原始frame矩阵和填充后的矩阵，按互质步长反向解析出原始颜色列表（仅包含填充色，排除基础结构）
    :param frame_matrix: 137×137的原始frame矩阵（基础结构为RGB元组，可填充位为None）
    :param filled_matrix: 137×137的填充后矩阵（基础结构不变，可填充位为填充色）
    :param patch_size: 预期提取的颜色列表长度（固定16844）
    :param start_pos: 填充时的起始一维索引（需与fill_matrix的start_pos一致）
    :param step: 填充时使用的互质步长（需与fill_matrix的step一致）
    :return: 还原后的原始颜色列表（仅包含填充的16844个颜色，无基础结构色）
    :raises TypeError: 输入类型非法
    :raises ValueError: 矩阵尺寸错误、步长不互质等
    :raises MatrixStructureMismatchError: frame矩阵与填充矩阵的基础结构不一致
    :raises ParseColorCountError: 解析出的颜色数量与patch_size不符
    """
    N = 137  # 矩阵固定边长
    patch_size = 16844  # 固定提取长度
    total_pos = N * N   # 矩阵总位置数

    # ===================== 1. 输入合法性校验 =====================
    # 校验两个矩阵的尺寸是否为137×137
    for mat_name, matrix in {"frame_matrix": frame_matrix, "filled_matrix": filled_matrix}.items():
        if len(matrix) != N or any(len(row) != N for row in matrix):
            raise ValueError(
                f"{mat_name}必须是{N}×{N}的列表套列表，当前行数={len(matrix)}，"
                f"列数={[len(row) for row in matrix][0] if matrix else 0}"
            )
    
    # 校验frame_matrix元素类型（只能是None或RGB元组）
    for i in range(N):
        for j in range(N):
            val = frame_matrix[i][j]
            if val is not None and (not isinstance(val, tuple) or len(val) != 3 or any(not isinstance(c, int) or c < 0 or c > 255 for c in val)):
                raise TypeError(f"frame_matrix[{i}][{j}]元素非法，只能是None或RGB元组，当前为{val}")
    
    # 校验filled_matrix元素类型（只能是RGB元组，无None）
    for i in range(N):
        for j in range(N):
            val = filled_matrix[i][j]
            if not isinstance(val, tuple) or len(val) != 3 or any(not isinstance(c, int) or c < 0 or c > 255 for c in val):
                raise TypeError(f"filled_matrix[{i}][{j}]元素非法，必须是RGB元组，当前为{val}")

    # 校验步长与N互质
    if math.gcd(step, N) != 1:
        raise ValueError(f"步长{step}与矩阵边长{N}不互质，请使用与{N}互质的步长（如68）")

    # ===================== 2. 反向解析核心逻辑（对比双矩阵） =====================
    color_list = []  # 存储解析出的原始颜色列表
    current_pos = start_pos  # 从填充时的起始位置开始遍历

    # 遍历次数上限为总位置数（避免无限循环）
    for _ in range(total_pos):
        # 一维索引转二维坐标（行优先：pos = i*N + j → i=pos//N, j=pos%N）
        i = current_pos // N
        j = current_pos % N

        # 仅收集frame_matrix中为None的位置（填充位）的颜色
        if frame_matrix[i][j] is None:
            # 检查是否已收集满预期数量，避免多余收集
            if len(color_list) < patch_size:
                color_list.append(filled_matrix[i][j])
        
        # 按互质步长更新当前位置（与填充函数完全一致）
        current_pos = (current_pos + step) % total_pos

        # 提前终止：已收集满16844个颜色
        if len(color_list) == patch_size:
            break

    # ===================== 3. 解析后校验 =====================
    if len(color_list) != patch_size:
        raise ParseColorCountError(
            f"解析出的颜色数量为{len(color_list)}，与预期的{patch_size}不符！\n"
            f"可能原因：1.步长/起始位置与填充时不一致；2.frame矩阵与填充矩阵不匹配；3.填充矩阵未填充满"
        )

    return color_list