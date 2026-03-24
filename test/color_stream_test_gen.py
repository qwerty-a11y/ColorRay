import numpy as np
import random
import src.common.Config as Config

STANDARD_COLORS = [
        (0, 0, 0),      # 黑
        (0, 0, 255),    # 蓝
        (0, 255, 0),    # 绿
        (0, 255, 255),  # 青
        (255, 0, 0),    # 红
        (255, 0, 255),  # 品红
        (255, 255, 0),  # 黄
        (255, 255, 255) # 白
    ]

def generate_discrete_block_color_array() -> list[tuple[int, int, int]]:
    """
    生成符合以下规则的测试颜色列表（长度等于 Config.DataBlocks）：
    1. 每连续16个颜色为1个块，块内颜色完全相同；
    2. 每个块的颜色从8种标准色中随机轮换选取（允许重复）；
    3. 总块数 = DataBlocks//16；若有收尾残余则复用最后一色填满。

    :return: 长度为 Config.DataBlocks 的颜色列表
    """
    # 核心参数
    TOTAL_LENGTH = Config.DataBlocks    # 总长度
    BLOCK_SIZE = 16         # 每个颜色块的长度
    FULL_BLOCK_COUNT = TOTAL_LENGTH // BLOCK_SIZE
    REMAIN_COUNT = TOTAL_LENGTH % BLOCK_SIZE       # 剩余位置数：0

    # 1. 生成颜色池：从8种标准色中随机选
    color_pool = [random.choice(STANDARD_COLORS) for _ in range(FULL_BLOCK_COUNT)]

    # 2. 生成完整颜色块（每个块16个相同颜色）
    color_array = []
    for block_idx in range(FULL_BLOCK_COUNT):
        current_color = color_pool[block_idx]
        color_array.extend([current_color] * BLOCK_SIZE)

    # 3. 收尾残余（现为0，逻辑保留）
    last_block_color = color_pool[-1] if color_pool else STANDARD_COLORS[0]
    color_array.extend([last_block_color] * REMAIN_COUNT)

    # 4. 验证逻辑
    assert len(color_array) == TOTAL_LENGTH, \
        f"数组长度错误，预期{TOTAL_LENGTH}，实际{len(color_array)}"
    for i in range(3):
        block_start = i * BLOCK_SIZE
        block_end = (i + 1) * BLOCK_SIZE - 1
        block_colors = color_array[block_start:block_end + 1]
        assert all(c == block_colors[0] for c in block_colors), \
            f"第{i}个块内颜色不一致"

    print(f"颜色数组生成完成！")
    print(f"- 总长度：{len(color_array)}（{FULL_BLOCK_COUNT}个完整16色块 + {REMAIN_COUNT}个收尾位置）")
    print(f"- 前5个块颜色（随机轮换）：")
    for i in range(5):
        block_start = i * BLOCK_SIZE
        print(f"  第{i+1}块（位置{block_start}~{block_start+15}）：{color_array[block_start]}")
    if REMAIN_COUNT:
        print(f"- 最后{REMAIN_COUNT}个位置颜色：{color_array[-1]}（复用最后一个块颜色）")

    return color_array

def generate_random_colors() -> list[tuple[int, int, int]]:
    """
    从指定的8种基础颜色中随机选择并生成颜色列表。

    返回:
    - List[tuple(int, int, int)]: RGB 颜色元组列表
    """

    num_blocks = Config.DataBlocks

    if num_blocks <= 0:
        return []

    return random.choices(STANDARD_COLORS, k=num_blocks)
