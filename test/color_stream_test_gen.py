import numpy as np
import random
def generate_discrete_block_color_array() -> list[tuple[int, int, int]]:
    """
    生成符合以下规则的测试颜色列表（长度16844）：
    1. 每连续16个颜色为1个块，块内颜色完全相同；
    2. 每个块的颜色从8种标准色中随机轮换选取（允许重复）；
    3. 总长度16844（1052个完整块+最后4个收尾位置）；
    4. 最后4个位置复用最后一个块的颜色（随意处理）。
    
    :return: 长度为16844的颜色列表
    """
    # 核心参数
    TOTAL_LENGTH = 16844    # 总长度
    BLOCK_SIZE = 16         # 每个颜色块的长度
    FULL_BLOCK_COUNT = TOTAL_LENGTH // BLOCK_SIZE  # 完整块数：16844//16=1052
    REMAIN_COUNT = TOTAL_LENGTH % BLOCK_SIZE       # 剩余位置数：4

    # 定义8种标准颜色（固定）
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

    # 1. 生成颜色池：从8种标准色中随机选，生成1052个块的颜色（随机轮换）
    color_pool = [random.choice(STANDARD_COLORS) for _ in range(FULL_BLOCK_COUNT)]

    # 2. 生成完整颜色块（前1052个块，每个块16个相同颜色）
    color_array = []
    for block_idx in range(FULL_BLOCK_COUNT):
        current_color = color_pool[block_idx]
        # 每个块添加16个相同颜色
        color_array.extend([current_color] * BLOCK_SIZE)

    # 3. 处理最后4个位置（复用最后一个块的颜色）
    last_block_color = color_pool[-1]
    color_array.extend([last_block_color] * REMAIN_COUNT)

    # 4. 验证逻辑
    # 验证总长度
    assert len(color_array) == TOTAL_LENGTH, \
        f"数组长度错误，预期{TOTAL_LENGTH}，实际{len(color_array)}"
    # 验证前3个完整块的块内颜色一致性（随机轮换允许块间重复）
    for i in range(3):
        block_start = i * BLOCK_SIZE
        block_end = (i + 1) * BLOCK_SIZE - 1
        # 块内颜色一致
        block_colors = color_array[block_start:block_end + 1]
        assert all(c == block_colors[0] for c in block_colors), \
            f"第{i}个块内颜色不一致"

    # 打印关键信息
    print(f"颜色数组生成完成！")
    print(f"- 总长度：{len(color_array)}（{FULL_BLOCK_COUNT}个完整16色块 + {REMAIN_COUNT}个收尾位置）")
    print(f"- 前5个块颜色（随机轮换）：")
    for i in range(5):
        block_start = i * BLOCK_SIZE
        print(f"  第{i+1}块（位置{block_start}~{block_start+15}）：{color_array[block_start]}")
    print(f"- 最后4个位置（16833~16843）颜色：{color_array[-1]}（复用最后一个块颜色）")

    return color_array
