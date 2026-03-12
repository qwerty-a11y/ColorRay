import numpy as np

# 生成基础结构，后续建议根据基础结构做解码器
# 修改自@kafuchino的代码，将其提取成函数

def generate_frame() -> tuple[list[list[tuple[int, int, int]]], list[list[bool]]]:
    """
    生成137×137的固定结构颜色矩阵，以及标记需要边框的矩阵
    矩阵规则：
    1. 最外侧2个网格宽度为纯白色边框；
    2. 包含左上/左下/右上3个大定位块、右下1个小定位块；
    3. 右下角有8种标准颜色的色块；
    4. 特定位置有黑/蓝色块；
    5. 未定义区域（原None）随机填充标准颜色，并标记需要边框。
    
    :return: 
        color_grid: 137×137的颜色矩阵（每个元素为RGB元组）
        need_border_grid: 137×137的布尔矩阵（True=需要边框，False=无需边框）
    """
    # ===================== 固定参数（与原逻辑一致） =====================
    CELL_PIXELS = 12  # 每个网格单元的像素大小 (12x12 像素)
    IMG_SIZE = 1644   # 图像总尺寸 1644x1644
    GRID_COUNT = IMG_SIZE // CELL_PIXELS  # 网格数量 = 1644/12 = 137

    # 8 种标准颜色 (RGB)
    COLORS = [
        (0, 0, 0),    # 黑
        (0, 0, 255),  # 蓝
        (0, 255, 0),  # 绿
        (0, 255, 255),# 青
        (255, 0, 0),  # 红
        (255, 0, 255),# 品红
        (255, 255, 0),# 黄
        (255, 255, 255)# 白
    ]
    BG_GRAY = (128, 128, 128)  # 边框灰色（原逻辑未实际使用，保留）

    # ===================== 初始化矩阵 =====================
    # 颜色矩阵：初始为None（未定义区域）
    color_grid:list[list[tuple[int,int,int]|None]] = [[None for _ in range(GRID_COUNT)] for _ in range(GRID_COUNT)]
    # 边框标记矩阵：初始为False（无需边框）
    need_border_grid = [[False for _ in range(GRID_COUNT)] for _ in range(GRID_COUNT)]

    # ===================== 内部辅助函数 =====================
    def set_cell(r: int, c: int, color: tuple[int, int, int]) -> None:
        """设置指定网格的颜色（边界安全校验）"""
        if 0 <= r < GRID_COUNT and 0 <= c < GRID_COUNT:
            color_grid[r][c] = color

    def draw_filled_rect(r1: int, c1: int, r2: int, c2: int, color: tuple[int, int, int]) -> None:
        """填充矩形区域（包含边界，r1/r2/c1/c2为网格坐标，边界安全校验）"""
        # 确保坐标不超出网格范围
        r_start = max(0, r1)
        r_end = min(r2, GRID_COUNT - 1)
        c_start = max(0, c1)
        c_end = min(c2, GRID_COUNT - 1)
        
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                set_cell(r, c, color)

    # ===================== 1. 绘制最外侧2单位宽度的纯白色边框 =====================
    # 上边2行
    for r in range(2):
        for c in range(GRID_COUNT):
            set_cell(r, c, (255, 255, 255))
    # 下边2行
    for r in range(GRID_COUNT - 2, GRID_COUNT):
        for c in range(GRID_COUNT):
            set_cell(r, c, (255, 255, 255))
    # 左边2列（排除上下已填充的2行）
    for r in range(2, GRID_COUNT - 2):
        for c in range(2):
            set_cell(r, c, (255, 255, 255))
    # 右边2列（排除上下已填充的2行）
    for r in range(2, GRID_COUNT - 2):
        for c in range(GRID_COUNT - 2, GRID_COUNT):
            set_cell(r, c, (255, 255, 255))

    # ===================== 2. 绘制三个大定位块（左上、左下、右上） =====================
    # 规则：从外到内 14x14黑 → 10x10白 → 6x6黑（基准坐标从2开始，避开外侧白色边框）
    # 左上定位块：(2,2) 开始
    draw_filled_rect(2, 2, 15, 15, (0, 0, 0))       # 14x14 黑色（2-15共14格）
    draw_filled_rect(4, 4, 13, 13, (255, 255, 255)) # 10x10 白色
    draw_filled_rect(6, 6, 11, 11, (0, 0, 0))       # 6x6 黑色

    # 右上定位块：(2, GRID_COUNT-16) 开始
    draw_filled_rect(2, GRID_COUNT - 16, 15, GRID_COUNT - 3, (0, 0, 0))       # 14x14 黑色
    draw_filled_rect(4, GRID_COUNT - 14, 13, GRID_COUNT - 5, (255, 255, 255)) # 10x10 白色
    draw_filled_rect(6, GRID_COUNT - 12, 11, GRID_COUNT - 7, (0, 0, 0))       # 6x6 黑色

    # 左下定位块：(GRID_COUNT-16, 2) 开始
    draw_filled_rect(GRID_COUNT - 16, 2, GRID_COUNT - 3, 15, (0, 0, 0))       # 14x14 黑色
    draw_filled_rect(GRID_COUNT - 14, 4, GRID_COUNT - 5, 13, (255, 255, 255)) # 10x10 白色
    draw_filled_rect(GRID_COUNT - 12, 6, GRID_COUNT - 7, 11, (0, 0, 0))       # 6x6 黑色

    # ===================== 3. 绘制右下小定位块 =====================
    # 起始位置：距离下方/右方白色边框内侧4单位，向左上扩展
    small_finder_start = GRID_COUNT - 2 - 4 - 10  # 137-2-4-10=121
    # 外层：12x12 白色（修正原注释笔误）
    draw_filled_rect(small_finder_start - 1, small_finder_start - 1,
                     small_finder_start + 10, small_finder_start + 10, (255, 255, 255))
    # 中层：10x10 黑色
    draw_filled_rect(small_finder_start, small_finder_start,
                     small_finder_start + 9, small_finder_start + 9, (0, 0, 0))
    # 内层：6x6 白色
    draw_filled_rect(small_finder_start + 2, small_finder_start + 2,
                     small_finder_start + 7, small_finder_start + 7, (255, 255, 255))
    # 核心：2x2 黑色
    draw_filled_rect(small_finder_start + 4, small_finder_start + 4,
                     small_finder_start + 5, small_finder_start + 5, (0, 0, 0))

    # ===================== 4. 绘制右下角8种标准色块（2行4列） =====================
    # 位置：紧贴白色边框内侧，倒数第4行、倒数第6列开始
    std_color_start_r = GRID_COUNT - 4
    std_color_start_c = GRID_COUNT - 6
    for k in range(8):
        r = std_color_start_r + (k // 4)  # 前4个在第1行，后4个在第2行
        c = std_color_start_c + (k % 4)   # 每行4列依次排列
        set_cell(r, c, COLORS[k])

    # ===================== 5. 给大定位块添加1单位宽度的白色边框 =====================
    # 左上定位块边框：围绕14x14区域，右/下方各扩展1格
    for c in range(2, 17):
        set_cell(16, c, (255, 255, 255))  # 下方边框
    for r in range(2, 17):
        set_cell(r, 16, (255, 255, 255))  # 右方边框

    # 右上定位块边框：围绕14x14区域，下/左方各扩展1格
    for c in range(GRID_COUNT - 17, GRID_COUNT - 2):
        set_cell(16, c, (255, 255, 255))  # 下方边框
    for r in range(2, 17):
        set_cell(r, GRID_COUNT - 17, (255, 255, 255))  # 左方边框

    # 左下定位块边框：围绕14x14区域，上/右方各扩展1格
    for c in range(2, 17):
        set_cell(GRID_COUNT - 17, c, (255, 255, 255))  # 上方边框
    for r in range(GRID_COUNT - 17, GRID_COUNT - 2):
        set_cell(r, 16, (255, 255, 255))  # 右方边框（与左上共享）

    # ===================== 6. 绘制特定位置的黑/蓝色块 =====================
    # 左上定位块边框下方靠左：(17,2)开始，前4黑、后2蓝
    for k in range(6):
        color = (0, 0, 0) if k < 4 else (0, 0, 255)
        set_cell(17, 2 + k, color)

    # 左下定位块边框上方靠左：(GRID_COUNT-18,2)开始，前4黑、后2蓝
    for k in range(6):
        color = (0, 0, 0) if k < 4 else (0, 0, 255)
        set_cell(GRID_COUNT - 18, 2 + k, color)

    # 右上定位块边框下方靠右：(17, GRID_COUNT-8)开始，前4黑、后2蓝
    for k in range(6):
        color = (0, 0, 0) if k < 4 else (0, 0, 255)
        set_cell(17, GRID_COUNT - 8 + k, color)

    # ===================== 7. 处理未定义区域（原None） =====================
    for r in range(GRID_COUNT):
        for c in range(GRID_COUNT):
            if color_grid[r][c] is None:
                # 随机填充标准颜色
                # color_grid[r][c] = random.choice(COLORS)
                # 标记该位置需要边框
                need_border_grid[r][c] = True
   
    return color_grid, need_border_grid