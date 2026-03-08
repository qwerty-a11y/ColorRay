import numpy as np
from PIL import Image, ImageDraw
import datetime

#输入颜色矩阵和需要边框矩阵，生成图像并保存

def drawer(grid, need_border):

    CELL_PIXELS = 12  # 每个网格单元的像素大小 (12x12 像素)
    IMG_SIZE = 1644   # 图像总尺寸 1644x1644
    GRID_COUNT = IMG_SIZE // CELL_PIXELS  # 网格数量 = 1644/12 = 137
    BG_GRAY = (128, 128, 128)

    # ========== 创建图像并绘制 ==========
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), BG_GRAY)
    draw = ImageDraw.Draw(img)

    # 绘制所有单元格
    for r in range(GRID_COUNT):
        for c in range(GRID_COUNT):
            x0 = c * CELL_PIXELS
            y0 = r * CELL_PIXELS
            x1 = x0 + CELL_PIXELS - 1
            y1 = y0 + CELL_PIXELS - 1

            # 绘制色块
            draw.rectangle([x0, y0, x1, y1], fill=grid[r][c])

            # 只有需要边框的才绘制灰色边框
            if need_border[r][c]:
                draw.rectangle([x0, y0, x1, y1], outline=BG_GRAY, width=1)

    # ========== 生成带时间的文件名 ==========
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'color_qr_{timestamp}.png'
    img.save(filename)
    print(f"图像已生成：{filename}")
    print(f"图像尺寸：{IMG_SIZE} x {IMG_SIZE} 像素")
    print(f"网格：{GRID_COUNT} x {GRID_COUNT} (每个单元{CELL_PIXELS}x{CELL_PIXELS}像素)")
