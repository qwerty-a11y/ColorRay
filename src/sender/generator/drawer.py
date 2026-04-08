import numpy as np
from PIL import Image, ImageDraw
import datetime
import os, sys

# 获取当前文件所在目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 向上跳两层，得到项目根目录
project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
# 将根目录加入 Python 模块搜索路径
sys.path.insert(0, project_root)

import common.Config as Config

#输入颜色矩阵和需要边框矩阵，生成图像并保存

def drawer(grid, need_border, filename, filepath=None):

    CELL_PIXELS = 12  # 每个网格单元的像素大小 (12x12 像素)
    GRID_COUNT = Config.QRSize + 4  # 网格数量 = 1644/12 = 137
    IMG_SIZE = GRID_COUNT*CELL_PIXELS   # 图像总尺寸 1644x1644
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

    # 1. 如果没有指定文件名，生成带时间戳的默认文件名
    if filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'color_qr_{timestamp}.png'

    # 2. 检查并创建目标文件夹 (filepath)
    if filepath is None:
        filepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))+'/data'


    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print(f"创建了目录: {filepath}")

    # 3. 拼接完整保存路径
    # 使用 os.path.join 可以自动处理不同系统的斜杠问题 (Windows '\' vs Linux '/')
    full_path = os.path.join(filepath, filename)

    # 4. 保存图片
    img.save(full_path)
    print(f"图像已生成：{filename}")
    print(f"图像尺寸：{IMG_SIZE} x {IMG_SIZE} 像素")
    print(f"网格：{GRID_COUNT} x {GRID_COUNT} (每个单元{CELL_PIXELS}x{CELL_PIXELS}像素)")

    return full_path


def mem_drawer(grid, need_border):

    CELL_PIXELS = 12  # 每个网格单元的像素大小 (12x12 像素)
    GRID_COUNT = Config.QRSize + 4  # 网格数量 = 1644/12 = 137
    IMG_SIZE = GRID_COUNT*CELL_PIXELS   # 图像总尺寸 1644x1644
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

    print(f"图像尺寸：{IMG_SIZE} x {IMG_SIZE} 像素")
    print(f"网格：{GRID_COUNT} x {GRID_COUNT} (每个单元{CELL_PIXELS}x{CELL_PIXELS}像素)")

    return img
