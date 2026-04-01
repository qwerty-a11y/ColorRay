import numpy as np
from PIL import Image
import os
import cv2

def array_to_pillow_image(img_array: np.ndarray) -> Image.Image:
    """
    将 numpy 数组 (_Array[tuple[int, int, int], unsignedinteger[_8Bit]]) 转换为 Pillow 图片对象
    
    :param img_array: 形状为 (H, W, 3) 的 numpy 数组，元素为 0-255 的整数
    :return: PIL.Image.Image 对象
    """
    # 确保数据在 0-255 范围内并转换为 uint8 类型
    safe_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # 创建 PIL 图片对象，指定模式为 RGB
    return Image.fromarray(safe_array, mode='RGB')

def array_to_matrix(img_array: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """
    将 numpy 矩阵转换为 RGB 元组矩阵
    :param img_array: 形状为 (H, W, 3) 的 numpy 数组，元素为 0-255 的整数
    :return: RGB 元组矩阵（列表套列表）
    """
    img = array_to_pillow_image(img_array)
    return pillow_to_matrix(img)
def pillow_to_matrix(image: Image.Image, run_mode: str = 'normal') -> list[list[tuple[int, int, int]]]:
    """
    将 Pillow 图片对象转换为 137×137 的 RGB 元组矩阵
    :param image: PIL.Image.Image 对象
    :param run_mode: 运行模式，'test' 或 'normal'，决定使用 137×137 的完整矩阵（test）还是 133×133 的核心矩阵（normal）
    :return: 137×137 的 RGB 元组矩阵（列表套列表）
    """
    block_num = 137 if run_mode == 'test' else 133
    win_size = 11

    img = image.convert('RGB')
    target_size = block_num * win_size
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    # save_debug_image(img_array, "01_original", prefix=file_prefix)

    # --- [核心修改] 步骤 A: 强行将全局 RGB 均值校准到 128 ---
    # 计算当前各通道均值
    current_means = np.mean(img_array, axis=(0, 1)) # [mean_r, mean_g, mean_b]
    
    # 计算增益系数，目标是 128
    # 使用 1e-6 防止除以 0（全黑图）
    gains = 128.0 / (current_means + 1e-6)
    
    # 限制增益范围，防止噪点被无限放大（可选，此处设为最大 5 倍增益）
    gains = np.clip(gains, 0.1, 5.0)
    
    # 应用增益
    img_array[:, :, 0] *= gains[0]
    img_array[:, :, 1] *= gains[1]
    img_array[:, :, 2] *= gains[2]
    
    # 这一步之后，图片的平均亮度会刚好处于中等水平，且消除了明显的色偏
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    # save_debug_image(img_array, "02_global_128_calibration", prefix=file_prefix)

    # --- 步骤 B: 提高饱和度 (池化前) ---
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    # 既然均值已经拉到了 128，饱和度可以更激进一点
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.8, 0, 255) 
    
    enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # save_debug_image(enhanced_rgb, "03_pre_conv_saturation_boosted", prefix=file_prefix)

    # --- 步骤 C: 严格分块加权平均 ---
    def get_gaussian_kernel(size, sigma=1.5):
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / kernel.sum()

    kernel = get_gaussian_kernel(win_size, sigma=0.5)
    
    # 分块逻辑 (Target_H, Target_W, C) -> (block, block, 11, 11, 3)
    blocks = enhanced_rgb.reshape(block_num, win_size, block_num, win_size, 3).transpose(0, 2, 1, 3, 4)
    # 向量化加权平均
    pooled_array = np.einsum('ijklm,kl->ijm', blocks, kernel)
    # save_debug_image(pooled_array, "04_after_block_pooled", prefix=file_prefix)

    # --- 步骤 D: 最终二值化判定 ---
    binary_array = np.where(pooled_array < 128, 0, 255).astype(np.uint8)
    # save_debug_image(binary_array, "05_final_binary", prefix=file_prefix)

    # --- 步骤 E: Padding 137 ---
    final_size = 137
    final_array = np.full((final_size, final_size, 3), 255, dtype=np.uint8)
    start = 2 if run_mode == 'normal' else 0
    # 注意：这里要确保切片范围正确
    final_array[start:start+block_num, start:start+block_num, :] = binary_array[:block_num, :block_num]

    return [[tuple(map(int, final_array[r, c])) for c in range(final_size)] for r in range(final_size)]

def save_debug_image(img_array, stage_name, prefix="debug"):
    """保存调试图片，覆盖旧文件"""
    output_dir = "debug_output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_arr = np.clip(img_array, 0, 255).astype(np.uint8)
    filepath = os.path.join(output_dir, f"{prefix}_{stage_name}.png")
    Image.fromarray(save_arr).save(filepath)

def image_to_matrix(image_path: str, run_mode='normal') -> list[list[tuple[int, int, int]]]:
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        raise OSError(f"无法打开图像文件 {image_path}")
    return pillow_to_matrix(image, run_mode)