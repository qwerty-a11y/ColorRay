import os

import numpy as np
from PIL import Image

import common.Config as Config


def save_debug_image(img_array, stage_name, prefix="debug"):
    """保存调试图片，覆盖旧文件"""
    output_dir = "debug_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_arr = np.clip(img_array, 0, 255).astype(np.uint8)
    filepath = os.path.join(output_dir, f"{prefix}_{stage_name}.png")
    Image.fromarray(save_arr).save(filepath)


def _prepare_square_then_resize_for_grid(img: Image.Image) -> Image.Image:
    """
    将任意尺寸画面变为与 drawer 一致的 (QRSize+4)*12 方形图。
    非正方形时取中心正方形；缩放一律最近邻，避免 LANCZOS/滤波破坏 tribit（直接导致 CRC 失败）。
    """
    grid_n = Config.QRSize + 4
    cell = 12
    target = grid_n * cell
    img = img.convert("RGB")
    w, h = img.size
    if w == target and h == target:
        return img
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    return img.resize((target, target), Image.Resampling.NEAREST)


def _matrix_from_exact_drawer_grid(
    img: Image.Image,
    *,
    majority_cell: bool = False,
    inner_margin: int = 0,
    cell_vote: str = "majority",
) -> list[list[tuple[int, int, int]]]:
    """
    与 drawer（12×12 单元、137×137 格）对齐。
    majority_cell=False：每格只取几何中心像素 (6,6)，避免 need_border 的灰色 outline 拉偏整块均值。
    majority_cell=True：在格内子块上二值化（抗 MP4 模糊）；inner_margin>0 时取中心子块（减轻 4:2:0 色度边缘渗透）。
    cell_vote：majority=过半像素>=128；mean=通道均值>=128。
    """
    cell = 12
    grid_n = Config.QRSize + 4
    target = grid_n * cell

    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    if arr.shape[0] != target or arr.shape[1] != target:
        raise ValueError(
            f"内部错误：期望 {target}×{target}，实际 {arr.shape[1]}×{arr.shape[0]}"
        )
    if inner_margin < 0 or inner_margin * 2 >= cell:
        raise ValueError("inner_margin 须满足 0 <= 2*inner_margin < cell")

    if majority_cell:
        sub_side = cell - 2 * inner_margin
        need = (sub_side * sub_side) // 2 + 1
        binary_array = np.zeros((grid_n, grid_n, 3), dtype=np.uint8)
        for r in range(grid_n):
            for c in range(grid_n):
                y0, x0 = r * cell + inner_margin, c * cell + inner_margin
                block = arr[y0 : y0 + sub_side, x0 : x0 + sub_side, :]
                for ch in range(3):
                    plane = block[:, :, ch]
                    if cell_vote == "mean":
                        binary_array[r, c, ch] = 255 if float(np.mean(plane)) >= 128.0 else 0
                    else:
                        binary_array[r, c, ch] = (
                            255 if np.count_nonzero(plane >= 128) >= need else 0
                        )
    else:
        cy = cx = cell // 2
        row_idx = np.arange(grid_n, dtype=np.intp) * cell + cy
        col_idx = np.arange(grid_n, dtype=np.intp) * cell + cx
        sampled = arr[row_idx[:, np.newaxis], col_idx, :]
        binary_array = np.where(sampled >= 128, 255, 0).astype(np.uint8)

    final_size = 137
    final_array = np.full((final_size, final_size, 3), 255, dtype=np.uint8)
    final_array[0:grid_n, 0:grid_n, :] = binary_array
    return [
        [tuple(map(int, final_array[r, c])) for c in range(final_size)]
        for r in range(final_size)
    ]


def image_to_matrix(image_path: str, run_mode="normal") -> list[list[tuple[int, int, int]]]:
    """
    :param run_mode:
        - ``test``：本仓库 PNG / MP4 帧 / 屏录。1644 对齐后按格取 **中心 1 像素** 二值化（抗 drawer 灰边）。
          若仍用 LANCZOS+高斯池化，tribit 会被抹糊，**CRC16 必失败**。
        - ``test_robust``：每格 12×12 内 **多数表决**（有损 MP4）。
        - ``test_robust_inner``：每格中心 8×8 **多数表决**（减轻 H.264 4:2:0 色度在格边界上的串扰）。
        - ``test_robust_mean``：每格 12×12 通道 **均值** ≥128 判白（高斯状模糊时偶发优于多数表决）。
        - ``normal``：未对齐实拍，保留 HSV+池化流水线。
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"读取出错: {e}")
        return []

    if run_mode == "test":
        img_ready = _prepare_square_then_resize_for_grid(img)
        return _matrix_from_exact_drawer_grid(img_ready, majority_cell=False)
    if run_mode == "test_robust":
        img_ready = _prepare_square_then_resize_for_grid(img)
        return _matrix_from_exact_drawer_grid(
            img_ready, majority_cell=True, inner_margin=0, cell_vote="majority"
        )
    if run_mode == "test_robust_inner":
        img_ready = _prepare_square_then_resize_for_grid(img)
        return _matrix_from_exact_drawer_grid(
            img_ready, majority_cell=True, inner_margin=2, cell_vote="majority"
        )
    if run_mode == "test_robust_mean":
        img_ready = _prepare_square_then_resize_for_grid(img)
        return _matrix_from_exact_drawer_grid(
            img_ready, majority_cell=True, inner_margin=0, cell_vote="mean"
        )

    import cv2

    block_num = 133
    win_size = 11
    target_size = block_num * win_size
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)

    current_means = np.mean(img_array, axis=(0, 1))
    gains = 128.0 / (current_means + 1e-6)
    gains = np.clip(gains, 0.1, 5.0)
    img_array[:, :, 0] *= gains[0]
    img_array[:, :, 1] *= gains[1]
    img_array[:, :, 2] *= gains[2]
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.8, 0, 255)
    enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(
        np.float32
    )

    def get_gaussian_kernel(size, sigma=1.5):
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / kernel.sum()

    kernel = get_gaussian_kernel(win_size, sigma=0.5)
    blocks = enhanced_rgb.reshape(
        block_num, win_size, block_num, win_size, 3
    ).transpose(0, 2, 1, 3, 4)
    pooled_array = np.einsum("ijklm,kl->ijm", blocks, kernel)
    binary_array = np.where(pooled_array < 128, 0, 255).astype(np.uint8)

    final_size = 137
    final_array = np.full((final_size, final_size, 3), 255, dtype=np.uint8)
    start = 2
    final_array[start : start + block_num, start : start + block_num, :] = (
        binary_array[:block_num, :block_num]
    )

    return [
        [tuple(map(int, final_array[r, c])) for c in range(final_size)]
        for r in range(final_size)
    ]
