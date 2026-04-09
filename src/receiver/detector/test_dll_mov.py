import cv2
import numpy as np
import ctypes
import os
import time
from sklearn.cluster import KMeans

# =============================================================================
# [0. DLL 依赖路径配置]
# =============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)

# =============================================================================
# [1. 加载 CPU 版 DLL 并更新接口签名]
# =============================================================================
dll_path = os.path.join(current_dir, 'warp_engine.dll')
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"找不到 DLL: {dll_path}，请确保已编译最新版 C++ 引擎。")

warp_engine = ctypes.CDLL(dll_path, winmode=0)
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), 
    ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_uint8), 
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_uint8) 
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool

# =============================================================================
# [2. 核心处理封装]
# =============================================================================
def process_frame(frame, pre_allocated_out, out_size=1024, grid_size=133, quiet_zone=2, large_finder=10):
    if frame is None:
        return False
    
    img_bgr = np.ascontiguousarray(frame)
    h, w, c = img_bgr.shape
    
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = pre_allocated_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    return warp_engine.ExtractQRCode(
        in_ptr, w, h, c, 
        out_ptr, out_size, out_size,
        grid_size, quiet_zone, large_finder,
        None 
    )

def fast_decode_and_reconstruct(img_bgr, grid_size=133):
    """
    利用向量化操作和K-Means，从存在微小形变的图像中提取完美的 0/255 标准网格。
    """
    h, w, _ = img_bgr.shape
    cell_h, cell_w = h / grid_size, w / grid_size
    # 搜索半径：在理论中心周围 35% 范围内寻找最纯净像素
    search_r = max(1, int(min(cell_w, cell_h) * 0.35))

    # 1. 极速计算全图局部方差矩阵
    img_float = img_bgr.astype(np.float32)
    img_sq = img_float ** 2
    E_x = cv2.blur(img_float, (3, 3))
    E_x2 = cv2.blur(img_sq, (3, 3))
    var_img = np.sum(E_x2 - E_x**2, axis=2)

    sampled_colors = np.zeros((grid_size * grid_size, 3), dtype=np.float32)

    # 2. 网格游走采样
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            cx = int(col * cell_w + cell_w / 2)
            cy = int(row * cell_h + cell_h / 2)
            cx, cy = np.clip(cx, search_r, w - search_r - 1), np.clip(cy, search_r, h - search_r - 1)

            # 寻找局部方差最小点
            roi_var = var_img[cy-search_r : cy+search_r+1, cx-search_r : cx+search_r+1]
            min_idx = np.argmin(roi_var)
            dy, dx = np.unravel_index(min_idx, roi_var.shape)

            # 采样平滑后的颜色
            sampled_colors[idx] = E_x[cy - search_r + dy, cx - search_r + dx]
            idx += 1

    # 3. K-Means 聚类还原 8 种标准色
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=3).fit(sampled_colors)
    standard_centers = np.where(kmeans.cluster_centers_ > 127, 255, 0).astype(np.uint8)

    # 4. 极速重构图像 (Kronecker 积)
    output_scale = 6 
    labels_2d = kmeans.labels_.reshape((grid_size, grid_size))
    color_map = standard_centers[labels_2d]
    reconstructed_img = np.kron(color_map, np.ones((output_scale, output_scale, 1), dtype=np.uint8))

    return reconstructed_img

# =============================================================================
# [3. 主运行程序]
# =============================================================================
if __name__ == "__main__":
    video_path = 'test1.MOV'  
    OUT_SIZE = 1024
    GRID_SIZE = 133
    
    shared_out_buffer = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    
    print(f">>> 自动解码引擎启动 | 数据源: {video_path}")
    print(f"[*] 当前配置: Grid={GRID_SIZE} | 解码状态: 始终开启")

    try:
        for loop_idx in range(1, 11):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): break
                
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # 1. 调用 DLL 进行全局拉平
                success = process_frame(frame, shared_out_buffer, out_size=OUT_SIZE, grid_size=GRID_SIZE)
                
                if success:
                    # 2. 自动进行局部纠偏与 K-Means 解码
                    # 如果觉得卡顿，可以每隔 2 帧处理一次
                    clean_grid = fast_decode_and_reconstruct(shared_out_buffer, grid_size=GRID_SIZE)
                    
                    # 显示结果
                    cv2.imshow("Warped Output", cv2.resize(shared_out_buffer, (512, 512)))
                    cv2.imshow("Perfect Decoded Grid", clean_grid)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                    
            cap.release()
            print(f"[*] 第 {loop_idx} 遍循环完成")
            
    finally:
        cv2.destroyAllWindows()