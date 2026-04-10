import cv2
import numpy as np
import ctypes
import os
import time
import concurrent.futures
import queue

# =============================================================================
# [底层引擎加载与接口声明]
# =============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)

warp_dll_path = os.path.join(current_dir, 'warp_engine.dll')
if not os.path.exists(warp_dll_path):
    raise FileNotFoundError(f"找不到 Warp 引擎: {warp_dll_path}")
warp_engine = ctypes.CDLL(warp_dll_path, winmode=0)
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, 
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool

color_dll_path = os.path.join(current_dir, 'kmeans_core.dll')
if not os.path.exists(color_dll_path):
    raise FileNotFoundError(f"找不到 KMeans 引擎: {color_dll_path}")
kmeans_core = ctypes.CDLL(color_dll_path, winmode=0)
kmeans_core.ProcessColorEngine.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int  
]
kmeans_core.ProcessColorEngine.restype = ctypes.c_bool

# =============================================================================
# [核心逻辑层]
# =============================================================================
def color_decode_worker(frame_idx: int, warp_img: np.ndarray, out_size: int, 
                        output_scale: int, grid_size: int, 
                        warp_pool: queue.Queue, decode_pool: queue.Queue) -> tuple:
    """
    消费者核心接口：对拉平后的单张图像进行色彩聚类解码，完全在内存中流转
    """
    decode_buf = decode_pool.get()
    success = kmeans_core.ProcessColorEngine(
        warp_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_size, out_size,
        decode_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output_scale, grid_size
    )
    # 用完即刻归还源图像内存至拉图引擎
    warp_pool.put(warp_img)
    result_img = None
    if success:
        # 提取有效数据作为标准 ndarray 输出
        result_img = decode_buf.copy()
    # 释放解码缓冲区
    decode_pool.put(decode_buf)

    return frame_idx, result_img

def process_video_pipeline(video_path: str, frame_limit: int = 60, 
                           out_size: int = 1024, grid_size: int = 133, 
                           output_scale: int = 6) -> list:
    """
    视频处理流水线：调度多线程进行视频流提取与解码
    返回:
        list[tuple[int, np.ndarray]]: 包含 (帧序号, 解码后的标准图像数组) 的列表
    """
    pool_size = 150
    warp_pool = queue.Queue(maxsize=pool_size)
    decode_pool = queue.Queue(maxsize=pool_size)
    for _ in range(pool_size):
        warp_pool.put(np.zeros((out_size, out_size, 3), dtype=np.uint8))
        decode_pool.put(np.zeros((grid_size * output_scale, grid_size * output_scale, 3), dtype=np.uint8))
    max_workers = min(24, os.cpu_count() or 8)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] 错误：无法读取视频文件 {video_path}")
        return []
    frame_idx = 0
    futures = []
    # 启用线程池并发执行消费者任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= frame_limit: 
                break
                
            frame_idx += 1
            # 生产者：拿取预分配内存并调用 Warp 引擎
            warp_buf = warp_pool.get()
            success = warp_engine.ExtractQRCode(
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 
                frame.shape[1], frame.shape[0], 3,
                warp_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                out_size, out_size, grid_size, 2, 10, None
            )
            if success:
                # 投递给消费者进行解码
                future = executor.submit(
                    color_decode_worker, 
                    frame_idx, warp_buf, out_size, output_scale, grid_size, 
                    warp_pool, decode_pool
                )
                futures.append(future)
            else:
                warp_pool.put(warp_buf)
        cap.release()
        # 收集标准化输出结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            f_idx, res_img = future.result()
            if res_img is not None:
                results.append((f_idx, res_img))
    # 按帧序列号重排保证时序
    results.sort(key=lambda x: x[0])
    return results
# =============================================================================
# [调用示例：数据交接点]
# =============================================================================
if __name__ == "__main__":
    #[输入接口]，输入视频
    video_input_path = 'test1.MOV'
    frame_limit = 60
    
    print(">>> 启动系统调度流水线 (纯内存无 I/O 模式)...")
    t_start = time.time()
    
    # [输出接口]: 组长需要的干净数据全在这里
    final_results = process_video_pipeline(video_input_path, frame_limit)
    
    t_final = time.time()
    print(f">>> [流水线完工] 成功解码 {len(final_results)} 帧, 耗时: {t_final - t_start:.2f} 秒")
    
    if final_results:
        print("\n>>> 数据流已就绪！可以直接传递给下一个模块。")
        # 示例：展示拿到数据的形态，绝不弹窗、绝不存盘
        sample_frame_idx, sample_img_data = final_results[0]
        print(f"    第一帧数据形态: {type(sample_img_data)}, 尺寸: {sample_img_data.shape}")