import cv2
import numpy as np
import ctypes
import os
import time
import concurrent.futures
import threading
import queue

# =============================================================================
# [0. 解决中文路径读取/写入问题的核心组件]
# =============================================================================
def imwrite_chinese(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            im_buf_arr.tofile(filename)
            return True
        return False
    except Exception as e:
        print(f"保存图片失败: {e}")
        return False

# =============================================================================
# [1. 动态链接库加载路径]
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
# [全局内存池设置]：锁定物理内存，支持 60fps 稳定传输
# =============================================================================
POOL_SIZE = 300  # 预留 300 帧缓冲区，足以容纳 120 帧全量数据
warp_pool = queue.Queue(maxsize=POOL_SIZE)
decode_pool = queue.Queue(maxsize=POOL_SIZE)
write_queue = queue.Queue()

# =============================================================================
# [2. 并发消费者任务：内存指针零拷贝处理]
# =============================================================================
def decode_in_memory(frame_idx, img_data, perfect_dir, out_size, output_scale, grid_size):
    """
    子线程接管内存指针，进行 C++ 猜色运算
    """
    # 1. 拿取预分配的解码容器
    decode_buffer = decode_pool.get()

    # 2. 压入 C++ Color Engine
    success = kmeans_core.ProcessColorEngine(
        img_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_size, out_size,
        decode_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output_scale, grid_size
    )

    # 3. 释放拉图内存还给池子
    warp_pool.put(img_data)

    if success:
        perfect_path = os.path.join(perfect_dir, f"perfect_{frame_idx:05d}.bmp")
        # 异步落盘，不阻塞猜色算法
        write_queue.put((perfect_path, decode_buffer))
        return True, frame_idx
    else:
        decode_pool.put(decode_buffer)
        return False, frame_idx

# =============================================================================
# [3. 硬盘幽灵写入线程]
# =============================================================================
def async_disk_writer():
    while True:
        task = write_queue.get()
        if task is None: break
        filepath, data = task
        imwrite_chinese(filepath, data)
        # 写入完成后，释放解码内存
        decode_pool.put(data)
        write_queue.task_done()

# =============================================================================
# [4. 120 帧限速流水线控制逻辑]
# =============================================================================
if __name__ == "__main__":
    video_path = 'test1.MOV'
    OUT_SIZE = 1024
    GRID_SIZE = 133
    OUTPUT_SCALE = 6
    FRAME_LIMIT = 60  # 设定 120 帧上限
    
    perfect_dir = os.path.join(current_dir, "最终色彩重建_Perfect")
    os.makedirs(perfect_dir, exist_ok=True)
    
    print(f">>> [60fps 专项流水线] 预分配内存池中...")
    
    for _ in range(POOL_SIZE):
        warp_pool.put(np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8))
        decode_pool.put(np.zeros((GRID_SIZE * OUTPUT_SCALE, GRID_SIZE * OUTPUT_SCALE, 3), dtype=np.uint8))
    
    writer_thread = threading.Thread(target=async_disk_writer, daemon=True)
    writer_thread.start()

    # 28 线程机器，分配 24 个消费者给猜色算法
    max_workers = min(24, os.cpu_count() or 8)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    print(f"[*] 猜色线程池已就绪: {max_workers} 线程")
    print(f">>> 开始拉图（上限 {FRAME_LIMIT} 帧）...")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    t_start = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= FRAME_LIMIT: 
                break
                
            frame_idx += 1
            
            # 无锁拿取内存块
            warp_buf = warp_pool.get()
            
            success = warp_engine.ExtractQRCode(
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 
                frame.shape[1], frame.shape[0], 3,
                warp_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                OUT_SIZE, OUT_SIZE, GRID_SIZE, 2, 10, None
            )
            
            if success:
                # 投递猜色任务，主线程不阻塞继续下一帧
                executor.submit(
                    decode_in_memory, 
                    frame_idx, warp_buf, perfect_dir, OUT_SIZE, OUTPUT_SCALE, GRID_SIZE
                )
            else:
                warp_pool.put(warp_buf)
                
            if frame_idx % 20 == 0:
                print(f"  -> [Warp] 已提取并分发 {frame_idx}/{FRAME_LIMIT} 帧...")

    finally:
        cap.release()
        t_warp_end = time.time()
        print(f"\n>>> [Warp 阶段结束] 耗时: {t_warp_end - t_start:.2f}s")
        print(">>> 正在全力进行猜色算法解码，请稍候...")
        
        # 停止生产后，主线程在此阻塞直到所有猜色任务清空
        executor.shutdown(wait=True)
        
        print(">>> 猜色解码完毕！等待最后磁盘写入...")
        write_queue.put(None)
        writer_thread.join()
        
        t_final = time.time()
        print(f"\n>>> [全线完工] 总处理帧数: {frame_idx}")
        print(f">>> 总耗时: {t_final - t_start:.2f} 秒")
        print(f">>> 平均处理速度: {frame_idx / (t_final - t_start):.1f} fps")