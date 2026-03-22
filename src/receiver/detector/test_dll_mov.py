﻿import cv2
import numpy as np
import ctypes
import os

# =============================================================================
# [1. 加载 DLL 并定义接口]
# =============================================================================
dll_path = os.path.join(os.path.dirname(__file__), 'warp_engine.dll')
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"找不到 DLL: {dll_path}，请先确保 C++ 代码已成功编译。")

warp_engine = ctypes.CDLL(dll_path)

warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # in_data
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.c_int,                    # channels
    ctypes.POINTER(ctypes.c_uint8),  # out_data
    ctypes.c_int,                    # out_width
    ctypes.c_int                     # out_height
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool

# =============================================================================
# [2. 核心处理封装 (零拷贝优化版)]
# =============================================================================
def process_frame(frame, pre_allocated_out, out_size=1024):
    """
    接收预先分配好的 out_img 内存块，直接将 C++ 的结果覆盖写入
    """
    if frame is None:
        return False

    img_bgr = np.ascontiguousarray(frame)
    h, w, c = img_bgr.shape
    
    # 获取输入和预分配输出的底层指针
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = pre_allocated_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    # C++ 直接在 pre_allocated_out 这块内存上进行写操作
    success = warp_engine.ExtractQRCode(in_ptr, w, h, c, out_ptr, out_size, out_size)
    
    return success

# =============================================================================
# [3. 主运行程序]
# =============================================================================
if __name__ == "__main__":
    video_path = 'test3.MOV' 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {video_path}")
        exit()

    print(">>> C++ 引擎已就绪 (OpenMP 并行 + Python 零拷贝内存池模式)")
    
    # -------------------------------------------------------------------------
    # 【极致性能优化：内存池技术】
    # 在循环外一次性申请好输出图像的内存，避免每帧重复申请导致的内存抖动和 GC 耗时
    # -------------------------------------------------------------------------
    OUT_SIZE = 1024
    shared_out_buffer = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 传入预分配的 buffer，函数只返回是否成功 (True/False)
            success = process_frame(frame, shared_out_buffer, out_size=OUT_SIZE)

            monitor_size = (800, 600)
            cv2.imshow("Input Stream", cv2.resize(frame, monitor_size))
            
            # 如果成功，shared_out_buffer 里的像素已经被 C++ 悄悄更新了，直接显示即可
            if success:
                cv2.imshow("Warped Result (High-Res)", shared_out_buffer)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(">>> 程序已安全退出。")