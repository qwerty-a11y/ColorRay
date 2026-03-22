﻿import cv2
import numpy as np
import ctypes
import os
import time
# =============================================================================
# [0. DLL 依赖路径配置]
# =============================================================================
current_dir = os.path.abspath(os.path.dirname(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)
# =============================================================================
# [1. 加载 CPU 版 DLL]
# =============================================================================
dll_path = os.path.join(current_dir, 'warp_engine.dll')
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"找不到 CPU 版 DLL: {dll_path}，请确保已通过 CPU 任务编译。")
# 加载动态库，winmode=0 确保能加载当前目录下的依赖项 (如 opencv_worldxxx.dll)
warp_engine = ctypes.CDLL(dll_path, winmode=0)
# 绑定接口签名
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # in_data
    ctypes.c_int,                   # width
    ctypes.c_int,                   # height
    ctypes.c_int,                   # channels
    ctypes.POINTER(ctypes.c_uint8),  # out_data
    ctypes.c_int,                   # out_width
    ctypes.c_int                    # out_height
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool
# =============================================================================
# [2. 核心处理封装 (零拷贝内存池版)]
# =============================================================================
def process_frame(frame, pre_allocated_out, out_size=1024):
    if frame is None:
        return False
    # 强制内存连续，这是 Python 传指针给 C++ 的安全红线
    img_bgr = np.ascontiguousarray(frame)
    h, w, c = img_bgr.shape
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = pre_allocated_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    # 直接在传入的 pre_allocated_out 内存上操作
    return warp_engine.ExtractQRCode(in_ptr, w, h, c, out_ptr, out_size, out_size)
# =============================================================================
# [3. 主运行程序：10次大循环性能测试]
# =============================================================================
if __name__ == "__main__":
    video_path = 'test4.MOV'  
    OUT_SIZE = 1024
    REPEAT_COUNT = 10         # 循环遍数
    # 预分配输出缓冲区 (内存池)
    shared_out_buffer = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    print(f">>> CPU 组性能压力测试启动 | 数据源: {video_path} | 循环次数: {REPEAT_COUNT}")
    try:
        for loop_idx in range(1, REPEAT_COUNT + 1):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误: 无法打开视频 {video_path}")
                break
            frame_in_loop = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_in_loop += 1
                success = process_frame(frame, shared_out_buffer, out_size=OUT_SIZE)
                monitor_size = (800, 600)
                cv2.imshow("Input Stream (CPU Group)", cv2.resize(frame, monitor_size))        
                if success:
                    cv2.imshow("Processed Output (8-Color)", shared_out_buffer)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(">>> 用户强制退出。")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
            cap.release()
            print(f"[*] 第 {loop_idx}/{REPEAT_COUNT} 遍循环处理完成 ({frame_in_loop} 帧)")
    finally:
        cv2.destroyAllWindows()
        print("\n>>> 全案 10 遍测试结束，程序已安全退出。")