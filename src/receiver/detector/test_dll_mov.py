import cv2
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
# [1. 加载 CPU 版 DLL 并更新接口签名]
# =============================================================================
dll_path = os.path.join(current_dir, 'warp_engine.dll')
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"找不到 DLL: {dll_path}，请确保已编译最新版 C++ 引擎。")

# 加载动态库
warp_engine = ctypes.CDLL(dll_path, winmode=0)
# 接口签名必须与 Canvas 中的 11 参数版本严格对齐
# 1-7: 图像基础参数, 8-10: 物理网格参数, 11: 解码指针
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # 1. in_data
    ctypes.c_int,                   # 2. width
    ctypes.c_int,                   # 3. height
    ctypes.c_int,                   # 4. channels
    ctypes.POINTER(ctypes.c_uint8),  # 5. out_data
    ctypes.c_int,                   # 6. out_width
    ctypes.c_int,                   # 7. out_height
    ctypes.c_int,                   # 8. grid_size
    ctypes.c_int,                   # 9. quiet_zone
    ctypes.c_int,                   # 10. large_finder
    ctypes.POINTER(ctypes.c_uint8)   # 11. decoded_data (这就是那个导致 0x02 崩溃的缺失位)
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool

# =============================================================================
# [2. 核心处理封装 零拷贝内存池]
# =============================================================================
def process_frame(frame, pre_allocated_out, out_size=1024, grid_size=283, quiet_zone=2, large_finder=14):
    if frame is None:
        return False
    
    # 强制内存连续，这是 Python 传指针给 C++ 的安全红线
    img_bgr = np.ascontiguousarray(frame)
    h, w, c = img_bgr.shape
    
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = pre_allocated_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    # 调用 11 参数接口，最后一位传 None 代表暂时不接收 137x137 解码矩阵
    return warp_engine.ExtractQRCode(
        in_ptr, w, h, c, 
        out_ptr, out_size, out_size,
        grid_size, quiet_zone, large_finder,
        None 
    )

# =============================================================================
# [3. 主运行程序]
# =============================================================================
if __name__ == "__main__":
    video_path = 'test3.mp4'  
    OUT_SIZE = 1024
    
    # --- 物理参数设定 (对应你的 283+2+14 方案) ---
    GRID_SIZE = 283           
    QUIET_ZONE = 2            
    LARGE_FINDER = 14         
    REPEAT_COUNT = 10         
    
    # 预分配输出缓冲区
    shared_out_buffer = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    
    print(f">>> 物理参数驱动引擎启动 | 数据源: {video_path}")
    print(f"[*] 架构配置: Grid={GRID_SIZE}, QuietZone={QUIET_ZONE}, Finder={LARGE_FINDER}")
    
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
                
                # 执行处理
                success = process_frame(
                    frame, shared_out_buffer, 
                    out_size=OUT_SIZE, 
                    grid_size=GRID_SIZE, 
                    quiet_zone=QUIET_ZONE, 
                    large_finder=LARGE_FINDER
                )
                
                # 显示原图监控
                cv2.imshow("Input Stream", cv2.resize(frame, (800, 600)))        
                
                # 显示拉伸结果
                if success:
                    cv2.imshow(f"Physical Warp Output ({GRID_SIZE})", shared_out_buffer)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(">>> 用户强制退出。")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                    
            cap.release()
            print(f"[*] 第 {loop_idx}/{REPEAT_COUNT} 遍循环完成 ({frame_in_loop} 帧)")
            
    finally:
        cv2.destroyAllWindows()
        print("\n>>> 全案测试结束。")