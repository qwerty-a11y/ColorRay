import cv2
import numpy as np
import ctypes
import os
import time

# 1. 加载 DLL
# 确保 warp_engine.dll 与此脚本在同一目录下
dll_path = os.path.join(os.path.dirname(__file__), 'warp_engine.dll')
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"找不到 DLL: {dll_path}")

warp_engine = ctypes.CDLL(dll_path)

# 2. 定义 C++ 函数的签名
# 必须与 Canvas 中的 extern "C" 函数参数严格对应
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # in_data (原图指针)
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.c_int,                    # channels
    ctypes.POINTER(ctypes.c_uint8),  # out_data (输出缓冲区指针)
    ctypes.c_int,                    # out_width
    ctypes.c_int                     # out_height
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool

def process_frame(img_bgr, out_size=800):
    """
    调用 C++ DLL 处理图像，提取并校正二维码区域
    """
    # 【核心安全】强制内存连续，防止 C++ 访问非法地址导致崩溃
    img_bgr = np.ascontiguousarray(img_bgr)
    h, w, c = img_bgr.shape
    
    # 预分配输出内存 (默认 800x800)
    out_img = np.zeros((out_size, out_size, c), dtype=np.uint8)

    # 获取底层 C 指针
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    # 执行 C++ 逻辑 (C++ 会直接在 in_ptr 原图内存上画绿圈)
    success = warp_engine.ExtractQRCode(in_ptr, w, h, c, out_ptr, out_size, out_size)
    
    return out_img if success else None

if __name__ == "__main__":
    # =========================================================================
    # 【视频源设置】
    # 填入你实际的视频文件路径，比如 'test_video.mp4'
    # 如果你想直接调用电脑摄像头，把这里改成数字 0
    video_source = 'test.MOV'  
    # =========================================================================
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"无法打开视频或摄像头: {video_source}")
        exit()

    print("开始调用 C++ DLL 引擎处理视频流...")
    print(">>> 提示: 在弹出的窗口中按下键盘上的 'q' 键或 'ESC' 键即可退出程序 <<<")
    
    frame_count = 0
    
    # 开始死循环读取视频帧
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果 ret 为 False，说明视频播完了，或者摄像头断开了
        if not ret:
            print("\n视频流结束或无法获取画面。")
            break
            
        frame_count += 1
        
        # 将当前帧丢进 C++ 引擎处理
        result = process_frame(frame, out_size=1024)

        # 1. 显示带有绿色标记的原图（C++ 已直接在内存中画好标记）
        # 4K图太大，屏幕放不下，我们在 Python 端把它缩放到 800x600 的小窗口展示
        monitor_size = (800, 600)
        cv2.imshow("Python Monitor: Original with Markers", cv2.resize(frame, monitor_size))
        
        # 2. 显示拉伸后的结果
        # 因为我们预先分配了 out_img 的全黑矩阵，如果C++没找到4个点，返回的可能是一张纯黑图
        if result is not None:
            cv2.imshow("Python Monitor: Warped Result", result)

        # 【核心操作：1ms 刷新机制】
        # waitKey(1) 意味着给底层 GUI 库 1 毫秒的时间去刷新画面，这样图像才会动起来。
        # 同时监控键盘输入，如果按下 'q' 键 (ASCII: 113) 或 ESC 键 (ASCII: 27) 则退出。
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n用户手动终止处理。")
            break

    # 善后工作：释放视频句柄，销毁所有 OpenCV 窗口
    cap.release()
    cv2.destroyAllWindows()
    print(f"处理完成，共处理了 {frame_count} 帧。")