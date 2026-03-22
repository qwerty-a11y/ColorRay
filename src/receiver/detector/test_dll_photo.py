import cv2
import numpy as np
import ctypes
import os

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

    # 执行 C++ 逻辑
    success = warp_engine.ExtractQRCode(in_ptr, w, h, c, out_ptr, out_size, out_size)
    
    # =====================================================================
    # 【调试拦截】Python 侧的强制监控窗口（默认注释）
    # 如果你想看 C++ 在返回 false 之前到底把图拉伸成了什么样，请取消下方注释。
    # 这对应 Canvas 中 C++ 的调试思路。
    
    cv2.imshow("Debug: Python Force Warped", out_img)
    cv2.waitKey(1)
    
    # =====================================================================
    
    return out_img if success else None

if __name__ == "__main__":
    # 替换为你实际的测试图片路径
    img_path = 'test4.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        exit()

    print("正在调用 C++ DLL 引擎进行特征提取与透视校正...")
    
    # 执行处理逻辑
    result = process_frame(img, out_size=800)

    # 1. 显示带有绿色标记的原图（C++ 已直接在内存中画好标记）
    # 缩放到合适大小展示
    monitor_size = (800, 600)
    cv2.imshow("Python Monitor: Original with Markers", cv2.resize(img, monitor_size))
    
    # 2. 显示拉伸后的结果
    if result is not None:
        cv2.imshow("Python Monitor: Warped Result", result)
        print("识别成功：已获取校正后的图像。")
    else:
        print("识别失败：C++ 引擎未找齐定位块。请检查原图窗口中的绿色标记。")

    # 等待按键退出
    print("按下任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()