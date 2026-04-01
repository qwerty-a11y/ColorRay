import cv2
import numpy as np
import ctypes
import os

def add_dll_search_path(path):
    """将目录添加到当前进程的 PATH 环境变量中"""
    if path not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] = path + os.pathsep + os.environ['PATH']

# 你的路径
current_dir = os.path.abspath(os.path.dirname(__file__))
opencv_bin_path = r"H:\OpenCV\opencv\build\x64\vc16\bin"

if os.path.exists(opencv_bin_path):
    add_dll_search_path(opencv_bin_path)
add_dll_search_path(current_dir)
dll_candidates = ['warp_engine.dll', 'warp_engine_cpu.dll', 'warp_engine_cuda.dll']
warp_engine = None
for dll_name in dll_candidates:
    path = os.path.join(current_dir, dll_name)
    if os.path.exists(path):
        try:
            warp_engine = ctypes.CDLL(path, winmode=0)
            print(f">>> 成功加载引擎: {dll_name}")
            break
        except OSError as e:
            print(f"尝试加载 {dll_name} 失败: {e}")
            # 在Windows上，可以进一步获取详细的错误码
            if hasattr(e, 'winerror'):
                print(f"Windows 错误码: {e.winerror}")
            # 打印异常的所有参数，可能包含更多细节
            print(f"异常详情: {e.args}")
if warp_engine is None:
    raise FileNotFoundError("在当前目录下未找到任何有效的 warp_engine DLL 文件。")
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # in_data (原图指针)
    ctypes.c_int,                   # width
    ctypes.c_int,                   # height
    ctypes.c_int,                   # channels
    ctypes.POINTER(ctypes.c_uint8),  # out_data (输出缓冲区指针)
    ctypes.c_int,                   # out_width
    ctypes.c_int                    # out_height
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool
def process_photo(img_path, out_size=800):
    """
    读取单张图片，调用 C++ 引擎进行校正
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像 {img_path}")
        return None
    img_bgr = np.ascontiguousarray(img)
    h, w, c = img_bgr.shape
    out_img = np.zeros((out_size, out_size, c), dtype=np.uint8)
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    success = warp_engine.ExtractQRCode(in_ptr, w, h, c, out_ptr, out_size, out_size) # type: ignore
    return out_img if success else None
