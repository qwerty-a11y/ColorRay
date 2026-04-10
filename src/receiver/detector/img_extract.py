import sys

import cv2
import numpy as np
import ctypes
import os

from common import Config
def base_path():
    """获取资源文件的绝对路径，兼容开发和 PyInstaller 打包"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return base_path
def add_dll_search_path(path):
    """将目录添加到当前进程的 PATH 环境变量中"""
    if path not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] = path + os.pathsep + os.environ['PATH']

# 你的路径
current_dir = base_path()

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
def process_photo(img: np.ndarray, out_size=2048, grid_size=Config.QRSize, quiet_zone=2, large_finder=14):
    """
    读取单张图片，调用 C++ 引擎进行校正
    """
    if img is None:
        print(f"错误：无法读取图像")
        return None
    img_bgr = np.ascontiguousarray(img)
    h, w, c = img_bgr.shape
    out_img = np.zeros((out_size, out_size, c), dtype=np.uint8)
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    success = warp_engine.ExtractQRCode(in_ptr, w, h, c, out_ptr, out_size, out_size, # type: ignore
        grid_size, 0, large_finder,
        None
    )

    return out_img if success else None


if __name__ == "__main__":
    test_image_path = os.path.join(current_dir, "test_photo.jpg")
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
    else:
        result = process_photo(cv2.imread(test_image_path))
        if result is not None:
            print(">>> 图片处理成功！")
        else:
            print(">>> 图片处理失败！")