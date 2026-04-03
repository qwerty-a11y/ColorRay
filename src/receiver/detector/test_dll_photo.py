import cv2
import numpy as np
import ctypes
import os

# =============================================================================
# [0. 环境配置：解决 Python 3.8+ 依赖加载问题]
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

# 接口签名必须与 C++ 最新 11 参数物理版完全对齐
# 1-7: 图像基础参数, 8: grid_size, 9: quiet_zone, 10: large_finder, 11: decoded_data
warp_engine.ExtractQRCode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # 1. in_data
    ctypes.c_int,                   # 2. width
    ctypes.c_int,                   # 3. height
    ctypes.c_int,                   # 4. channels
    ctypes.POINTER(ctypes.c_uint8),  # 5. out_data
    ctypes.c_int,                   # 6. out_width
    ctypes.c_int,                   # 7. out_height
    ctypes.c_int,                   # 8. grid_size (如 283)
    ctypes.c_int,                   # 9. quiet_zone (如 2)
    ctypes.c_int,                   # 10. large_finder (如 14)
    ctypes.POINTER(ctypes.c_uint8)   # 11. decoded_data (此处传 None 代表仅测试矫正)
]
warp_engine.ExtractQRCode.restype = ctypes.c_bool
# =============================================================================
# [2. 核心处理封装]
# =============================================================================
def process_photo(img_path, out_size=2048, grid_size=283, quiet_zone=2, large_finder=14):
    """
    读取单张图片，调用 C++ 引擎进行自适应校正
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像 {img_path}")
        return None
    # 【核心安全】强制内存连续
    img_bgr = np.ascontiguousarray(img)
    h, w, c = img_bgr.shape
    # 预分配输出内存
    out_img = np.zeros((out_size, out_size, c), dtype=np.uint8)
    # 获取指针
    in_ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    # 执行 C++ 逻辑
    # 比例计算现在由 C++ 内部根据物理参数处理
    success = warp_engine.ExtractQRCode(
        in_ptr, w, h, c, out_ptr, out_size, out_size,
        grid_size, quiet_zone, large_finder,
        None # 单图测试暂时不需要返回 137x137 解码矩阵
    )
    # 显示结果
    if success:
        monitor_size = (800, 800)
        cv2.imshow("Original Monitor", cv2.resize(img, (800, 600)))
        # 展示校正后的 2048 高清图（缩小到 1024 展示）
        cv2.imshow(f"Physical Warp Output (Grid: {grid_size})", cv2.resize(out_img, (1024, 1024)))
    
    return out_img if success else None

# =============================================================================
# [3. 运行测试]
# =============================================================================
if __name__ == "__main__":
    # 替换为你想要测试的图片文件名
    test_image = 'test9.png'
    
    # --- 物理参数设定 (对应你的 283+2+14 方案) ---
    MY_GRID_SIZE = 283     #[二维码数据区尺寸]      
    MY_QUIET_ZONE = 2      #[全图外圈白边尺寸]      
    MY_LARGE_FINDER = 14  #[大定位块尺寸，不包含白边]
    print(f"正在处理图片: {test_image} ...")
    print(f"[*] 物理配置: Grid={MY_GRID_SIZE}, Quiet={MY_QUIET_ZONE}, Finder={MY_LARGE_FINDER}")
    
    # out_size: 建议对于 283 这种高密度阵列，拉伸到 2048x2048 保证细节充足
    result = process_photo(
        test_image, 
        out_size=2048, 
        grid_size=MY_GRID_SIZE,
        quiet_zone=MY_QUIET_ZONE,
        large_finder=MY_LARGE_FINDER
    )

    if result is not None:
        print(">>> 物理映射校正成功！")
        cv2.imwrite("result_output.png", result)
        print("已保存高清校正后的图片至: result_output.png")
    else:
        print(">>> 识别失败。请确认 C++ 引擎是否找齐了 4 个圆心。")

    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()