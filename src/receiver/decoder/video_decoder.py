import asyncio
import os
import sys
from PIL import Image
from typing import AsyncIterator, Union

import numpy

# 如需 numpy/OpenCV 支持，请取消注释：
# import numpy as np
# import cv2
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

async def probe_video_dimensions(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(cmd)
        raise RuntimeError(f"ffprobe 失败: {stderr.decode().strip()}")
    try:
        # 去除首尾空白及可能的尾随逗号
        output = stdout.decode().strip().rstrip(',')
        w, h = map(int, output.split(','))
        return w, h
    except Exception:
        raise RuntimeError(f"无法解析视频尺寸: {stdout.decode().strip()}")

async def stream_frames_rgb24(
    video_path: str,
    output_format: str = "pil"   # "pil" 或 "numpy"
) -> AsyncIterator[Union[Image.Image, "numpy.ndarray"]]:
    """
    异步迭代器：从任意视频文件中流式读取帧，统一转换为 8-bit RGB。

    Args:
        video_path: 输入视频路径（支持所有 ffmpeg 可解码格式）
        output_format:
            - "pil" : 返回 PIL.Image 对象（RGB 模式，8-bit）
            - "numpy": 返回 numpy 数组（BGR 格式，兼容 OpenCV）

    Yields:
        PIL.Image 或 numpy.ndarray

    Raises:
        RuntimeError: 视频尺寸探测失败或 FFmpeg 执行错误
    """
    # 1. 自动获取视频尺寸
    width, height = await probe_video_dimensions(video_path)

    # 2. 构建 FFmpeg 命令：解码 → 强制输出 8-bit RGB rawvideo 到 stdout
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,           # 输入文件
        "-f", "rawvideo",           # 输出原始视频流
        "-pix_fmt", "rgb24",        # 强制 8-bit RGB，每通道 8 位
        "-vcodec", "rawvideo",      # 不进行二次编码
        "-an",                      # 忽略音频
        "-sn",                      # 忽略字幕
        "-nostats",                 # 减少控制台输出
        "-loglevel", "error",       # 只输出错误信息
        "pipe:1"                    # 输出到标准输出
    ]

    # 3. 启动异步子进程
    process = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    frame_size = width * height * 3  # RGB24 每像素 3 字节

    try:
        while True:
            # 精确读取一帧的原始字节
            data = await process.stdout.readexactly(frame_size) # type: ignore
            # 将字节流转换为 PIL Image
            # 修复: Image.frombytes 的 size 参数应为 (width, height)
            img = Image.frombytes("RGB", (height, width), data)

            if output_format == "pil":
                yield img
            elif output_format == "numpy":
                import numpy as np
                import cv2
                # 转为 numpy 数组 (RGB)
                # 此时 img.size 为 (width, height)，np.array(img) 形状为 (height, width, 3)
                rgb_array = np.array(img)
                # 转为 BGR 供 OpenCV 使用
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                yield bgr_array
            else:
                raise ValueError(f"不支持的 output_format: {output_format}")

    except asyncio.IncompleteReadError:
        # 正常流结束（读取到的数据不足一帧）
        pass
    except Exception as e:
        # 其他异常，尝试读取 stderr 获取 FFmpeg 错误信息
        stderr_data = await process.stderr.read() # type: ignore
        if stderr_data:
            print(f"FFmpeg 错误输出:\n{stderr_data.decode()}")
        raise e
    finally:
        # 确保子进程被正确终止
        process.terminate()
        await process.wait()
