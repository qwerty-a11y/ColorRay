import asyncio
from typing import AsyncIterator

from PIL import Image
import ffmpeg


async def async_pil_images_to_lossless_video(
    images: AsyncIterator[Image.Image],
    output_path: str,
    fps: float = 30.0,
    codec: str = "libx264",
    pix_fmt_in: str = "rgb24",
    preset: str = "ultrafast"
) -> bool:
    """
    纯异步版本：使用 asyncio.subprocess 将内存中的 Pillow 图像流式编码为无损视频。
    无任何线程池阻塞，stdin 关闭可靠。
    """
    img_iter = images

    # 获取第一帧以确定尺寸
    try:
        first_img = await img_iter.__anext__()
    except StopAsyncIteration:
        print("错误：图像序列为空")
        return False

    # 统一图像模式
    if pix_fmt_in == "rgb24" and first_img.mode != "RGB":
        first_img = first_img.convert("RGB")
    elif pix_fmt_in == "gray" and first_img.mode != "L":
        first_img = first_img.convert("L")

    width, height = first_img.size

    # 重建包含第一帧的异步生成器
    async def restored_generator():
        yield first_img
        async for img in img_iter:
            yield img

    # ---------- 构建 FFmpeg 命令行参数 ----------
    input_args = [
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt_in,
        "-s", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0"               # 从 stdin 读取
    ]

    if codec == "libx264":
        output_args = [
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", preset,
            "-pix_fmt", "yuv444p",
            "-bf", "0"
        ]
    elif codec == "ffv1":
        if not output_path.lower().endswith(('.mkv', '.avi')):
            print("警告：FFV1 在 MP4 中兼容性差，建议使用 .mkv")
        output_args = [
            "-c:v", "ffv1",
            "-level", "3",
            "-coder", "1",
            "-context", "1",
            "-slices", "24",
            "-pix_fmt", "gbrp" if pix_fmt_in == "rgb24" else "gray"
        ]
    else:
        raise ValueError(f"不支持的编码器: {codec}")

    cmd = ["ffmpeg", "-y"] + input_args + output_args + [output_path]

    # ---------- 启动异步子进程 ----------
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        # 逐帧写入管道
        async for img in restored_generator():
            if pix_fmt_in == "rgb24" and img.mode != "RGB":
                img = img.convert("RGB")
            elif pix_fmt_in == "gray" and img.mode != "L":
                img = img.convert("L")

            img_bytes = img.tobytes()
            process.stdin.write(img_bytes)
            # 关键：等待管道排空，避免数据堆积
            await process.stdin.drain()

        # 所有帧写入完毕，关闭 stdin 发送 EOF
        process.stdin.close()
        await process.stdin.wait_closed()

        # 等待 FFmpeg 进程结束
        returncode = await process.wait()

        if returncode != 0:
            stderr = await process.stderr.read()
            print(f"FFmpeg 错误 (返回码 {returncode}):\n{stderr.decode()}")
            return False

        print(f"无损视频已生成: {output_path}")
        return True

    except Exception as e:
        print(f"编码过程中发生异常: {e}")
        process.stdin.close()
        await process.wait()
        return False