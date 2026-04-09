import asyncio
import sys
import os

# 获取当前脚本所在目录 (test)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (ColorRay)
project_root = os.path.dirname(current_dir)
# 构建 src 目录的绝对路径
src_path = os.path.join(project_root, 'src')

# 将 src 目录添加到系统路径，以便导入 common 等模块
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from receiver.decoder.Decode import DecodeFull

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FileToVideoTest.py <video_file>")
        print("Example: python FileToVideoTest.py video.mp4")
        sys.exit(1)

    video_file = sys.argv[1]

    import cProfile
    cProfile.run(f"asyncio.run(DecodeFull('{video_file}'))")
    #asyncio.run(DecodeFull(video_file))
    