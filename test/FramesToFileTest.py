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

from receiver.decoder.Decode import DecodeFull,GetCorrectionPagesInfo
def Decode(frames: str):
    """
    从帧序列中解码出文件并保存到磁盘
    frames: 帧序列文件夹
    """
    #读取第一帧，获取总页数和raid、rs级别
    total_pages, raid, rs = None, None, None
    first_frame_path = os.path.join(frames, "0.png")
    try:
        total_pages, raid, rs = GetCorrectionPagesInfo(first_frame_path)
    except Exception as e:
        second_frame_path = os.path.join(frames, "1.png")
        if os.path.exists(second_frame_path):
            try:
                total_pages, raid, rs = GetCorrectionPagesInfo(second_frame_path)
            except Exception as e:
                print(f"Error: 无法从前两帧获取总页数和纠错级别。请检查帧文件是否正确。")
                sys.exit(1)

    DecodeFull(total_pages, frames, raid, rs) # type: ignore

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FileToVideoTest.py <frames_folder>")
        print("Example: python FileToVideoTest.py frames")
        sys.exit(1)

    frames_folder = sys.argv[1]
    Decode(frames_folder)
    