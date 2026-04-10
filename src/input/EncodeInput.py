import asyncio
import sys
import os

from common import Config
from common.CorrectionLevel import RSLevel, RaidLevel
from sender.encoder.Encode import Encode, GroupToFrames
from sender.generator.drawer import drawer, mem_drawer
from sender.generator.frame_gen import generate_frame
from sender.generator.bytes_to_colors import bytes_to_colors
from sender.generator.colors_to_matrix import colors_to_matrix
from sender.generator.video_generator import async_pil_images_to_lossless_video

async def async_generator(frames: list[bytes], raid: RaidLevel, rs: RSLevel):
    pages = len(frames)
    for i in range(pages):
        colors = bytes_to_colors(frames[i])
        frame, need_border = generate_frame(i, pages, raid, rs)
        matrix = colors_to_matrix(frame, colors) # type: ignore
        yield mem_drawer(matrix, need_border)

def EncodeFull(path:str, raid:RaidLevel, rs:RSLevel):
    data_groups = Encode(path, raid, rs)
    frames = GroupToFrames(data_groups) # type: ignore
    generator = async_generator(frames, raid, rs)
    match raid:
        case RaidLevel.LEVEL1_10:
            raid_text = "10%"
        case RaidLevel.LEVEL2_20:
            raid_text = "20%"
        case RaidLevel.LEVEL3_40:
            raid_text = "40%"
        case RaidLevel.NONE:
            raid_text = "0%"
    match rs:
        case RSLevel.LEVEL1_5: 
            rs_text = "5%"
        case RSLevel.LEVEL2_10: 
            rs_text = "10%"
        case RSLevel.LEVEL3_15: 
            rs_text = "15%"
        case RSLevel.NONE:
            rs_text = "0%"
    print(f"开始编码: 共{len(frames)}页, raid纠错率{raid_text}, rs纠错率{rs_text}")
    output_video_path = os.path.join("video", "output_video.mp4")
    if not os.path.exists("video"):
        os.makedirs("video", exist_ok=True)
    asyncio.run(async_pil_images_to_lossless_video(generator, output_video_path, fps=Config.VideoFrameRate, codec="libx264", pix_fmt_in="rgb24", preset="ultrafast"))


def EncodeInput(raid_val:int, rs_val:int, file_path:str):
    #print("Usage: python FileToVideoTest.py <raid_level_int> <rs_level_int> <file_path>")
    #print("Example: python FileToVideoTest.py 0 0 input.txt")
    #sys.exit(1)

    # 获取参数
    #raid_val = int(sys.argv[1])
    #rs_val = int(sys.argv[2])
    #file_path = sys.argv[3]

    # 将整数转换为对应的枚举类型 (假设枚举值是从 0 开始的连续整数，或者根据实际枚举定义调整)
    # 这里使用 list(RaidLevel)[raid_val] 的方式尝试获取，如果枚举定义不同请手动映射
    try:
        match(raid_val):
            case 0:
                raid_level = RaidLevel.NONE
            case 1:
                raid_level = RaidLevel.LEVEL1_10
            case 2:
                raid_level = RaidLevel.LEVEL2_20
            case 3:
                raid_level = RaidLevel.LEVEL3_40
        match(rs_val):
            case 0:
                rs_level = RSLevel.NONE
            case 1:
                rs_level = RSLevel.LEVEL1_5
            case 2:
                rs_level = RSLevel.LEVEL2_10
            case 3:
                rs_level = RSLevel.LEVEL3_15
    except IndexError:
        print(f"Error: Invalid level index. Raid: {raid_val}, RS: {rs_val}")
        sys.exit(1)
    
    # 执行编码流程
    EncodeFull(file_path, raid_level, rs_level)