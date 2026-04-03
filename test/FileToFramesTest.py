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

from common.CorrectionLevel import RSLevel, RaidLevel
from sender.encoder.Encode import Encode, GroupToFrames
from sender.generator.drawer import drawer
from sender.generator.frame_gen import generate_frame
from sender.generator.bytes_to_colors import bytes_to_colors
from sender.generator.colors_to_matrix import colors_to_matrix


def EncodeFull(path:str, raid:RaidLevel, rs:RSLevel):
    data_groups = Encode(path, raid, rs)
    frames = GroupToFrames(data_groups) # type: ignore
    print(frames)
    pages = len(frames)
    for i in range(pages):
        colors = bytes_to_colors(frames[i])
        frame, need_border = generate_frame(i, pages, raid, rs)
        matrix = colors_to_matrix(frame, colors) # type: ignore
        drawer(matrix, need_border, f"{i}.png")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python FileToVideoTest.py <raid_level_int> <rs_level_int> <file_path>")
        print("Example: python FileToVideoTest.py 0 0 input.txt")
        sys.exit(1)

    # 获取参数
    raid_val = int(sys.argv[1])
    rs_val = int(sys.argv[2])
    file_path = sys.argv[3]

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

    print(f"Input parameters - RAID Level: {raid_level}, RS Level: {rs_level}, File Path: {file_path}")

    # 执行编码流程
    EncodeFull(file_path, raid_level, rs_level)