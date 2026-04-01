import math
import os
from typing import List

from common import Config
from common.CRC16 import verify_crc16
from common.CorrectionLevel import RSLevel, RaidLevel
from common.Raid import Raid5Decode, Raid6Decode
from receiver.decoder.colors_to_bytes import colors_to_bytes
from receiver.decoder.image_to_matrix import array_to_matrix, image_to_matrix
from receiver.decoder.matrix_to_colors import matrix_to_colors
from common.RSmodule import rs_decode_bytes
from receiver.detector.img_extract import process_photo
from sender.generator.frame_gen import COLORS, generate_frame

'''
def Decode(data_groups: List[List[bytes]], raid: RaidLevel, rs: RSLevel, 
           output_dir: str = None) -> tuple[str, bytes]:
    """
    完整的文件解码流程（Encode 函数的反向操作）
    
    处理流程：
    1. RAID 解码 → 恢复原始数据块
    2. RS 解码每个块 → 恢复数据和 CRC
    3. CRC16 验证 → 区分真实块和填充块
    4. 合并真实块 → 恢复原始数据流
    5. 解析文件头 → 提取文件名和大小
    6. 生成文件 → 保存到指定路径
    
    :param data_groups: RAID 编码后的二维数据组 [行数][列数]
    :param raid: RAID 等级
    :param rs: RS 纠错等级
    :param output_dir: 输出目录（默认为 ./data）
    :return: (文件名, 文件内容) 元组
    :raises ValueError: 数据损坏或无法恢复
    """
    import os
    
    # 设置默认输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 获取 RS 参数
    rs_correct_bytes = _get_rs_correct_bytes(rs)
    
    # 2. RAID 解码
    print(f"[1/6] 执行 RAID 解码 ({raid.name})...")
    data_groups_decoded = _perform_raid_decode(data_groups, raid)
    
    # 3. 从二维数组提取块序列
    print(f"[2/6] 提取数据块...")
    all_blocks = _extract_blocks_from_2d(data_groups_decoded)
    print(f"  共 {len(all_blocks)} 个块")
    
    # 4. RS 解码 + CRC 验证
    print(f"[3/6] RS 解码和 CRC 验证...")
    decoded_blocks, block_validity = _decode_and_verify_blocks(all_blocks, rs_correct_bytes)
    valid_count = sum(1 for v in block_validity if v)
    print(f"  有效块: {valid_count}/{len(all_blocks)}")
    
    # 5. 从真实块恢复原始数据
    print(f"[4/6] 合并数据块...")
    reconstructed_data = _reconstruct_data_from_blocks(decoded_blocks, block_validity)
    print(f"  恢复数据大小: {len(reconstructed_data)} 字节")
    
    # 6. 解析文件头
    print(f"[5/6] 解析文件头...")
    file_name, original_size, file_data = _extract_file_from_data(reconstructed_data)
    print(f"  文件名: {file_name}")
    print(f"  原始大小: {original_size} 字节")
    
    # 7. 保存文件
    print(f"[6/6] 保存文件...")
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'wb') as f:
        f.write(file_data)
    print(f"  已保存: {output_path}")
    
    return file_name, file_data


# ============================================================================
# 辅助函数
# ============================================================================

def _get_rs_correct_bytes(rs: RSLevel) -> int:
    """获取 RS 纠错字节数"""
    match rs:
        case RSLevel.LEVEL1_5:
            return 10
        case RSLevel.LEVEL2_10:
            return 20
        case RSLevel.LEVEL3_15:
            return 40
        case RSLevel.NONE:
            return 0
        case _:
            raise ValueError(f"未知的 RS 等级: {rs}")


def _perform_raid_decode(data_groups: List[List[bytes]], raid: RaidLevel) -> List[List[bytes | None]]:
    """执行 RAID 解码"""
    match raid:
        case RaidLevel.LEVEL1_10 | RaidLevel.LEVEL2_20:
            return Raid5Decode(data_groups)
        case RaidLevel.LEVEL3_40:
            return Raid6Decode(data_groups)
        case RaidLevel.NONE:
            return data_groups
        case _:
            raise ValueError(f"未知的 RAID 等级: {raid}")


def _extract_blocks_from_2d(data_groups: List[List[bytes | None]]) -> List[bytes]:
    """从 2D 数组按列优先顺序提取块序列"""
    blocks = []
    
    if not data_groups:
        return blocks
    
    rows = len(data_groups)
    cols = len(data_groups[0]) if rows > 0 else 0
    
    # 按列优先顺序遍历
    for col in range(cols):
        for row in range(rows):
            if data_groups[row][col] is not None:
                blocks.append(data_groups[row][col])
    
    return blocks


def _decode_and_verify_blocks(blocks: List[bytes], 
                               rs_correct_bytes: int) -> tuple[List[bytes], List[bool]]:
    """RS 解码和 CRC16 验证（区分真实块和填充块）"""
    decoded_blocks = []
    block_validity = []
    
    for i, block in enumerate(blocks):
        try:
            # 检查块大小
            if len(block) != GroupDataSize:
                print(f"  警告: 块 {i} 大小异常 ({len(block)} != {GroupDataSize})")
                decoded_blocks.append(b'')
                block_validity.append(False)
                continue
            
            # 【修改】如果是 LEVEL3_20（rs_correct_bytes=40），只截取前 749 字节
            # 因为后 40 字节被随机填充了（789 - 40 = 749）
            block_to_decode = block
            if rs_correct_bytes == 40:
                block_to_decode = block[:749]
            
            # RS 解码
            decoded = rs_decode_bytes(block_to_decode, rs_correct_bytes)
            
            # CRC16 验证
            if verify_crc16(decoded):
                # CRC 通过：真实数据块
                actual_data = decoded[2:]  # 去掉前 2 字节的 CRC
                decoded_blocks.append(actual_data)
                block_validity.append(True)
            else:
                # CRC 失败：填充块
                decoded_blocks.append(b'')
                block_validity.append(False)
                
        except Exception as e:
            print(f"  警告: 块 {i} 解码失败 - {str(e)}")
            decoded_blocks.append(b'')
            block_validity.append(False)
    
    return decoded_blocks, block_validity


def _reconstruct_data_from_blocks(decoded_blocks: List[bytes],
                                   block_validity: List[bool]) -> bytes:
    """从真实块重建原始数据流"""
    reconstructed = b''
    
    for block_data, is_valid in zip(decoded_blocks, block_validity):
        if is_valid:
            reconstructed += block_data
    
    return reconstructed


def _extract_file_from_data(data: bytes) -> tuple[str, int, bytes]:
    """
    从数据中提取文件头（264字节）并恢复原始文件
    
    文件头格式：
    - 字节 0-1: 文件名长度（2 字节大端序）
    - 字节 2-255: 文件名（254 字节，不足补 0）
    - 字节 256-263: 文件大小（8 字节大端序）
    - 字节 264+: 文件实际内容
    """
    HEADER_SIZE = 264
    FILE_NAME_MAX_LEN = 254
    
    if len(data) < HEADER_SIZE:
        raise ValueError(f"数据不足以解析文件头 (期望 {HEADER_SIZE}, 实际 {len(data)})")
    
    header = data[:HEADER_SIZE]
    file_content = data[HEADER_SIZE:]
    
    # 解析文件名长度
    name_len = int.from_bytes(header[0:2], byteorder='big')
    if name_len > FILE_NAME_MAX_LEN:
        raise ValueError(f"文件名长度无效: {name_len}")
    
    # 解析文件名
    padded_name = header[2:2+FILE_NAME_MAX_LEN]
    file_name = padded_name[:name_len].decode('utf-8', errors='ignore')
    
    # 解析原始文件大小
    original_size = int.from_bytes(header[256:264], byteorder='big')
    
    # 根据原始大小裁剪文件内容
    if len(file_content) >= original_size:
        file_content = file_content[:original_size]
    else:
        print(f"  警告: 文件内容不足 (期望 {original_size}, 实际 {len(file_content)})")
    
    return file_name, original_size, file_content

'''
def GetCorrectionPagesInfo(frame:str) -> tuple[int, RaidLevel, RSLevel]:
    """
    从图片帧中读取三处元信息（左上、左下、右上），进行多数表决。
    若有两处及以上相同以此为准，否则抛出异常。
    注意：不返回当前页码，仅返回 (allpage, raid, rs)。
    右上角读取时需反转颜色顺序以匹配生成逻辑。
    
    :return: (allpage, raid, rs)
    :raises ValueError: 当三处元信息无法达成多数一致时
    """
    array = process_photo(frame)
    if array is None:
        raise ValueError("无法处理图片帧，未提取到有效数据")
    
    matrix = array_to_matrix(array)
    # 获取网格尺寸配置 (与 frame_gen 一致)
    GRID_COUNT = Config.QRSize
    
    def extract_meta_at_location(start_r: int, start_c: int, reverse_colors: bool = False) -> tuple[int, RaidLevel, RSLevel]:
        """
        从指定起始位置读取 10 个颜色块并解析元信息
        对应 frame_gen 中的逻辑：
        header = page_to_color(curpage)+page_to_color(allpage)+raid_rs_to_color(raid, rs)
        前 4 位：页码 (4 位 8 进制) - 【本次修改：忽略】
        中 4 位：总页数 (4 位 8 进制)
        后 2 位：RAID(1 位) + RS(1 位)
        
        :param reverse_colors: 是否反转颜色列表（用于右上角修复）
        """
        colors = []
        for k in range(10):
            c = start_c + k
            if 0 <= start_r < GRID_COUNT and 0 <= c < GRID_COUNT:
                color_val = matrix[start_r][c]
                # 将 RGB 元组映射回 0-7 的索引
                try:
                    idx = COLORS.index(color_val)
                    colors.append(idx)
                except ValueError:
                    # 如果颜色不匹配标准色，抛出错误或返回无效标记
                    raise ValueError(f"位置 ({start_r}, {c}) 颜色无效：{color_val}")
            else:
                raise ValueError(f"坐标越界：({start_r}, {c})")
        
        # 【修改】如果是右上角，需要反转颜色列表以匹配生成时的反向写入
        if reverse_colors:
            colors = colors[::-1]
        
        # 【修改】跳过前 4 位页码，直接从索引 4 开始解析总页数
        # 解析总页数 (原中 4 位，现索引 4-7)
        allpage_str = "".join(str(d) for d in colors[4:8])
        allpage = int(allpage_str, 8)
        
        # 解析 RAID 等级 (第 9 位，索引 8)
        raid_map = {
            0: RaidLevel.NONE,
            1: RaidLevel.LEVEL1_10,
            2: RaidLevel.LEVEL2_20,
            3: RaidLevel.LEVEL3_40
        }
        raid = raid_map.get(colors[8])
        if raid is None:
            raise ValueError(f"无效的 RAID 等级颜色索引：{colors[8]}")
            
        # 解析 RS 等级 (第 10 位，索引 9)
        rs_map = {
            0: RSLevel.NONE,
            1: RSLevel.LEVEL1_5,
            2: RSLevel.LEVEL2_10,
            3: RSLevel.LEVEL3_15
        }
        rs = rs_map.get(colors[9])
        if rs is None:
            raise ValueError(f"无效的 RS 等级颜色索引：{colors[9]}")
            
        print(colors)
        # 【修改】返回不包含页码的元组
        return allpage, raid, rs

    # 定义三处读取位置的起始坐标 (参考 frame_gen 中的绘制逻辑)
    # 1. 左上：(17, 2) - 顺序读取
    pos1 = (15, 0)
    # 2. 左下：(GRID_COUNT - 18, 2) - 顺序读取
    pos2 = (GRID_COUNT - 16, 0)
    # 3. 右上：(17, GRID_COUNT - 12) - 【修改】需反转颜色顺序
    # frame_gen 中写入逻辑为：for k in range(10): set_cell(17, GRID_COUNT - 3 - k, header[k])
    # 即物理位置上从左到右读取到的颜色，对应逻辑 header 的顺序是反的
    pos3 = (15, GRID_COUNT - 10)
    
    results = []
    errors = []
    
    # 左上
    try:
        res = extract_meta_at_location(pos1[0], pos1[1], reverse_colors=False)
        results.append(res)
    except Exception as e:
        errors.append(f"位置 1(左上) 读取失败：{str(e)}")
        results.append(None)

    # 左下
    try:
        res = extract_meta_at_location(pos2[0], pos2[1], reverse_colors=False)
        results.append(res)
    except Exception as e:
        errors.append(f"位置 2(左下) 读取失败：{str(e)}")
        results.append(None)

    # 右上 - 【修改】启用 reverse_colors=True
    try:
        res = extract_meta_at_location(pos3[0], pos3[1], reverse_colors=True)
        results.append(res)
    except Exception as e:
        errors.append(f"位置 3(右上) 读取失败：{str(e)}")
        results.append(None)

    # 多数表决逻辑
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) < 2:
        raise ValueError(f"无法读取足够的有效元信息 (成功 {len(valid_results)}/3). 错误详情：{'; '.join(errors)}")
    
    # 检查是否有两个或以上相同
    # 比较元组 (allpage, raid, rs)
    count_map = {}
    for res in valid_results:
        count_map[res] = count_map.get(res, 0) + 1
    
    final_result = None
    for res, count in count_map.items():
        if count >= 2:
            final_result = res
            break
    
    if final_result is None:
        # 如果没有出现两次以上的相同结果，说明三者互不相同或只有两两不同且无多数
        raise ValueError(f"三处元信息不一致且无多数派：{results}. 无法确定正确元信息。")
    
    # final_result 现在是 (allpage, raid, rs)
    return final_result

def GetCurrentPage(frame: str) -> int:
    """
    从图片帧中提取当前页码
    :param frame: 图片帧路径
    :return: 当前页码
    :raises ValueError: 当无法读取页码或三处页码不一致时抛出异常
    """

    array = process_photo(frame)
    if array is None:
        raise ValueError("无法处理图片帧，未提取到有效数据")
    
    matrix = array_to_matrix(array)
    # 获取网格尺寸配置 (与 frame_gen 一致)
    GRID_COUNT = Config.QRSize
    # 读取三个角落的的前 4 个颜色块解析当前页码
    def extract_page_at_location(start_r: int, start_c: int) -> int:
        colors = []
        for k in range(4):
            c = start_c + k
            if 0 <= start_r < GRID_COUNT and 0 <= c < GRID_COUNT:
                color_val = matrix[start_r][c]
                # 将 RGB 元组映射回 0-7 的索引
                try:
                    idx = COLORS.index(color_val)
                    colors.append(idx)
                except ValueError:
                    # 如果颜色不匹配标准色，抛出错误或返回无效标记
                    raise ValueError(f"位置 ({start_r}, {c}) 颜色无效：{color_val}")
                else:
                    raise ValueError(f"坐标越界：({start_r}, {c})")
        page_str = "".join(str(d) for d in colors)
        return int(page_str, 8)
    # 定义三个读取位置的起始坐标 (参考 frame_gen 中的绘制逻辑)
    # 1. 左上：(15, 0)
    pos1 = (15, 0)
    # 2. 左下：(GRID_COUNT - 16, 0)
    pos2 = (GRID_COUNT - 16, 0)
    # 3. 右上：(15, GRID_COUNT - 10)
    pos3 = (15, GRID_COUNT - 10)
    pages = []
    errors = []
    for pos in [pos1, pos2, pos3]:
        try:
            page = extract_page_at_location(pos[0], pos[1])
            pages.append(page)
        except Exception as e:
            errors.append(f"位置 {pos} 读取页码失败：{str(e)}")
    if not pages:
        raise ValueError(f"无法读取任何页码。错误详情：{'; '.join(errors)}")
    # 多数表决
    page_count = {}
    for p in pages:
        page_count[p] = page_count.get(p, 0) + 1
    final_page = None
    for p, count in page_count.items():
        if count >= 2:
            final_page = p
            break
    if final_page is None:
        raise ValueError(f"读取的页码不一致且无多数派：{pages}. 无法确定当前页码。")
    return final_page

def calculate_encoded_size(file_size: int, rs: RSLevel) -> int:
    """
    根据原始文件大小和 RS 纠错码位数，计算最终的 RS 编码后数据大小。
    
    参数:
        file_size (int): 原始文件的大小（字节）。
        rs_correct_bytes (int): 每个数据块的 RS 纠错码字节数。
        
    返回:
        int: 编码后的理论总数据大小。
        
    注意:
        此处为框架占位，具体计算逻辑需结合分块策略（Config.FrameGroupCount）
        和填充规则后续完善。
    """
    RSBlocks = 0
    match rs:
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = 10
            #向下取整
            RSBlocks = file_size // (Config.GroupDataSize - 2 - RSCorrectBytesPerGroup * 4)
            RSBlocks += math.ceil((file_size - RSBlocks * Config.GroupDataSize) / 245)
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = 20
            RSBlocks = file_size // (Config.GroupDataSize - 2 - RSCorrectBytesPerGroup * 4)
            RSBlocks += math.ceil((file_size - RSBlocks * Config.GroupDataSize) / 235)
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = 40
            RSBlocks = file_size // (Config.GroupDataSize - 2 - RSCorrectBytesPerGroup * 3)
            RSBlocks += math.ceil((file_size - RSBlocks * Config.GroupDataSize) / 215)
        case RSLevel.NONE:
            RSCorrectBytesPerGroup = 0
    return file_size + RSBlocks * RSCorrectBytesPerGroup

def DecodeFull(pages: int, path:str, raid:RaidLevel, rs:RSLevel):
    print(f"开始解码: pages={pages}, path='{path}', raid={raid}, rs={rs}")
    full_data_groups:list[list[bytes]] = []
    match raid:
        case RaidLevel.NONE:
            full_data_groups = [[]]
        case RaidLevel.LEVEL1_10:
            full_data_groups = [[] for _ in range(10)]
        case RaidLevel.LEVEL2_20, RaidLevel.LEVEL3_40:
            full_data_groups = [[] for _ in range(5)]
    raid_disk = 0
    img_index = -1
    for k in range(pages):
        raid_disk = (raid_disk + 1) % len(full_data_groups)
        while (True):
            img_index += 1
            img_path = os.path.join(path, str(k) + ".png")
            array = process_photo(img_path)
            if array is None:
                print(f"警告: 无法处理图片 {img_path}，跳过该页")
                continue
        
        matrix = array_to_matrix(array)
        frame, _ = generate_frame(k, pages, raid, rs)
        colors = matrix_to_colors(frame, matrix)
        data_bytes = colors_to_bytes(colors)
        print(f"第 {k} 页颜色数据转换为字节完成，长度 {len(data_bytes)} 字节")
        #将data_bytes均分为8组
        group_size = len(data_bytes) // Config.FrameGroupCount
        for i in range(Config.FrameGroupCount):
            start = i * group_size
            end = (i + 1) * group_size if i < Config.FrameGroupCount - 1 else len(data_bytes)
            full_data_groups[raid_disk].append(data_bytes[start:end])
    # Raid解码
    match raid:
        case RaidLevel.NONE:
            pass
        case RaidLevel.LEVEL1_10:
            full_data_groups = Raid5Decode(full_data_groups) # type: ignore
        case RaidLevel.LEVEL2_20:
            full_data_groups = Raid5Decode(full_data_groups) # type: ignore
        case RaidLevel.LEVEL3_40:
            full_data_groups = Raid6Decode(full_data_groups) # type: ignore
    # RS解码
    RSCorrectBytesPerGroup = 0
    match rs:
        case RSLevel.NONE:
            pass
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = 10
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = 20
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = 40
    final_data:list[bytes] = []
    # 先解码第一个数据块，获取元数据后对多余填充进行截断，再对剩余数据块进行解码
    print(full_data_groups)
    first_group_decoded = rs_decode_bytes(full_data_groups[0][0], RSCorrectBytesPerGroup) # type: ignore
    file_name_size = int.from_bytes(first_group_decoded[:4], byteorder='big')
    file_name = first_group_decoded[4:4+file_name_size].decode('utf-8')
    file_size = int.from_bytes(first_group_decoded[4+file_name_size:4+file_name_size+8], byteorder='big')
    encoded_size = calculate_encoded_size(file_size, rs)
    
    # 截断多余填充：按顺序以 [0][0],[1][0],[2][0]...[0][1],[1][1]... 顺序保留数据块
    current_length = 0
    
    # 获取最大块数，防止索引越界
    max_blocks = max(len(group) for group in full_data_groups) if full_data_groups else 0
    
    # 外层循环遍历块索引 (第 0 块，第 1 块...)
    for block_idx in range(max_blocks):
        if current_length >= encoded_size:
            break
        # 内层循环遍历组索引 (第 0 组，第 1 组...)
        for group_idx in range(len(full_data_groups)):
            if current_length >= encoded_size:
                break
            if block_idx < len(full_data_groups[group_idx]):
                chunk = full_data_groups[group_idx][block_idx]
                needed = encoded_size - current_length
                if len(chunk) <= needed:
                    final_data.append(chunk)
                    current_length += len(chunk)
                else:
                    final_data.append(chunk[:needed])
    if RSCorrectBytesPerGroup:
        for bytesblock in final_data:
            rs_decode_bytes(bytesblock, RSCorrectBytesPerGroup)
    with open(path, 'wb') as f:
        f.write(b''.join([bytesblock if bytesblock is not None else b'\x00'*Config.GroupDataSize for bytesblock in final_data ]))

