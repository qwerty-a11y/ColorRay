import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sys

import numpy as np

from common import Config
from common.CorrectionLevel import RSLevel, RaidLevel
from common.Raid import Raid5Decode, Raid6Decode
from receiver.decoder.array_to_bytes import array_to_bytes, numpy_to_int
from common.RSmodule import rs_decode_bytes
from receiver.detector.test_dll_mov import process_video_pipeline
from sender.generator.frame_gen import generate_frame
from threading import Lock

def GetCorrectionPagesInfo(matrix: np.ndarray) -> tuple[int, RaidLevel, RSLevel]:
    """
    从图片帧中读取三处元信息（左上、左下、右上），进行多数表决。
    若有两处及以上相同以此为准，否则抛出异常。
    注意：不返回当前页码，仅返回 (allpage, raid, rs)。
    右上角读取时需反转颜色顺序以匹配生成逻辑。
    
    :return: (allpage, raid, rs)
    :raises ValueError: 当三处元信息无法达成多数一致时
    """
    # 获取网格尺寸配置 (与 frame_gen 一致)
    GRID_COUNT = Config.QRSize

    '''
    # 调试：输出整个矩阵
    for row in range(GRID_COUNT):
        row_colors = []
        for col in range(GRID_COUNT):
            idx_val = numpy_to_int(matrix, row, col)
            row_colors.append(str(idx_val))
        print(" ".join(row_colors))
    '''
    
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
                idx = numpy_to_int(matrix, start_r, c)  # 每个像素3字节
                colors.append(idx)
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
            
        
        # 【修改】返回不包含页码的元组
        return allpage, raid, rs

    # 定义三处读取位置的起始坐标 (参考 frame_gen 中的绘制逻辑)
    # 1. 左上：(15, 0) - 顺序读取
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

def GetCurrentPage(matrix: np.ndarray) -> int:
    """
    从图片帧中提取当前页码
    :param frame: 图片帧路径
    :return: 当前页码
    :raises ValueError: 当无法读取页码或三处页码不一致时抛出异常
    """
    # 获取网格尺寸配置 (与 frame_gen 一致)
    GRID_COUNT = Config.QRSize
    # 读取三个角落的的前 4 个颜色块解析当前页码
    def extract_page_at_location(start_r: int, start_c: int, reverse_colors: bool = False) -> int:
        colors = []
        for k in range(4):
            c = start_c
            if reverse_colors:  # 【修改】反转颜色顺序
                c = start_c - k
            else:
                c = start_c + k
            if 0 <= start_r < GRID_COUNT and 0 <= c < GRID_COUNT:
                idx = numpy_to_int(matrix, start_r, c)
                colors.append(idx)
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
    pos3 = (15, GRID_COUNT - 1)
    pages = []
    errors = []
    # 带索引循环读取
    for index,pos in enumerate([pos1, pos2, pos3]):
        try:
            page = extract_page_at_location(pos[0], pos[1], index == 2)  # 右上角启用 reverse_colors
            pages.append(page)
        except Exception as e:
            errors.append(f"位置 {index + 1} ({pos}) 读取页码失败：{str(e)}")
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
    """
    RSCorrectBytesPerGroup = 0
    RSBlocks = 0
    match rs:
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = Config.RSCorrectionBytes.LEVEL1_5.r
            RSBlocks = Config.RSCorrectionBytes.LEVEL1_5.b
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = Config.RSCorrectionBytes.LEVEL2_10.r
            RSBlocks = Config.RSCorrectionBytes.LEVEL2_10.b
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = Config.RSCorrectionBytes.LEVEL3_15.r
            RSBlocks = Config.RSCorrectionBytes.LEVEL3_15.b
        case RSLevel.NONE:
            RSCorrectBytesPerGroup = 0
    return file_size + RSBlocks * RSCorrectBytesPerGroup

async def DecodeFull(video: str):

    frame_generator = process_video_pipeline(video, 2000)
    
    # 初始化异步视频抽帧
    first_frame = (await frame_generator.__anext__())
    second_frame = (await frame_generator.__anext__())
    #读取第一帧，获取总页数和raid、rs级别
    pages, raid, rs = None, None, None
    try:
        if first_frame is None:
            raise ValueError(f"无法处理第一帧图片")
        pages, raid, rs = GetCorrectionPagesInfo(first_frame[1])
    except Exception as e:
        print(e.args)
        try:
            if second_frame is None:
                raise ValueError(f"无法处理第二帧图片")
            pages, raid, rs = GetCorrectionPagesInfo(second_frame[1])
        except Exception as e:
            print(f"Error: 无法从前两帧获取总页数和纠错级别。请检查帧文件是否正确。")
            sys.exit(1)

    async def get_frame():
        yield first_frame
        yield second_frame
        async for result in frame_generator:
            yield result

    RSCorrectBytesPerGroup = 0
    FileGroupSize = 0
    raid_text = None
    rs_text = None
    match rs:
        case RSLevel.LEVEL1_5:
            RSCorrectBytesPerGroup = Config.RSCorrectionBytes.LEVEL1_5.r
            FileGroupSize = Config.GroupDataSize - RSCorrectBytesPerGroup * Config.RSCorrectionBytes.LEVEL1_5.b
            rs_text = "5%"
        case RSLevel.LEVEL2_10:
            RSCorrectBytesPerGroup = Config.RSCorrectionBytes.LEVEL2_10.r
            FileGroupSize = Config.GroupDataSize - RSCorrectBytesPerGroup * Config.RSCorrectionBytes.LEVEL2_10.b
            rs_text = "10%"
        case RSLevel.LEVEL3_15:
            RSCorrectBytesPerGroup = Config.RSCorrectionBytes.LEVEL3_15.r
            FileGroupSize = Config.GroupDataSize - RSCorrectBytesPerGroup * Config.RSCorrectionBytes.LEVEL3_15.b
            rs_text = "15%"
        case RSLevel.NONE:
            FileGroupSize = Config.GroupDataSize
            rs_text = "0%"
    
    # 初始化 full_data_groups（锯齿状，用于存储每页的8个数据块）
    full_data_groups: list[list[bytes]] = []
    rows = 0
    match raid:
        case RaidLevel.NONE:
            rows = pages * Config.FrameGroupCount
            full_data_groups = [[None for _ in range(rows)]]  # type: ignore
            raid_text = "0%"
        case RaidLevel.LEVEL1_10:
            rows = pages * Config.FrameGroupCount // 10
            full_data_groups = [[None for _ in range(rows)] for i in range(10)]  # type: ignore
            raid_text = "10%"
        case RaidLevel.LEVEL2_20:
            rows = pages * Config.FrameGroupCount // 5
            full_data_groups = [[None for _ in range(rows)] for i in range(5)]  # type: ignore
            raid_text = "20%"
        case RaidLevel.LEVEL3_40:
            rows = pages * Config.FrameGroupCount // 5
            full_data_groups = [[None for _ in range(rows)] for i in range(5)]  # type: ignore
            raid_text = "40%"

    print(f"开始解码: 共{pages}页, raid纠错率{raid_text}, rs纠错率{rs_text}")

    def decode_frame(img_index: int,matrix: np.ndarray):
        curpage = None
        try:
            curpage = GetCurrentPage(matrix) # type: ignore
        except Exception as e:
            print(f"页码读取失败: {str(e)}. 跳过此帧.")
            return []
        raid_disk = curpage // (rows // 8)
        raid_page = curpage % (rows // 8)
        print(f"读取到第 {img_index+1} 帧，当前页码 {curpage}，对应 RAID 盘 {raid_disk} 页 {raid_page}")
        frame, _ = generate_frame(curpage, pages if pages > 0 else 1, raid, rs)

        data_bytes = array_to_bytes(frame, matrix) # type: ignore

        group_size = len(data_bytes) // Config.FrameGroupCount
        info:list[tuple[int,int,bytes|None]] = []
        for i in range(Config.FrameGroupCount):
            start = i * group_size
            end = (i + 1) * group_size if i < Config.FrameGroupCount - 1 else len(data_bytes)
            chunk = data_bytes[start:end]

            # RS 解码
            decoded_data = rs_decode_bytes(chunk, RSCorrectBytesPerGroup)
            if decoded_data == b'':
                decoded_data = None
                print(f"警告: RAID {raid_disk} 页 {raid_page} 块 {i} 解码失败。")
            info.append((raid_disk, i+raid_page*Config.FrameGroupCount, decoded_data))
        
        return info



    max_workers = min(24, os.cpu_count() or 8)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    lock = Lock()
    loop = asyncio.get_running_loop()
    result_queue = asyncio.Queue(maxsize=32)
    async def producer():
        try:
            async for index,raw_frame in get_frame():
                # 提交到线程池（非阻塞）
                future = loop.run_in_executor(
                    executor,
                    decode_frame,
                    index,
                    raw_frame
                )
                # 将 future 放入队列，后续按顺序取结果
                await result_queue.put(future)
        finally:
            for _ in range(max_workers):
                await result_queue.put(None)

    async def consumer(index:int):
        while True:
            future = await result_queue.get()
            if future is None:
                result_queue.task_done()
                break
            try:
                chunks_info = await future
                with lock:
                    for disk, col, data in chunks_info:
                        if data is not None and full_data_groups[disk][col] is None:
                            full_data_groups[disk][col] = data
            except Exception as e:
                print(f"帧 {index} 写入数据时出错: {e}")
            finally:
                result_queue.task_done()

    producer_task = asyncio.create_task(producer())
    consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(max_workers)]

    await producer_task
    await result_queue.join()
    for c in consumer_tasks:
        c.cancel()                     # 停止消费者
    await asyncio.gather(*consumer_tasks, return_exceptions=True)

    executor.shutdown(wait=False) 
    
    disk_count = len(full_data_groups)
    decoded_disk_count = 1
    # RAID 解码
    match raid:
        case RaidLevel.NONE:
            pass
        case RaidLevel.LEVEL1_10:
            full_data_groups = Raid5Decode(full_data_groups) # type: ignore
            decoded_disk_count = 9
        case RaidLevel.LEVEL2_20:
            full_data_groups = Raid5Decode(full_data_groups) # type: ignore
            decoded_disk_count = 4
        case RaidLevel.LEVEL3_40:
            full_data_groups = Raid6Decode(full_data_groups) # type: ignore
            decoded_disk_count = 3

    # 此时 full_data_groups 是原始数据矩阵（行数 = 数据盘数，列数可能不等，但编码时所有行等长）
    # 按列优先顺序提取所有数据块（与编码填充顺序一致）
    # 首先确定最大列数
    max_cols = max(len(row) for row in full_data_groups) if full_data_groups else 0
    raw_data = b''
    error_count = 0
    for col in range(max_cols):
        for row in range(len(full_data_groups)):
            if col < len(full_data_groups[row]):
                chunk = full_data_groups[row][col]
                if chunk is None:
                    chunk = b'\x00' * FileGroupSize
                    error_count += 1
            else:
                chunk = b'\x00' * FileGroupSize
                error_count += 1
            raw_data += chunk
    print(f"数据长度 {len(raw_data)} 字节，缺失块数 {error_count}，占比 {error_count/(pages*Config.FrameGroupCount*decoded_disk_count//disk_count):.2%}")

    # 解析文件头
    HEADER_LEN = 2 + 254 + 8
    if len(raw_data) < HEADER_LEN:
        raise ValueError("数据过短，无法解析文件头")
    name_len = int.from_bytes(raw_data[:2], 'big')
    if name_len > 254:
        name_len = 254
    file_name_bytes = raw_data[2:2+name_len]
    try:
        file_name = file_name_bytes.decode('utf-8', errors='ignore')
    except Exception:
        file_name = "decoded_file"
    if not file_name:
        file_name = "decoded_file"
    file_size = int.from_bytes(raw_data[256:256+8], 'big')
    if file_size < 0 or file_size > 10 * 1024 * 1024 * 1024:
        print(f"警告：文件大小异常 ({file_size})，设为0")
        file_size = 0

    # 提取有效内容
    file_content = raw_data[HEADER_LEN:HEADER_LEN + file_size]
    if len(file_content) < file_size:
        print(f"警告：有效数据不足，实际 {len(file_content)} 字节，预期 {file_size}")

    print(f"文件名: {file_name}, 文件大小: {file_size} 字节, 实际数据长度: {len(file_content)} 字节")


    # 写入文件
    final_name = file_name
    exe_path = os.path.dirname(sys.executable)
    final_path = os.path.join(exe_path,"decoded", final_name)
    os.makedirs(os.path.join(exe_path,"decoded"), exist_ok=True)
    counter = 1
    while os.path.exists(final_name):
        base, ext = os.path.splitext(file_name)
        final_name = f"{base}({counter}){ext}"
        counter += 1
    with open(final_path, 'wb') as f:
        f.write(file_content)
    print(f"解码完成，输出文件：{final_name}")