from typing import List
from functools import reduce
import operator

def xor_blocks(blocks: List[bytes]) -> bytes:
    """
    对多个等长字节数组按字节进行异或，返回结果。
    """
    if not blocks:
        return b''
    # 将每个块的对应字节组合，逐字节异或
    return bytes(reduce(operator.xor, group) for group in zip(*blocks))

def Raid5Encode(data_disks: List[List[bytes]]) -> List[List[bytes]]:
    """
    RAID-5 编码（磁盘视角）。

    参数:
        data_disks: 输入数据磁盘列表。每个内层列表代表一个磁盘上的所有数据块（bytes）。
                    要求：
                    - 所有磁盘的块数相等。
                    - 同一列（即同一索引）的所有数据块长度相等（因为异或运算需要等长）。

    返回:
        编码后的磁盘列表，长度 = 输入磁盘数 + 1。
        每个磁盘的块数与输入相同，但内容已重新排列：
        - 每个条带（列）中有一个校验块，其余为数据块。
        - 校验块位置由经典 RAID-5 公式决定：
          parity_disk = (total_disks - 1 - stripe_index) % total_disks
    """
    if not data_disks:
        return []

    num_data_disks = len(data_disks)
    num_blocks = len(data_disks[0])
    total_disks = num_data_disks + 1

    # 检查所有磁盘块数一致
    for i, disk in enumerate(data_disks):
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 的块数不一致，应为 {num_blocks}，实际为 {len(disk)}")

    # 检查同一列数据块长度是否相等
    if num_blocks > 0:
        for col in range(num_blocks):
            col_blocks = [data_disks[d][col] for d in range(num_data_disks)]
            first_len = len(col_blocks[0])
            if not all(len(blk) == first_len for blk in col_blocks):
                raise ValueError(f"条带（列）{col} 中数据块长度不一致")

    # 初始化输出磁盘列表，全部占位 None
    result_disks:List[List[bytes|None]] = [ [None] * num_blocks for _ in range(total_disks) ]

    for stripe in range(num_blocks):
        # 计算校验块位置
        parity_disk = (total_disks - 1 - stripe) % total_disks

        # 收集该条带所有数据块（用于计算校验块）
        data_blocks = [data_disks[d][stripe] for d in range(num_data_disks)]
        parity = xor_blocks(data_blocks)

        # 放置校验块
        result_disks[parity_disk][stripe] = parity

        # 放置数据块
        for d in range(num_data_disks):
            if d == parity_disk:
                # 该磁盘在此条带是校验盘，其数据块需要移到新增磁盘（索引 num_data_disks）
                result_disks[num_data_disks][stripe] = data_disks[d][stripe]
            else:
                # 数据块留在原磁盘
                result_disks[d][stripe] = data_disks[d][stripe]

    return result_disks # type: ignore

def Raid5Decode(disks: List[List[bytes|None]]) -> List[List[bytes|None]]:
    """
    RAID-5 解码，恢复原始数据磁盘（不含校验块）。
    
    参数:
        disks: 编码后的磁盘列表，每个内层列表代表一个磁盘上的所有块（bytes 或 None）。
               要求所有磁盘块数相等。
    
    返回:
        原始数据磁盘列表，每个内层列表代表一个数据磁盘上的所有块。
        若某块无法恢复，则对应位置为 None。
    """
    if not disks:
        return []

    total_disks = len(disks)
    if total_disks < 2:
        raise ValueError("至少需要2个磁盘（1个数据盘+1个校验盘）")

    num_blocks = len(disks[0])
    # 检查所有磁盘块数一致
    for i, disk in enumerate(disks):
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 块数不一致")

    num_data_disks = total_disks - 1
    result:List[List[bytes|None]] = [[None] * num_blocks for _ in range(num_data_disks)]

    for stripe in range(num_blocks):
        parity_disk = (total_disks - 1 - stripe) % total_disks
        stripe_blocks = [disks[d][stripe] for d in range(total_disks)]

        # 确定每个原始数据磁盘对应的实际源磁盘，并记录缺失情况
        missing_data = []          # 缺失的原始数据磁盘索引
        present_data_blocks = []   # 存在的原始数据块（用于恢复）
        data_src = []              # 每个原始数据磁盘对应的源磁盘索引

        for d in range(num_data_disks):
            src = num_data_disks if d == parity_disk else d
            data_src.append(src)
            block = stripe_blocks[src]
            if block is None:
                missing_data.append(d)
            else:
                present_data_blocks.append(block)

        parity_block = stripe_blocks[parity_disk]
        missing_count = len(missing_data)

        if missing_count == 0:
            # 所有数据块完好，直接赋值
            for d in range(num_data_disks):
                result[d][stripe] = stripe_blocks[data_src[d]]
        elif missing_count == 1 and parity_block is not None:
            # 检查长度一致性
            if present_data_blocks:
                blk_len = len(present_data_blocks[0])
                for blk in present_data_blocks:
                    if len(blk) != blk_len:
                        raise ValueError(f"条带 {stripe} 中数据块长度不一致")
                if len(parity_block) != blk_len:
                    raise ValueError(f"条带 {stripe} 中校验块长度不一致")
            # 恢复缺失块
            all_existing = present_data_blocks + [parity_block]
            recovered = xor_blocks(all_existing)
            missing_d = missing_data[0]
            for d in range(num_data_disks):
                if d == missing_d:
                    result[d][stripe] = recovered
                else:
                    result[d][stripe] = stripe_blocks[data_src[d]]
        else:
            # 无法恢复，该条带所有数据块保持 None（已初始化）
            pass

    return result

def Raid6Encode(data_disks: List[List[bytes]]) -> List[List[bytes]]:
    """
    RAID-6 编码（使用 pyjerasure 库）。
    参数:
        data_disks: 数据磁盘列表，每个内层列表代表一个磁盘上的所有数据块（bytes）。
                    要求所有磁盘的块数相等，且同一列（条带）的所有数据块长度相等。
    返回:
        编码后的磁盘列表，长度 = 输入磁盘数 + 2。
        校验块（P 和 Q）按 RAID-6 规则放置。
    """
    import pyjerasure

    if not data_disks:
        return []

    num_data_disks = len(data_disks)
    num_blocks = len(data_disks[0])
    total_disks = num_data_disks + 2

    # 检查所有磁盘块数一致
    for i, disk in enumerate(data_disks):
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 的块数不一致")

    # 检查同一列数据块长度相等
    block_len = None
    for col in range(num_blocks):
        col_blocks = [data_disks[d][col] for d in range(num_data_disks)]
        first_len = len(col_blocks[0])
        if not all(len(blk) == first_len for blk in col_blocks):
            raise ValueError(f"条带 {col} 中数据块长度不一致")
        if block_len is None:
            block_len = first_len
        elif block_len != first_len:
            raise ValueError("不同条带的数据块长度不一致")

    # 初始化输出磁盘列表
    result_disks:List[List[bytes|None]] = [[None] * num_blocks for _ in range(total_disks)]

    # 对于每个条带，使用 pyjerasure 编码
    for stripe in range(num_blocks):
        # 收集该条带所有数据块
        data_blocks = [data_disks[d][stripe] for d in range(num_data_disks)]
        
        # 将数据块拼接成连续内存
        data = b''.join(data_blocks)
        
        # pyjerasure 参数：
        # k = 数据盘数量
        # m = 校验盘数量（2）
        # w = 字大小（8 bits，即一个字节）
        # size = 每个数据块的大小
        k = num_data_disks
        m = 2
        w = 8
        size:int = block_len # type: ignore
        
        # 使用 RAID-6 专用矩阵 (reed_sol_r6_op)
        matrix = pyjerasure.Matrix("reed_sol_r6_op", k, m, w)
        
        # 编码：输入数据长度应为 k * size，输出长度为 (k + m) * size
        # 输出格式：[data_chunk_0, data_chunk_1, ..., data_chunk_k-1, coding_chunk_0, coding_chunk_1]
        encoded = pyjerasure.encode(matrix, data, size)
        
        # 分离出校验块（最后两个块）
        parity_blocks = [
            encoded[k * size : (k + 1) * size],  # P 校验块
            encoded[(k + 1) * size : (k + 2) * size]   # Q 校验块
        ]
        
        # 确定 P 和 Q 的放置位置（这里简化：P 在倒数第二盘，Q 在最后）
        p_disk = num_data_disks
        q_disk = num_data_disks + 1
        
        # 放置校验块
        result_disks[p_disk][stripe] = parity_blocks[0]
        result_disks[q_disk][stripe] = parity_blocks[1]
        
        # 放置数据块
        for d in range(num_data_disks):
            result_disks[d][stripe] = data_blocks[d]

    return result_disks # type: ignore

def Raid6Decode(disks: List[List[bytes|None]]) -> List[List[bytes|None]]:
    """
    RAID-6 解码，恢复原始数据磁盘（不含校验块）。
    参数:
        disks: 编码后的磁盘列表，每个内层列表代表一个磁盘上的所有块（bytes 或 None）。
               要求所有磁盘块数相等，最后两个磁盘为 P 和 Q 校验盘。
    返回:
        原始数据磁盘列表，每个内层列表代表一个数据磁盘上的所有块。
        若某块无法恢复（如丢失块数超过 2），则对应位置为 None。
    """
    import pyjerasure

    if not disks:
        return []

    total_disks = len(disks)
    if total_disks < 3:
        raise ValueError("至少需要 3 个磁盘（1个数据盘 + 2个校验盘）")

    num_blocks = len(disks[0])
    for i, disk in enumerate(disks):
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 块数不一致")

    num_data_disks = total_disks - 2
    result:List[List[bytes|None]] = [[None] * num_blocks for _ in range(num_data_disks)]

    for stripe in range(num_blocks):
        # 收集该条带所有块
        stripe_blocks = [disks[d][stripe] for d in range(total_disks)]
        
        # 分离数据块和校验块（P 在倒数第二，Q 在最后）
        data_blocks = stripe_blocks[:num_data_disks]
        p_block = stripe_blocks[num_data_disks]
        q_block = stripe_blocks[num_data_disks + 1]

        # 检查数据块是否有 None
        missing_indices = [i for i, blk in enumerate(data_blocks) if blk is None]
        missing_count = len(missing_indices)

        # 情况 1：无缺失
        if missing_count == 0:
            for d in range(num_data_disks):
                result[d][stripe] = data_blocks[d]
            continue

        # 情况 2：缺失 1 块且 P 存在，可用 XOR 恢复（pyjerasure 也会自动处理）
        # 情况 3：缺失 1-2 块，统一用 pyjerasure 解码
        
        # 确定块大小（取任意存在块的长度）
        block_len = None
        for blk in stripe_blocks:
            if blk is not None:
                block_len = len(blk)
                break
        if block_len is None:
            # 没有可用块，无法恢复
            continue

        # 构建 pyjerasure 解码所需的输入
        # 需要准备一个长度为 (total_disks * block_len) 的字节数组，
        # 缺失的位置用任意字节填充（例如全零），并记录擦除位置（缺失块的索引）
        data_to_decode = bytearray(total_disks * block_len)
        erasures = []
        
        # 填充数据
        for d in range(total_disks):
            if stripe_blocks[d] is not None:
                # 存在块，复制到对应位置
                start = d * block_len
                data_to_decode[start:start + block_len] = stripe_blocks[d] # type: ignore
            else:
                # 缺失块，填充零并记录擦除位置
                erasures.append(d)
        
        # 如果擦除数量 > 2，无法恢复
        if len(erasures) > 2:
            continue
        
        # 使用 pyjerasure 解码
        try:
            k = num_data_disks
            m = 2
            w = 8
            matrix = pyjerasure.Matrix("reed_sol_r6_op", k, m, w)
            
            # 解码，返回恢复后的完整数据
            recovered = pyjerasure.decode(matrix, bytes(data_to_decode), block_len, erasures)
            
            # recovered 长度应为 (k + m) * block_len
            # 提取数据块部分（前 k 块）
            for d in range(num_data_disks):
                start = d * block_len
                result[d][stripe] = recovered[start:start + block_len]
        except Exception as e:
            # 解码失败，该条带无法恢复
            continue

    return result