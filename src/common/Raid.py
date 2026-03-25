from typing import List
from functools import reduce
import operator

import pyjerasure

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
    RAID-5 编码（磁盘视角），以每 8 个条带为单位进行编码。
    """
    if not data_disks:
        return []

    num_data_disks = len(data_disks)
    
    # 【修改】过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in data_disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []
        
    num_blocks = len(non_empty_disks[0])
    total_disks = num_data_disks + 1
    BATCH_SIZE = 8  # 每 8 个条带为一个单位

    # 【修改】检查所有非空磁盘块数一致，空盘忽略
    for i, disk in enumerate(data_disks):
        if len(disk) == 0:
            continue
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 的块数不一致，应为 {num_blocks}，实际为 {len(disk)}")

    if num_blocks > 0:
        for col in range(num_blocks):
            col_blocks = [data_disks[d][col] for d in range(num_data_disks) if len(data_disks[d]) > 0]
            if not col_blocks:
                continue
            first_len = len(col_blocks[0])
            if not all(len(blk) == first_len for blk in col_blocks):
                sizes = [[len(data_disks[d][c]) for c in range(num_blocks)] for d in range(num_data_disks) if len(data_disks[d]) > 0]
                print(f"数据块长度不一致，磁盘数据块长度分布：{sizes}")
                raise ValueError(f"条带（列）{col} 中数据块长度不一致")

    # 初始化输出磁盘列表
    result_disks:List[List[bytes|None]] = [ [None] * num_blocks for _ in range(total_disks) ]

    # 【修改】按批次处理，每 8 个条带为一个单位
    for batch_start in range(0, num_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_blocks)
        
        for stripe in range(batch_start, batch_end):
            # 【修改】校验盘位置计算基于当前批次内的相对位置，或保持全局逻辑？
            # 需求要求“以每 8 个条带为单位”，通常意味着每个单元独立计算校验。
            # 这里采用相对偏移计算，使每个 8 条带组内的校验盘轮转独立。
            relative_stripe = stripe - batch_start
            parity_disk = (total_disks - 1 - relative_stripe) % total_disks

            data_blocks = [data_disks[d][stripe] for d in range(num_data_disks)]
            parity = xor_blocks(data_blocks)

            result_disks[parity_disk][stripe] = parity

            for d in range(num_data_disks):
                if d == parity_disk:
                    result_disks[num_data_disks][stripe] = data_disks[d][stripe]
                else:
                    result_disks[d][stripe] = data_disks[d][stripe]

    return result_disks # type: ignore

def Raid5Decode(disks: List[List[bytes|None]]) -> List[List[bytes|None]]:
    """
    RAID-5 解码，以每 8 个条带为单位进行恢复。
    """
    if not disks:
        return []

    total_disks = len(disks)
    if total_disks < 2:
        raise ValueError("至少需要 2 个磁盘（1 个数据盘 +1 个校验盘）")

    # 【修改】过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []

    num_blocks = len(non_empty_disks[0])
    BATCH_SIZE = 8

    # 【修改】检查所有非空磁盘块数一致，空盘忽略
    for i, disk in enumerate(disks):
        if len(disk) == 0:
            continue
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 块数不一致")

    num_data_disks = total_disks - 1
    result:List[List[bytes|None]] = [[None] * num_blocks for _ in range(num_data_disks)]

    # 【修改】按批次处理
    for batch_start in range(0, num_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_blocks)

        for stripe in range(batch_start, batch_end):
            relative_stripe = stripe - batch_start
            parity_disk = (total_disks - 1 - relative_stripe) % total_disks
            
            stripe_blocks = [disks[d][stripe] for d in range(total_disks)]

            missing_data = []
            present_data_blocks = []
            data_src = []

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
                for d in range(num_data_disks):
                    result[d][stripe] = stripe_blocks[data_src[d]]
            elif missing_count == 1 and parity_block is not None:
                if present_data_blocks:
                    blk_len = len(present_data_blocks[0])
                    for blk in present_data_blocks:
                        if len(blk) != blk_len:
                            raise ValueError(f"条带 {stripe} 中数据块长度不一致")
                    if len(parity_block) != blk_len:
                        raise ValueError(f"条带 {stripe} 中校验块长度不一致")
                
                all_existing = present_data_blocks + [parity_block]
                recovered = xor_blocks(all_existing)
                missing_d = missing_data[0]
                for d in range(num_data_disks):
                    if d == missing_d:
                        result[d][stripe] = recovered
                    else:
                        result[d][stripe] = stripe_blocks[data_src[d]]
            else:
                pass

    return result

def Raid6Encode(data_disks: List[List[bytes]]) -> List[List[bytes]]:
    """
    RAID-6 编码（使用 pyjerasure 库），以每 8 个条带为单位进行编码。
    """
    if not data_disks:
        return []

    num_data_disks = len(data_disks)
    
    # 【修改】过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in data_disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []
        
    num_blocks = len(non_empty_disks[0])
    total_disks = num_data_disks + 2
    BATCH_SIZE = 8

    # 【修改】检查所有非空磁盘块数一致，空盘忽略
    for i, disk in enumerate(data_disks):
        if len(disk) == 0:
            continue
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 的块数不一致")

    block_len = None
    for col in range(num_blocks):
        col_blocks = [data_disks[d][col] for d in range(num_data_disks) if len(data_disks[d]) > 0]
        if not col_blocks:
            continue
        first_len = len(col_blocks[0])
        if not all(len(blk) == first_len for blk in col_blocks):
            raise ValueError(f"条带 {col} 中数据块长度不一致")
        if block_len is None:
            block_len = first_len
        elif block_len != first_len:
            raise ValueError("不同条带的数据块长度不一致")

    result_disks:List[List[bytes|None]] = [[None] * num_blocks for _ in range(total_disks)]

    # 【修改】按批次处理
    for batch_start in range(0, num_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_blocks)
        
        for stripe in range(batch_start, batch_end):
            data_blocks = [data_disks[d][stripe] for d in range(num_data_disks)]
            data = b''.join(data_blocks)
            
            k = num_data_disks
            m = 2
            w = 8
            size:int = block_len # type: ignore
            
            matrix = pyjerasure.Matrix("reed_sol_r6_op", k, m, w)
            encoded = pyjerasure.encode(matrix, data, size)
            
            parity_blocks = [
                encoded[k * size : (k + 1) * size],
                encoded[(k + 1) * size : (k + 2) * size]
            ]
            
            # 校验盘位置固定为最后两个，因为我们是按条带独立编码，不涉及跨条带轮转
            p_disk = num_data_disks
            q_disk = num_data_disks + 1
            
            result_disks[p_disk][stripe] = parity_blocks[0]
            result_disks[q_disk][stripe] = parity_blocks[1]
            
            for d in range(num_data_disks):
                result_disks[d][stripe] = data_blocks[d]

    return result_disks # type: ignore

def Raid6Decode(disks: List[List[bytes|None]]) -> List[List[bytes|None]]:
    """
    RAID-6 解码，以每 8 个条带为单位进行恢复。
    """
    if not disks:
        return []

    total_disks = len(disks)
    if total_disks < 3:
        raise ValueError("至少需要 3 个磁盘（1 个数据盘 + 2 个校验盘）")

    # 【修改】过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []

    num_blocks = len(non_empty_disks[0])
    BATCH_SIZE = 8

    # 【修改】检查所有非空磁盘块数一致，空盘忽略
    for i, disk in enumerate(disks):
        if len(disk) == 0:
            continue
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 块数不一致")

    num_data_disks = total_disks - 2
    result:List[List[bytes|None]] = [[None] * num_blocks for _ in range(num_data_disks)]

    # 【修改】按批次处理（虽然逻辑上是逐条带独立，但结构上遵循批次循环）
    for batch_start in range(0, num_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_blocks)

        for stripe in range(batch_start, batch_end):
            stripe_blocks = [disks[d][stripe] for d in range(total_disks)]
            
            data_blocks = stripe_blocks[:num_data_disks]
            p_block = stripe_blocks[num_data_disks]
            q_block = stripe_blocks[num_data_disks + 1]

            missing_indices = [i for i, blk in enumerate(data_blocks) if blk is None]
            missing_count = len(missing_indices)

            if missing_count == 0:
                for d in range(num_data_disks):
                    result[d][stripe] = data_blocks[d]
                continue

            block_len = None
            for blk in stripe_blocks:
                if blk is not None:
                    block_len = len(blk)
                    break
            if block_len is None:
                continue

            data_to_decode = bytearray(total_disks * block_len)
            erasures = []
            
            for d in range(total_disks):
                if stripe_blocks[d] is not None:
                    start = d * block_len
                    data_to_decode[start:start + block_len] = stripe_blocks[d] # type: ignore
                else:
                    erasures.append(d)
            
            if len(erasures) > 2:
                continue
            
            try:
                k = num_data_disks
                m = 2
                w = 8
                matrix = pyjerasure.Matrix("reed_sol_r6_op", k, m, w)
                
                recovered = pyjerasure.decode(matrix, bytes(data_to_decode), block_len, erasures)
                
                for d in range(num_data_disks):
                    start = d * block_len
                    result[d][stripe] = recovered[start:start + block_len]
            except Exception as e:
                continue

    return result