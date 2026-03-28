from typing import List
from functools import reduce
import operator
from zfec import Encoder, Decoder  # 导入 zfec


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
    
    # 过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in data_disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []
        
    num_blocks = len(non_empty_disks[0])
    total_disks = num_data_disks + 1
    BATCH_SIZE = 8  # 每 8 个条带为一个单位

    # 检查所有非空磁盘块数一致，空盘忽略
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
    result_disks: List[List[bytes | None]] = [[None] * num_blocks for _ in range(total_disks)]

    # 按批次处理，每 8 个条带为一个单位
    for batch_start in range(0, num_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_blocks)
        
        for stripe in range(batch_start, batch_end):
            # 校验盘位置计算基于当前批次内的相对位置
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

    return result_disks  # type: ignore


def Raid5Decode(disks: List[List[bytes | None]]) -> List[List[bytes | None]]:
    """
    RAID-5 解码，以每 8 个条带为单位进行恢复。
    """
    if not disks:
        return []

    total_disks = len(disks)
    if total_disks < 2:
        raise ValueError("至少需要 2 个磁盘（1 个数据盘 +1 个校验盘）")

    # 过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []

    num_blocks = len(non_empty_disks[0])
    BATCH_SIZE = 8

    # 检查所有非空磁盘块数一致，空盘忽略
    for i, disk in enumerate(disks):
        if len(disk) == 0:
            continue
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 块数不一致")

    num_data_disks = total_disks - 1
    result: List[List[bytes | None]] = [[None] * num_blocks for _ in range(num_data_disks)]

    # 按批次处理
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
    RAID-6 编码（使用 zfec 库），以每 8 个条带为单位进行编码。
    """
    if not data_disks:
        return []

    num_data_disks = len(data_disks)
    
    # 过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in data_disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []
        
    num_blocks = len(non_empty_disks[0])
    total_disks = num_data_disks + 2
    BATCH_SIZE = 8

    # 检查所有非空磁盘块数一致，空盘忽略
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

    result_disks: List[List[bytes | None]] = [[None] * num_blocks for _ in range(total_disks)]

    # 按批次处理
    for batch_start in range(0, num_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_blocks)
        
        for stripe in range(batch_start, batch_end):
            data_blocks = [data_disks[d][stripe] for d in range(num_data_disks)]
            
            # 【修改】使用 zfec 替代 pyjerasure
            # zfec 要求：Encoder(k, m) 其中 k 是最少需要的块数，m 是总输出块数
            # 对于 RAID-6：k = num_data_disks，m = num_data_disks + 2（数据块 + 2个校验块）
            k = num_data_disks
            m = num_data_disks + 2  # 总块数 = 数据块数 + 2个校验块
            
            # 创建编码器
            encoder = Encoder(k, m)
            
            # zfec 编码：输入数据块列表，输出包含所有块的列表
            # 返回的列表包含前 k 个原始数据块和后 (m-k) 个校验块
            encoded_blocks = encoder.encode(data_blocks)
            
            # 提取校验块（最后 m-k 个，共 2 个用于 RAID-6）
            parity_blocks = encoded_blocks[k:]
            
            # 校验盘位置固定为最后两个
            p_disk = num_data_disks
            q_disk = num_data_disks + 1
            
            result_disks[p_disk][stripe] = parity_blocks[0]
            result_disks[q_disk][stripe] = parity_blocks[1]
            
            for d in range(num_data_disks):
                result_disks[d][stripe] = data_blocks[d]

    return result_disks  # type: ignore


def Raid6Decode(disks: List[List[bytes | None]]) -> List[List[bytes | None]]:
    """
    RAID-6 解码，以每 8 个条带为单位进行恢复。
    """
    if not disks:
        return []

    total_disks = len(disks)
    if total_disks < 3:
        raise ValueError("至少需要 3 个磁盘（1 个数据盘 + 2 个校验盘）")

    # 过滤出非空磁盘以确定基准块数，忽略空盘
    non_empty_disks = [disk for disk in disks if len(disk) > 0]
    
    if not non_empty_disks:
        return []

    num_blocks = len(non_empty_disks[0])
    BATCH_SIZE = 8

    # 检查所有非空磁盘块数一致，空盘忽略
    for i, disk in enumerate(disks):
        if len(disk) == 0:
            continue
        if len(disk) != num_blocks:
            raise ValueError(f"磁盘 {i} 块数不一致")

    num_data_disks = total_disks - 2
    result: List[List[bytes | None]] = [[None] * num_blocks for _ in range(num_data_disks)]

    # 按批次处理
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

            # 【修改】使用 zfec 替代 pyjerasure
            k = num_data_disks
            m = num_data_disks + 2  # 总块数 = 数据块数 + 2个校验块
            
            # 构建用于解码的块列表和索引列表
            # zfec 需要：所有存在的块（包括数据块和校验块）及其原始索引
            blocks_for_decode: List[bytes] = []
            block_indices: List[int] = []
            
            # 添加存在的数据块
            for i, blk in enumerate(data_blocks):
                if blk is not None:
                    blocks_for_decode.append(blk)
                    block_indices.append(i)  # 数据块索引 0 到 k-1
            
            # 添加存在的校验块（索引 k 和 k+1）
            if p_block is not None:
                blocks_for_decode.append(p_block)
                block_indices.append(k)  # P 校验块索引为 k
            
            if q_block is not None:
                blocks_for_decode.append(q_block)
                block_indices.append(k + 1)  # Q 校验块索引为 k+1
            
            # 检查是否有足够的块进行解码（至少需要 k 个块）
            if len(blocks_for_decode) < k:
                continue  # 无法恢复，块不够
            
            try:
                # zfec 的 decoder.decode() 要求：
                # - blocks_for_decode 中恰好包含 k 个块
                # - block_indices 对应这 k 个块在原始 m 个块中的位置
                # 如果有超过 k 个块，只使用前 k 个
                if len(blocks_for_decode) > k:
                    blocks_for_decode = blocks_for_decode[:k]
                    block_indices = block_indices[:k]
                
                # 创建解码器
                decoder = Decoder(k, m)
                
                # zfec 解码：输入存在的块列表和它们的索引，输出所有 k 个数据块
                recovered_blocks = decoder.decode(blocks_for_decode, block_indices)
                
                # 恢复结果（recovered_blocks 包含 k 个数据块）
                for d in range(num_data_disks):
                    result[d][stripe] = recovered_blocks[d]
                    
            except Exception as e:
                # 解码失败，跳过此条带
                continue

    return result
