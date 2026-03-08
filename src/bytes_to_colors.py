import numpy as np
# 字节流和颜色流的双向转换，注意转换颜色流时会填充字节到3的倍数，而转换回字节流时不会处理填充字节

def bytes_to_colors(byte_data: bytes | bytearray) -> list[tuple[int, int, int]]:
    """
    将偶数个字节的二进制数据转换为RGB颜色数组
    规则：
    1. 字节按二进制位拆分，低位（第0位）对应数组靠前位置；
    2. 每3位二进制位转换为一个RGB颜色值；
    3. 单个字节不足3位时，与下一个字节的位拼接后再分组。
    
    :param byte_data: 输入的偶数个字节（bytes/bytearray类型）
    :return: RGB颜色值数组
    :raises TypeError: 输入类型不是bytes/bytearray
    """
    # 1. 输入类型校验
    if not isinstance(byte_data, (bytes, bytearray)):
        raise TypeError(f"输入类型必须是bytes/bytearray，当前为{type(byte_data)}")
    
    # 填充0字节至3的整数倍长度
    byte_len = len(byte_data)
    pad_count = (3 - (byte_len % 3)) % 3  # 计算需要填充的0字节数
    if pad_count > 0:
        byte_data_padded = byte_data + b'\x00' * pad_count  # 末尾填充0字节
        print(f"输入字节数{byte_len}，填充{pad_count}个0字节，新长度{len(byte_data_padded)}")
    else:
        byte_data_padded = byte_data
        print(f"输入字节数{byte_len}已是3的整数倍，无需填充")
    
    # 2. 展开所有字节为连续的二进制位列表（低位在前）
    bits = []
    for b in byte_data:
        # 遍历单个字节的0-7位（从低位到高位）
        for bit_pos in range(8):
            # 提取第bit_pos位的值（0或1）：右移bit_pos位后与1按位与
            current_bit = (b >> bit_pos) & 1
            bits.append(current_bit)
    # 4. 每3位分组，直接映射为RGB颜色（无中间数值数组）
    color_array = []
    for i in range(0, len(bits), 3):
        # 取当前3位，不足3位时末尾补0（保证RGB三通道）
        bit_group = bits[i:i+3]
        while len(bit_group) < 3:
            bit_group.append(0)
        
        # 映射规则：第1位→R，第2位→G，第3位→B；0→0，1→255
        r = bit_group[0] * 255
        g = bit_group[1] * 255
        b = bit_group[2] * 255
        color_array.append((r, g, b))
    
    return color_array

def colors_to_bytes(color_array: list[tuple[int, int, int]]) -> bytes:
    """
    将RGB颜色数组还原为原始字节数据（无需考虑填充）
    逆向规则：
    1. 每个RGB元组→3位二进制位：
       - R分量：<=127→0，>=128→1（对应255）
       - G分量：<=127→0，>=128→1（对应255）
       - B分量：<=127→0，>=128→1（对应255）
       顺序为R=第1位、G=第2位、B=第3位；
    2. 所有位按顺序拼接后，每8位一组还原为字节（低位在前）；
    3. 无需处理编码时的填充0字节，还原结果包含填充位（可自行截断）。
    
    :param color_array: RGB颜色元组列表（如[(255,0,0), (0,255,255), ...]）
    :return: 还原后的字节数据
    :raises TypeError: 输入不是列表，或列表元素不是3元组
    :raises ValueError: 元组元素不是0-255的整数
    """
    # 1. 输入合法性校验
    if not isinstance(color_array, list):
        raise TypeError(f"输入必须是RGB颜色元组列表，当前为{type(color_array)}")
    for idx, color in enumerate(color_array):
        if not isinstance(color, tuple) or len(color) != 3:
            raise TypeError(f"第{idx}个元素必须是3元组（R,G,B），当前为{color}")
        for c in color:
            if not isinstance(c, int) or c < 0 or c > 255:
                raise ValueError(f"颜色分量必须是0-255的整数，当前出现{c}")
    
    # 2. RGB颜色数组→连续二进制位列表（低位在前）
    bits = []
    for (r, g, b) in color_array:
        # 颜色分量判断规则（<=127→0，>=128→1）
        bit1 = 1 if r >= 128 else 0  # R→第1位
        bit2 = 1 if g >= 128 else 0  # G→第2位
        bit3 = 1 if b >= 128 else 0  # B→第3位
        bits.extend([bit1, bit2, bit3])
    
    # 3. 二进制位列表→字节数据（每8位一组，低位在前）
    byte_list = []
    # 按每8位分组处理（不足8位时丢弃，因原始字节是8位整数）
    for i in range(0, len(bits), 8):
        bit_group = bits[i:i+8]
        # 8位二进制→字节：bit_group[0]是2^0位（最低位），bit_group[7]是2^7位（最高位）
        byte_val = 0
        for idx, bit in enumerate(bit_group):
            byte_val += bit * (2 ** idx)
        byte_list.append(byte_val)
    
    # 4. 转换为bytes并返回
    return bytes(byte_list)
