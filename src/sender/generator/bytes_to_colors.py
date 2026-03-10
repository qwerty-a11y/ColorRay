import numpy as np

# 字节流转换为颜色流，注意转换颜色流时会填充字节到3的倍数

def bytes_to_colors(byte_data: bytes | bytearray) -> list[tuple[int, int, int]]:
    """
    将3的倍数个字节的二进制数据转换为RGB颜色数组
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


