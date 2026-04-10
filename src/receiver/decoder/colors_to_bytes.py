import numpy as np

# 颜色流转字节流，需要注意还原后不会处理填充字节

def colors_to_bytes(color_array: list[int]) -> bytes:
    
    # 2. RGB颜色数组→连续二进制位列表（低位在前）
    # 注意：输入的 color_array 元素应为 0-7 的颜色索引
    # 根据 numpy_to_int 的定义：
    # Bit 0 (LSB): Blue
    # Bit 1:       Green
    # Bit 2 (MSB): Red
    bits = []
    for val in color_array:
        # 提取 R, G, B 位
        r_bit = (val >> 2) & 1
        g_bit = (val >> 1) & 1
        b_bit = val & 1
        
        # 按照 R, G, B 的顺序加入位列表
        # 如果协议要求其他顺序，请调整此处 append 的顺序
        bits.extend([r_bit, g_bit, b_bit])
    
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