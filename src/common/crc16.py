"""
CRC16 校验实现
支持常用的 CRC-16-CCITT (0x1021) 算法，初始值 0xFFFF。
"""

def crc16(data: bytes, poly: int = 0x1021, init_val: int = 0xFFFF) -> int:
    """
    计算输入字节流的 CRC16 校验值（CRC-16-CCITT 标准实现）
    :param data: 待校验的字节流
    :param poly: 多项式，默认 0x1021
    :param init_val: 初始值，默认 0xFFFF
    :return: 16 位无符号整数 CRC 校验值
    """
    crc = init_val
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

if __name__ == "__main__":
    file_path = "my_data.bin"  # 替换为你的文件路径
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        checksum = crc16(file_data)
        print(f"'{file_path}' 的CRC16-CCITT(0x1021) 校验值: {hex(checksum)}")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

    # test_data = b"123456789"
    # print(f"CRC16-CCITT(0x1021) of '123456789': {hex(crc16(test_data))}")