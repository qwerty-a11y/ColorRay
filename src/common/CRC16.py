
def _calculate_crc16(data: bytes) -> int:
    """
    计算数据的 CRC16 校验值 (CRC-16-IBM 标准)
    :param data: 输入数据
    :return: 16 位整数校验值
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def add_crc16(data: bytes) -> bytes:
    """
    在输入数据前面添加 2 字节的 CRC16 校验码
    :param data: 原始数据
    :return: 校验码 (2 字节，小端序) + 原始数据
    """
    crc_value = _calculate_crc16(data)
    # 将 CRC 值转换为 2 字节的小端序
    crc_bytes = crc_value.to_bytes(2, byteorder='little')
    return crc_bytes + data


def verify_crc16(data_with_crc: bytes) -> bool:
    """
    验证带校验码的数据是否通过校验
    :param data_with_crc: 前 2 字节为校验码，后续为原始数据
    :return: 校验是否通过
    """
    if len(data_with_crc) < 2:
        return False
    
    received_crc_bytes = data_with_crc[:2]
    original_data = data_with_crc[2:]
    
    # 将接收到的校验码转换为整数 (小端序)
    received_crc = int.from_bytes(received_crc_bytes, byteorder='little')
    
    # 计算原始数据的 CRC
    calculated_crc = _calculate_crc16(original_data)
    
    return received_crc == calculated_crc