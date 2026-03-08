import numpy as np
from reedsolo import RSCodec, ReedSolomonError

# Reed-Solomon 编码与解码函数

def rs_encode_bytes(raw_bytes: bytes, error_correction_symbols: int = 10) -> bytes:
    """
    对二进制字节流进行Reed-Solomon编码（添加冗余校验字节）
    :param raw_bytes: 原始二进制字节流（如b"hello"、文件读取的字节）
    :param error_correction_symbols: 冗余校验符号数（1个符号=1个字节），数值越大纠错能力越强
    :return: 带校验的编码后字节流
    """
    # 初始化RS编码器：指定冗余校验字节数（决定纠错能力）
    rsc = RSCodec(error_correction_symbols)
    # 对字节流编码：返回编码后的字节（原始字节 + 校验字节）
    encoded_bytes = rsc.encode(raw_bytes)
    return encoded_bytes

def rs_decode_bytes(encoded_bytes: bytes, error_correction_symbols: int = 10) -> bytes:
    """
    对带RS校验的字节流解码，恢复原始数据（可纠正错误）
    :param encoded_bytes: 编码后的字节流（含校验）
    :param error_correction_symbols: 需与编码时的冗余数一致
    :return: 恢复后的原始字节流
    """
    rsc = RSCodec(error_correction_symbols)
    try:
        # 解码：自动检测并纠正错误，返回原始字节
        decoded_bytes = rsc.decode(encoded_bytes)[0]
        return decoded_bytes
    except ReedSolomonError as e:
        # 错误超出纠错能力时抛出异常
        print(f"解码失败：错误超出纠错能力，异常信息：{e}")
        return b""
