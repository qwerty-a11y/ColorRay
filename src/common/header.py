import struct
from typing import Tuple, List, Optional

def CalculateHeaderCRC8(Data: List[int]) -> int: 
    """ 
    Header 内置的CRC8校验码
    """ 
    CrcValue = 0x00
    for Byte in Data: 
        CrcValue ^= Byte
        for _ in range(8): 
            if CrcValue & 0x80: 
                CrcValue = (CrcValue << 1) ^ 0x07
            else: 
                CrcValue <<= 1
            CrcValue &= 0xFF
    return CrcValue

class FrameHeader: 
    """ 
    48-bit 稳健型数据头管理器
    落位方案：图像 Row 18, Column 2 开始的 2x24 像素区域
    """ 
    
    # 同步头：用于快速定位 Header 起始位置并确认读取方向是否正确
    SYNC_BYTE = 0xEB  # 二进制: 11101011

    def __init__(self, Mode: int, GroupId: int, InGroupId: int, PayloadLen: int, IsLast: bool): 
        """
        初始化 Header 字段
        :param Mode: 纠错模式 (2bit)
        :param GroupId: 组 ID (8bit) - 标识当前属于第几组 RAID
        :param InGroupId: 组内 ID (4bit) - 标识组内的第几帧
        :param PayloadLen: 线格式有效字节数 (16bit)；0xFFFF 表示满帧 DataSize（见 Config）
        :param IsLast: 终止帧标记 (1bit) - 是否为整个文件的最后一帧数据
        """
        self.Mode = Mode               
        self.GroupId = GroupId       
        self.InGroupId = InGroupId   
        self.PayloadLen = PayloadLen 
        self.IsLast = 1 if IsLast else 0 

    def ToBits(self) -> List[int]: 
        """ 
        [发送端使用] 将逻辑字段打包成 48 个二进制位 (用于绘制黑白块)
        打包顺序（40bit 数据 + 8bit CRC8 = 48bit）：
        Sync(8) + Mode(2) + GroupId(8) + InGroupId(4) + PayloadLen(16) + IsLast(1) + Reserved(1)
        """ 
        # 1. 拼接前 40 位核心逻辑数据 (留下最后 8 位给 CRC)
        # LSB 区：Rsv(1) + IsLast(1) + PayloadLen(16) << 2，避免 12bit 无法表示 >0xFFE 的线长
        CombinedValue = (self.SYNC_BYTE << 32) | \
                        ((self.Mode & 0x03) << 30) | \
                        ((self.GroupId & 0xFF) << 22) | \
                        ((self.InGroupId & 0x0F) << 18) | \
                        ((self.PayloadLen & 0xFFFF) << 2) | \
                        ((self.IsLast & 0x01) << 1)
        # bit0：预留 1 bit（当前恒 0）
        
        # 2. 将这 40 位转为 5 个字节计算 CRC8
        HeaderBytes = CombinedValue.to_bytes(5, 'big') 
        CheckSum = CalculateHeaderCRC8(list(HeaderBytes)) 
        
        # 3. 将 CRC8 合并到最后，形成完整的 48 位数值
        FinalValue = (CombinedValue << 8) | CheckSum
        
        # 4. 固定 48 位（勿用 bin+zfill，极端值下长度可能偏离 48）
        return [1 if c == "1" else 0 for c in f"{FinalValue & 0xFFFFFFFFFFFF:048b}"] 

    @classmethod
    def FromBits(cls, BitList: List[int]) -> Optional['FrameHeader']: 
        """ 
        [接收端使用] 从 48 个二进制位中还原 Header 对象
        包含自校验逻辑：如果 CRC 失败或同步头不对，返回 None
        """ 
        if len(BitList) < 48: return None
        
        # 1. 将 Bit 列表转回整数数值
        BitString = "".join(map(str, BitList)) 
        FinalValue = int(BitString, 2) 
        
        # 2. 剥离并校验 CRC8
        ReceivedCheckSum = FinalValue & 0xFF
        RawValue = FinalValue >> 8
        
        HeaderBytes = RawValue.to_bytes(5, 'big') 
        if CalculateHeaderCRC8(list(HeaderBytes)) != ReceivedCheckSum: 
            return None # 校验失败，此帧图像可能存在严重畸变
            
        # 3. 剥离并校验同步头 (Sync Byte)
        Sync = (RawValue >> 32) & 0xFF
        if Sync != cls.SYNC_BYTE: return None
        
        # 4. 解析各业务字段
        return cls( 
            Mode = (RawValue >> 30) & 0x03, 
            GroupId = (RawValue >> 22) & 0xFF, 
            InGroupId = (RawValue >> 18) & 0x0F, 
            PayloadLen = (RawValue >> 2) & 0xFFFF, 
            IsLast = bool((RawValue >> 1) & 0x01) 
        ) 

# ================= 物理落位函数 (用于和图像矩阵对接) =================

def WriteHeaderToGrid(ColorGrid: List[List[Tuple[int, int, int]]], HeaderObj: FrameHeader): 
    """ 
    [发送端]：在 Row 18, Col 2 开始的 2x24 区域绘制 48 个黑白块
    黑色 (0,0,0) 代表 0，白色 (255,255,255) 代表 1
    """ 
    HeaderBits = HeaderObj.ToBits() 
    for Index, BitValue in enumerate(HeaderBits): 
        # 计算坐标：共 2 行，每行 24 个点
        RowOffset = 18 + (Index // 24) 
        ColOffset = 2 + (Index % 24) 
        # 填充颜色
        ColorGrid[RowOffset][ColOffset] = (255, 255, 255) if BitValue == 1 else (0, 0, 0) 

def ReadHeaderFromGrid(WarpedGrid: List[List[Tuple[int, int, int]]]) -> Optional[FrameHeader]: 
    """ 
    [接收端]：从校正后的 137x137 矩阵中快速提取 48 位 Header
    这是性能优化的核心：先执行此函数，再决定是否进行后续 1.6 万个格子的色彩解析
    """ 
    ExtractedBits = [] 
    for Index in range(48): 
        RowOffset = 18 + (Index // 24) 
        ColOffset = 2 + (Index % 24) 
        
        # 提取像素色值
        PixelColor = WarpedGrid[RowOffset][ColOffset] 
        # 亮度阈值判定 (简单三通道求平均)
        Brightness = sum(PixelColor) / 3.0
        ExtractedBits.append(1 if Brightness > 128 else 0) 
        
    return FrameHeader.FromBits(ExtractedBits)
