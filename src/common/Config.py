from enum import Enum


QRSize = 133
FrameDataBlocks = QRSize * QRSize - 225*3 - 144 - 8 - 30
FrameDataSize = (FrameDataBlocks*3) // 8
FrameGroupCount = 8
GroupDataSize = FrameDataSize // FrameGroupCount
VideoFrameRate = 30

def find_optimal_rb(L_target, p_target):
    best_r = None
    best_b = None
    best_diff = float('inf')
    b_min = (L_target + 254) // 255
    b_max = (L_target - 1) // 255 + 1
    for b in range(b_min, b_max + 1):
        n_last = L_target - (b - 1) * 255
        r_ideal = p_target * L_target / b
        # 扩大搜索范围：±10
        for r in range(max(1, int(r_ideal) - 10), min(254, int(r_ideal) + 11)):
            # 只要求最后一块至少包含1字节原始数据
            if n_last >= r + 1 and n_last <= 255:
                p_actual = (b * r) / L_target
                diff = abs(p_actual - p_target)
                if diff < best_diff - 1e-12:
                    best_diff = diff
                    best_r = r
                    best_b = b
    return best_r, best_b

class RSCorrectionBytes(Enum):
    LEVEL1_5 = find_optimal_rb(GroupDataSize, 0.05)
    LEVEL2_10 = find_optimal_rb(GroupDataSize, 0.10)
    LEVEL3_15 = find_optimal_rb(GroupDataSize, 0.15)
    NONE = 0,0

    def __init__(self, r, b):
        self.r = r
        self.b = b
