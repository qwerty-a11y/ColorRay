from enum import Enum


class RSLevel(Enum):
    NONE = 0
    LEVEL1_5 = 1
    LEVEL2_10 = 2
    LEVEL3_15 = 3
    
class RaidLevel(Enum):
    NONE = 0
    LEVEL1_10 = 1
    LEVEL2_20 = 2
    LEVEL3_40 = 3