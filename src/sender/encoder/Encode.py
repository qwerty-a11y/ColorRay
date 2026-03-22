from common.CorrectionLevel import RaidLevel,RSLevel
from common.File import FileToBinary

def Encode(path:str, raid:RaidLevel, rs:RSLevel):
    binary = FileToBinary(path)
    