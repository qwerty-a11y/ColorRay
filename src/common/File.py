def FileToBinary(path:str):
    with open(path, 'rb') as file:
        return file.read()

def BinaryToFile(path:str, data:bytes):
    with open(path, 'xb') as file:
        file.write(data)
    return None