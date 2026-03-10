import json
import os

# --- 默认配置参数 ---
DEFAULT_CONFIG = {
    "QR_VERSION": 1,
    "COLOR_MAP": [
        [0, 0, 0],      # 黑
        [0, 0, 255],    # 蓝
        [0, 255, 0],    # 绿
        [0, 255, 255],  # 青
        [255, 0, 0],    # 红
        [255, 0, 255],  # 品红
        [255, 255, 0],  # 黄
        [255, 255, 255] # 白
    ],
    "CELL_PIXELS": 12,
    "IMG_SIZE": 1644,
    "RS_ERROR_CORRECTION_SYMBOLS": 10,
    "DEBUG": False
}

def load_config(config_path="config.json"):
    """从文件读取配置，如果文件不存在则返回默认配置"""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                full_config = {**DEFAULT_CONFIG, **user_config}
                return full_config
        except Exception as e:
            print(f"读取配置文件失败，使用默认值。错误: {e}")
    return DEFAULT_CONFIG

_current_config = load_config()

QR_VERSION = _current_config["QR_VERSION"]
COLOR_MAP = [tuple(c) for c in _current_config["COLOR_MAP"]]
CELL_PIXELS = _current_config["CELL_PIXELS"]
IMG_SIZE = _current_config["IMG_SIZE"]
GRID_COUNT = IMG_SIZE // CELL_PIXELS
RS_ERROR_CORRECTION_SYMBOLS = _current_config["RS_ERROR_CORRECTION_SYMBOLS"]
DEBUG = _current_config["DEBUG"]

def save_default_config(path="config.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
    print(f"默认配置已保存至: {path}")

if __name__ == "__main__":
    print("--- 当前加载的配置参数 ---")
    print(f"QR_VERSION: {QR_VERSION}")
    print(f"CELL_PIXELS: {CELL_PIXELS}, GRID_COUNT: {GRID_COUNT}")
    print(f"DEBUG 模式: {DEBUG}")
    
    # save_default_config()