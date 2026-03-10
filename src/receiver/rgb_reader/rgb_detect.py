import numpy as np
from collections import deque
from typing import Tuple, List

# 八色RGB值定义 (R, G, B)
COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}

class RGBDetector:
    COLOR_NAMES = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black']
    def __init__(self, buffer_size: int = 5, stability_threshold: float = 30.0, match_threshold: float = 60.0):
        """
        初始化检测器
        :param buffer_size: 滑动窗口大小
        :param stability_threshold: 防噪声的稳定性阈值
        :param match_threshold: 颜色匹配的距离阈值
        """
        self.buffer_size = buffer_size
        self.stability_threshold = stability_threshold
        self.match_threshold = match_threshold
        self.color_buffer: deque = deque(maxlen=buffer_size)
        # 生成标准色矩阵和名称数组
        self._target_matrix = np.array([COLOR_MAP[name] for name in self.COLOR_NAMES])
        self._color_names_arr = np.array(self.COLOR_NAMES)

    def denoise_sample(self, rgb: Tuple[int, int, int]) -> bool:
        self.color_buffer.append(rgb)
        if len(self.color_buffer) < self.buffer_size:
            return False
        buffer_array = np.array(self.color_buffer)
        variance = np.sum(np.var(buffer_array, axis=0))
        return variance < self.stability_threshold ** 2

    def detect_color(self, rgb: Tuple[int, int, int]) -> str:
        rgb_array = np.array(rgb)
        distances = np.linalg.norm(self._target_matrix - rgb_array, axis=1)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        if min_distance < self.match_threshold:
            return self._color_names_arr[min_idx]
        return 'unknown'

    def read_and_detect(self, rgb: Tuple[int, int, int]) -> str:
        if self.denoise_sample(rgb):
            return self.detect_color(rgb)
        return 'noisy'

    def reset(self):
        self.color_buffer.clear()


    def denoise_sample(self, rgb: Tuple[int, int, int]) -> bool:
        """
        防噪声机制：采样缓冲区
        :param rgb: RGB元组
        :return: 是否通过防噪声检查
        """
        self.color_buffer.append(rgb)
        if len(self.color_buffer) < self.buffer_size:
            return False
        buffer_array = np.array(self.color_buffer)
        variance = np.sum(np.var(buffer_array, axis=0))
        
        return variance < self.stability_threshold ** 2
    
    def detect_color(self, rgb: Tuple[int, int, int]) -> str:
        """
        [修改] 优化算法：移除 for 循环，采用向量化计算欧氏距离
        """
        rgb_array = np.array(rgb)
        distances = np.linalg.norm(self._target_matrix - rgb_array, axis=1)
        
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance < self.match_threshold:
            return self._color_names_arr[min_idx]
        return 'unknown'
    
    def read_and_detect(self, rgb: Tuple[int, int, int]) -> str:
        if self.denoise_sample(rgb):
            return self.detect_color(rgb)
        return 'noisy'
    
    def reset(self):
        self.color_buffer.clear()

if __name__ == "__main__":
    detector = RGBDetector()
    test_rgb = (5, 250, 10)
    print("开始采样...")
    for i in range(20):
        result = detector.read_and_detect(test_rgb)
        print(f"采样 {i+1}: {result}")