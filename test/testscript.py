import os, sys
import cv2
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from src.receiver.decoder.image_to_matrix import image_to_matrix
from src.receiver.detector.test_dll_photo import process_frame
from src.sender.generator.frame_gen import generate_frame
from src.sender.generator.drawer import drawer

raw_img = cv2.imread(root+'/src/receiver/test4.png')
transformed_img = process_frame(raw_img, 133*11)
cv2.imwrite(root+'/src/receiver/test4_transformed.png', transformed_img)

frame, need_border = generate_frame()

decoded_matrix = image_to_matrix(root+'/src/receiver/test4_transformed.png', run_mode='normal')
drawer(decoded_matrix, need_border, 'test4_decoded.png', root+'/src/receiver')

decoded_matrix2 = image_to_matrix(root+'/src/receiver/test4_origin.png', run_mode='test')
drawer(decoded_matrix2, need_border, 'test4_decoded_origin.png', root+'/src/receiver')

def compare_with_frame_mask(matrix_actual, matrix_expected, frame_matrix, name_actual="实际", name_expected="预期"):
    """
    根据基础结构 frame 过滤，仅比较携带信息的格点。
    frame[r][c] 为 None 表示该位置是信息区（需要一致）。
    frame[r][c] 不为 None 表示该位置是基础结构（忽略一致性）。
    """
    rows = len(frame_matrix)
    cols = len(frame_matrix[0])
    
    # 维度预检查
    if len(matrix_actual) != rows or len(matrix_expected) != rows:
        print("❌ 错误：矩阵维度与基础结构 frame 不符！")
        return False

    info_diff_points = []
    total_info_cells = 0
    
    print(f"🔍 开始对比信息区 (忽略基础结构占位)... \n")

    for r in range(rows):
        for c in range(cols):
            # 只有当 frame 在该位置为 None 时，才进行核心信息比对
            if frame_matrix[r][c] is None:
                total_info_cells += 1
                color_act = matrix_actual[r][c]
                color_exp = matrix_expected[r][c]
                
                if color_act != color_exp:
                    info_diff_points.append({
                        "pos": (r, c),
                        "actual": color_act,
                        "expected": color_exp
                    })

    # 输出统计结果
    if not info_diff_points:
        print(f"✅ 信息区完全一致! (共比对 {total_info_cells} 个有效信息点)")
        return True
    else:
        diff_count = len(info_diff_points)
        error_rate = (diff_count / total_info_cells) * 100
        print(f"❌ 信息区发现不一致! 错误率: {error_rate:.2f}% ({diff_count}/{total_info_cells})")
        
        print(f"\n{'坐标 (R, C)':<12} | {name_actual:<18} | {name_expected:<18} | 错误类型")
        print("-" * 70)
        
        for diff in info_diff_points[:20]: # 展示前20个
            r, c = diff['pos']
            act = diff['actual']
            exp = diff['expected']
            
            # 简单的颜色语义映射（便于阅读输出）
            def color_name(rgb):
                mapping = {
                    (255, 0, 0): "红", (0, 255, 0): "绿", (0, 0, 255): "蓝",
                    (255, 255, 0): "黄", (255, 0, 255): "洋红", (0, 255, 255): "青",
                    (0, 0, 0): "黑", (255, 255, 255): "白"
                }
                return mapping.get(rgb, str(rgb))

            print(f"({r:>3}, {c:>3})    | {color_name(act):<18} | {color_name(exp):<18} | {color_name(exp)}误判为{color_name(act)}")
            
        if diff_count > 20:
            print(f"... 还有 {diff_count - 20} 个差异点未列出。")
            
        return False
    
if compare_with_frame_mask(decoded_matrix, decoded_matrix2, frame, "解码结果", "原始结果"):
    print("解码结果一致！")