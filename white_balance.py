import cv2
import numpy as np
from typing import Any

# 灰度世界假設白平衡
def gray_world_white_balance(image: Any) -> Any:
    b, g, r = cv2.split(image)
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3

    b = np.clip(b * (avg_gray / avg_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg_gray / avg_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg_gray / avg_r), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])

# 完美反射假設白平衡
def perfect_reflector_white_balance(image: Any) -> Any:
    b, g, r = cv2.split(image)
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)

    b = np.clip(b * (255 / max_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (255 / max_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (255 / max_r), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])

# 白點法白平衡
def white_patch_white_balance(image: Any) -> Any:
    b, g, r = cv2.split(image)
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)

    b = np.clip(b * (255 / max_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (255 / max_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (255 / max_r), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])

# 自適應白平衡
def adaptive_white_balance(image: Any) -> Any:
    """
    自適應白平衡：根據圖像的特徵動態調整白平衡。
    """
    result = image.copy()
    b, g, r = cv2.split(result)

    # 計算每個通道的平均值
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)

    # 計算整體平均值
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # 調整每個通道
    b = np.clip(b * (avg_gray / avg_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg_gray / avg_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg_gray / avg_r), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])

def brighten_shadows(image: Any, threshold: int = 60, factor: float = 1.5) -> Any:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 增亮暗部
    mask = v < threshold
    v[mask] = np.clip(v[mask] * factor, 0, 255)
    
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# 使用多種白平衡算法多次處理並調整暗部亮度
def apply_multiple_white_balance(image: Any) -> Any:
    image = white_patch_white_balance(image)
    image = gray_world_white_balance(image)
    image = perfect_reflector_white_balance(image)
    image = adaptive_white_balance(image)
    image = brighten_shadows(image, 80, 1.2)
    return image