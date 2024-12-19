import cv2
import numpy as np
from typing import Any
from numba import jit

@jit(nopython=True)
def adjust_channels_gray_world(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3

    b = np.clip(b * (avg_gray / avg_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg_gray / avg_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg_gray / avg_r), 0, 255).astype(np.uint8)

    return b, g, r

@jit(nopython=True)
def adjust_channels_perfect_reflector(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)

    b = np.clip(b * (255 / max_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (255 / max_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (255 / max_r), 0, 255).astype(np.uint8)

    return b, g, r

@jit(nopython=True)
def adjust_channels_white_patch(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)

    b = np.clip(b * (255 / max_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (255 / max_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (255 / max_r), 0, 255).astype(np.uint8)

    return b, g, r

@jit(nopython=True)
def adjust_channels_adaptive(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3

    b = np.clip(b * (avg_gray / avg_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg_gray / avg_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg_gray / avg_r), 0, 255).astype(np.uint8)

    return b, g, r

# 灰度世界假設白平衡
def gray_world_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        b, g, r = cv2.cuda.split(gpu_image)
        avg_b = cv2.cuda.mean(b)[0]
        avg_g = cv2.cuda.mean(g)[0]
        avg_r = cv2.cuda.mean(r)[0]
        avg_gray = (avg_b + avg_g + avg_r) / 3

        b = cv2.cuda.multiply(b, avg_gray / avg_b)
        g = cv2.cuda.multiply(g, avg_gray / avg_g)
        r = cv2.cuda.multiply(r, avg_gray / avg_r)

        gpu_result = cv2.cuda.merge([b, g, r])
        result = gpu_result.download()
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_gray_world(b, g, r)
        result = cv2.merge([b, g, r])

    return result

# 完美反射假設白平衡
def perfect_reflector_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        b, g, r = cv2.cuda.split(gpu_image)
        max_b = cv2.cuda.minMaxLoc(b)[1]
        max_g = cv2.cuda.minMaxLoc(g)[1]
        max_r = cv2.cuda.minMaxLoc(r)[1]

        b = cv2.cuda.multiply(b, 255 / max_b)
        g = cv2.cuda.multiply(g, 255 / max_g)
        r = cv2.cuda.multiply(r, 255 / max_r)

        gpu_result = cv2.cuda.merge([b, g, r])
        result = gpu_result.download()
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_perfect_reflector(b, g, r)
        result = cv2.merge([b, g, r])

    return result

# 白點法白平衡
def white_patch_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        b, g, r = cv2.cuda.split(gpu_image)
        max_b = cv2.cuda.minMaxLoc(b)[1]
        max_g = cv2.cuda.minMaxLoc(g)[1]
        max_r = cv2.cuda.minMaxLoc(r)[1]

        b = cv2.cuda.multiply(b, 255 / max_b)
        g = cv2.cuda.multiply(g, 255 / max_g)
        r = cv2.cuda.multiply(r, 255 / max_r)

        gpu_result = cv2.cuda.merge([b, g, r])
        result = gpu_result.download()
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_white_patch(b, g, r)
        result = cv2.merge([b, g, r])

    return result

# 自適應白平衡
def adaptive_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        b, g, r = cv2.cuda.split(gpu_image)
        avg_b = cv2.cuda.mean(b)[0]
        avg_g = cv2.cuda.mean(g)[0]
        avg_r = cv2.cuda.mean(r)[0]
        avg_gray = (avg_b + avg_g + avg_r) / 3

        b = cv2.cuda.multiply(b, avg_gray / avg_b)
        g = cv2.cuda.multiply(g, avg_gray / avg_g)
        r = cv2.cuda.multiply(r, avg_gray / avg_r)

        gpu_result = cv2.cuda.merge([b, g, r])
        result = gpu_result.download()
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_adaptive(b, g, r)
        result = cv2.merge([b, g, r])

    return result

def brighten_shadows(image: Any, threshold: int = 60, factor: float = 1.5) -> Any:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 增亮暗部
    mask = v < threshold
    v[mask] = np.clip(v[mask] * factor, 0, 255)
    
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 使用多種白平衡算法多次處理並調整暗部亮度
def apply_multiple_white_balance(image: Any, support_cuda: bool) -> Any:
    image = white_patch_white_balance(image, support_cuda)
    image = gray_world_white_balance(image, support_cuda)
    image = perfect_reflector_white_balance(image, support_cuda)
    image = adaptive_white_balance(image, support_cuda)
    # image = brighten_shadows(image, 80, 1.2)
    return image