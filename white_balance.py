import cv2
import numpy as np
from typing import Any
from numba import jit, cuda

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

@cuda.jit
def adjust_channels_perfect_reflector_cuda(b, g, r, max_b, max_g, max_r):
    i, j = cuda.grid(2)
    if i < b.shape[0] and j < b.shape[1]:
        b[i, j] = min(b[i, j] * (255 / max_b), 255)
        g[i, j] = min(g[i, j] * (255 / max_g), 255)
        r[i, j] = min(r[i, j] * (255 / max_r), 255)

@cuda.jit
def adjust_channels_gray_world_cuda(b, g, r, avg_gray, avg_b, avg_g, avg_r):
    i, j = cuda.grid(2)
    if i < b.shape[0] and j < b.shape[1]:
        b[i, j] = min(b[i, j] * (avg_gray / avg_b), 255)
        g[i, j] = min(g[i, j] * (avg_gray / avg_g), 255)
        r[i, j] = min(r[i, j] * (avg_gray / avg_r), 255)

@cuda.jit
def adjust_channels_white_patch_cuda(b, g, r, max_b, max_g, max_r):
    i, j = cuda.grid(2)
    if i < b.shape[0] and j < b.shape[1]:
        b[i, j] = min(b[i, j] * (255 / max_b), 255)
        g[i, j] = min(g[i, j] * (255 / max_g), 255)
        r[i, j] = min(r[i, j] * (255 / max_r), 255)

@cuda.jit
def adjust_channels_adaptive_cuda(b, g, r, avg_gray, avg_b, avg_g, avg_r):
    i, j = cuda.grid(2)
    if i < b.shape[0] and j < b.shape[1]:
        b[i, j] = min(b[i, j] * (avg_gray / avg_b), 255)
        g[i, j] = min(g[i, j] * (avg_gray / avg_g), 255)
        r[i, j] = min(r[i, j] * (avg_gray / avg_r), 255)

# 灰度世界假設白平衡
def gray_world_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)
        avg_gray = (avg_b + avg_g + avg_r) / 3

        b_device = cuda.to_device(b)
        g_device = cuda.to_device(g)
        r_device = cuda.to_device(r)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(b.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(b.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        adjust_channels_gray_world_cuda[blockspergrid, threadsperblock](b_device, g_device, r_device, avg_gray, avg_b, avg_g, avg_r)

        b = b_device.copy_to_host().astype(np.uint8)
        g = g_device.copy_to_host().astype(np.uint8)
        r = r_device.copy_to_host().astype(np.uint8)

        result = cv2.merge([b, g, r])
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_gray_world(b, g, r)
        result = cv2.merge([b, g, r])

    return result

# 完美反射假設白平衡
def perfect_reflector_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        max_b = np.max(b)
        max_g = np.max(g)
        max_r = np.max(r)

        b_device = cuda.to_device(b)
        g_device = cuda.to_device(g)
        r_device = cuda.to_device(r)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(b.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(b.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        adjust_channels_perfect_reflector_cuda[blockspergrid, threadsperblock](b_device, g_device, r_device, max_b, max_g, max_r)

        b = b_device.copy_to_host().astype(np.uint8)
        g = g_device.copy_to_host().astype(np.uint8)
        r = r_device.copy_to_host().astype(np.uint8)

        result = cv2.merge([b, g, r])
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_perfect_reflector(b, g, r)
        result = cv2.merge([b, g, r])

    return result

# 白點法白平衡
def white_patch_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        max_b = np.max(b)
        max_g = np.max(g)
        max_r = np.max(r)

        b_device = cuda.to_device(b)
        g_device = cuda.to_device(g)
        r_device = cuda.to_device(r)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(b.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(b.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        adjust_channels_white_patch_cuda[blockspergrid, threadsperblock](b_device, g_device, r_device, max_b, max_g, max_r)

        b = b_device.copy_to_host().astype(np.uint8)
        g = g_device.copy_to_host().astype(np.uint8)
        r = r_device.copy_to_host().astype(np.uint8)

        result = cv2.merge([b, g, r])
    else:
        b, g, r = cv2.split(image)
        b, g, r = adjust_channels_white_patch(b, g, r)
        result = cv2.merge([b, g, r])

    return result

# 自適應白平衡
def adaptive_white_balance(image: Any, support_cuda: bool) -> Any:
    if support_cuda:
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)
        avg_gray = (avg_b + avg_g + avg_r) / 3

        b_device = cuda.to_device(b)
        g_device = cuda.to_device(g)
        r_device = cuda.to_device(r)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(b.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(b.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        adjust_channels_adaptive_cuda[blockspergrid, threadsperblock](b_device, g_device, r_device, avg_gray, avg_b, avg_g, avg_r)

        b = b_device.copy_to_host().astype(np.uint8)
        g = g_device.copy_to_host().astype(np.uint8)
        r = r_device.copy_to_host().astype(np.uint8)

        result = cv2.merge([b, g, r])
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