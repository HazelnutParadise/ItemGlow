import os
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from rembg import remove
import cv2
from typing import Any
from tqdm import tqdm
import numpy as np
from white_balance import apply_multiple_white_balance
from numba import jit

# 建立線程池
executor = ThreadPoolExecutor()

SUPPORT_CUDA = False

# 檢查CUDA支援
print(f"CUDA支援狀態: {'可用' if (SUPPORT_CUDA := cv2.cuda.getCudaEnabledDeviceCount() > 0) else '不可用'}")

@jit(nopython=True)
def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image * factor, 0, 255).astype(np.uint8)

@jit(nopython=True)
def fill_white_background(result_img: np.ndarray, alpha_factor: np.ndarray) -> np.ndarray:
    white_background = np.ones_like(result_img, dtype=np.uint8) * 255
    for c in range(3):  # RGB 通道
        white_background[:, :, c] = np.clip(
            (1 - alpha_factor) * 255 + alpha_factor * result_img[:, :, c],
            0,
            255
        ).astype(np.uint8)
    return white_background

async def process_image(input_path: str, output_path: str) -> None:
    """
    非同步處理圖片：去背、白點法白平衡、適度提升飽和度、提高亮度，最後填充白色背景。
    自動決定使用 GPU 或 CPU 處理。
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"輸入檔案不存在: {input_path}")

        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 讀取圖片
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"無法讀取圖片: {input_path}")

        # 非同步讀取檔案
        async with aiofiles.open(input_path, "rb") as f:
            input_image = await f.read()

        # 在線程池中執行耗時的圖片處理
        loop = asyncio.get_event_loop()
        output_image = await loop.run_in_executor(executor, remove, input_image)
        image_np = await loop.run_in_executor(
            executor,
            lambda: cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        )

        # 在線程池中處理圖片
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA 圖片
            b, g, r, a = cv2.split(image_np)
            result_img = await loop.run_in_executor(
                executor,
                apply_multiple_white_balance,
                cv2.merge([b, g, r]),
                SUPPORT_CUDA
            )

            # 飽和度
            result_img = await loop.run_in_executor(
                executor,
                increase_saturation,
                result_img,
                1
            )
            
            # 調高亮度
            if SUPPORT_CUDA:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(result_img)
                gpu_img = cv2.cuda.multiply(gpu_img, 1.3)
                result_img = gpu_img.download()
            else:
                result_img = await loop.run_in_executor(
                    executor,
                    adjust_brightness,
                    result_img,
                    1.3
                )

            # 填充白色背景
            alpha_factor = a / 255.0
            white_background = await loop.run_in_executor(
                executor,
                fill_white_background,
                result_img,
                alpha_factor
            )

            await loop.run_in_executor(executor, cv2.imwrite, output_path, white_background)
        else:
            print("圖片格式錯誤，無法處理！")

    except Exception as e:
        print(f"處理圖片時發生錯誤 {input_path}: {str(e)}")
        raise

# 提升飽和度
def increase_saturation(image: Any, saturation_scale: float) -> Any:
    """
    提升飽和度：轉換到 HSV 色彩空間，增強飽和度 S 通道。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 提升飽和度
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_all_images(input_dir: str) -> list[str]:
    """遞迴搜尋所有圖片檔案"""
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append(os.path.join(root, file))
    return image_files

async def process_multiple_images(input_dir: str, output_dir: str) -> None:
    """批次處理多張圖片"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 取得所有圖片檔案
    image_files = get_all_images(input_dir)
    if not image_files:
        print("未找到任何圖片檔案")
        return

    print(f"找到 {len(image_files)} 個圖片檔案")
    
    # 建立進度條
    tasks = []
    with tqdm(total=len(image_files), desc="處理圖片") as pbar:
        for input_path in image_files:
            # 保持相同的目錄結構
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            task = asyncio.create_task(process_image(input_path, output_path))
            task.add_done_callback(lambda _: pbar.update(1))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

def main():
    input_dir = "input"
    output_dir = "output"
    
    try:
        asyncio.run(process_multiple_images(input_dir, output_dir))
        print("所有圖片處理完成")
    except Exception as e:
        print(f"程式執行錯誤: {str(e)}")

if __name__ == "__main__":
    main()
