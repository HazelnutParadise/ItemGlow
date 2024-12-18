import os
from rembg import remove
import cv2
import numpy as np
from typing import Any

# 去背 -> 白點法白平衡 -> 適度對比度 -> 適度飽和度 -> 填充白色背景
def process_image(input_path: str, output_path: str) -> None:
    """
    去背、白點法白平衡、適度提升飽和度、柔化銳利感，最後填充白色背景。
    """
    with open(input_path, "rb") as f:
        input_image = f.read()

    # 去背
    output_image = remove(input_image)
    image_np = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)

    # 保留 Alpha 通道，白點法白平衡
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA 圖片
        b, g, r, a = cv2.split(image_np)
        result_img = white_patch_white_balance(cv2.merge([b, g, r]))

        # 適度飽和度調整
        result_img = increase_saturation(result_img, saturation_scale=1.6)
        
        # 調高亮度
        result_img = np.clip(result_img * 1.45, 0, 255).astype(np.uint8)

        # 填充白色背景
        alpha_factor = a / 255.0
        white_background = np.ones_like(result_img, dtype=np.uint8) * 255

        for c in range(3):  # RGB 通道
            white_background[:, :, c] = np.clip(
                (1 - alpha_factor) * 255 + alpha_factor * result_img[:, :, c],
                0,
                255
            ).astype(np.uint8)

        cv2.imwrite(output_path, white_background)
    else:
        print("圖片格式錯誤，無法處理！")

# 白點法白平衡
def white_patch_white_balance(image: Any) -> Any:
    """
    白點法白平衡：使用最亮點作為白色基準。
    """
    b, g, r = cv2.split(image)
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)

    b = np.clip(b * (255 / max_b), 0, 255).astype(np.uint8)
    g = np.clip(g * (255 / max_g), 0, 255).astype(np.uint8)
    r = np.clip(r * (255 / max_r), 0, 255).astype(np.uint8)
    
    # 削弱藍色跟綠色通道
    b = np.clip(b * 0.8, 0, 255).astype(np.uint8)
    g = np.clip(g * 0.85, 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])

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

# 批次處理圖片
def batch_process_images(input_dir: str, output_dir: str) -> None:
    """
    批次處理目錄下所有圖片。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")

            print(f"處理圖片: {filename}")
            try:
                process_image(input_path, output_path)
                print(f"已完成: {output_path}")
            except Exception as e:
                print(f"處理失敗: {filename}, 錯誤訊息: {e}")

    print("所有圖片處理完成！")

# 主程式
if __name__ == "__main__":
    def main() -> None:
        input_directory = "input"    # 輸入圖片目錄
        output_directory = "output"  # 輸出處理後圖片目錄

        batch_process_images(input_directory, output_directory)
    
    main()
