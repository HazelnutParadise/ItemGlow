import os
from rembg import remove
import cv2
import numpy as np

# 強化邊緣平滑去背，填充白色背景
def enhanced_remove_background(input_path, output_path):
    # 讀取圖片
    with open(input_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)

    # 轉為 OpenCV 格式
    image_np = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)

    # 檢查是否有 Alpha 通道
    if image_np.shape[2] == 4:  # RGBA 格式
        alpha_channel = image_np[:, :, 3]
        white_background = np.ones_like(image_np) * 255
        white_background[:, :, :3] = image_np[:, :, :3]

        # 平滑邊緣處理 (高斯模糊)
        alpha_blur = cv2.GaussianBlur(alpha_channel, (7, 7), 0)
        alpha_mask = alpha_blur / 255.0

        # 融合白色背景
        for c in range(3):
            white_background[:, :, c] = (1 - alpha_mask) * 255 + alpha_mask * image_np[:, :, c]

        cv2.imwrite(output_path, white_background[:, :, :3])
    else:
        cv2.imwrite(output_path, image_np)

# 批次處理整個目錄的圖片
def batch_process_images(input_dir, output_dir, model_name="u2netp"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 建立輸出資料夾

    # 遍歷輸入目錄內的所有圖片檔案
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")

            print(f"處理圖片: {filename}")
            try:
                # 執行去背處理
                enhanced_remove_background(input_path, output_path)
                print(f"已完成: {output_path}")
            except Exception as e:
                print(f"處理失敗: {filename}, 錯誤訊息: {e}")

    print("所有圖片處理完成！")

# 主程式
if __name__ == "__main__":
    input_directory = "input"   # 輸入圖片目錄
    output_directory = "output"  # 輸出處理後圖片目錄
    
    batch_process_images(input_directory, output_directory)
