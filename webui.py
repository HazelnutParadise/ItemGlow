import gradio as gr
import tempfile
import shutil
import zipfile
import asyncio
from pathlib import Path
from io import BytesIO
from main import process_multiple_images
import datetime
import os

async def process_files(files):
    """處理上傳的檔案並返回 ZIP"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # 處理上傳的檔案或資料夾
        for file in files:
            file_path = Path(file.name)
            if file_path.is_dir():
                # 複製整個資料夾結構
                shutil.copytree(
                    file_path, 
                    input_dir / file_path.name,
                    dirs_exist_ok=True
                )
            else:
                # 複製單一檔案
                shutil.copy2(file_path, input_dir)
        
        await process_multiple_images(str(input_dir), str(output_dir))
        
        # 建立 ZIP
        memory_zip = BytesIO()
        with zipfile.ZipFile(memory_zip, 'w') as zf:
            for file in output_dir.rglob('*'):
                if file.is_file():
                    zf.write(file, file.relative_to(output_dir))
        
        # 使用時間戳記建立有意義的檔名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'ItemGlow_{timestamp}.zip'
        
        # 在系統臨時目錄建立具名檔案
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, output_filename)
        
        with open(output_path, 'wb') as f:
            f.write(memory_zip.getvalue())
        
        return output_path

def launch_ui():
    with gr.Blocks() as app:
        gr.Markdown("## 圖片去背處理工具")
        with gr.Row():
            file_input = gr.File(
                label="拖曳上傳圖片或資料夾",
                file_types=["image"],
                file_count="multiple",
                type="filepath"
            )
            output = gr.File(label="下載處理後的圖片")
        
        file_input.upload(
            fn=lambda x: asyncio.run(process_files(x)),
            inputs=file_input,
            outputs=output
        )
    
    app.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_ui()