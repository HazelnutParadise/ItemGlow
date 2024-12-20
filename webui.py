import gradio as gr
import tempfile
import shutil
import zipfile
import asyncio
from pathlib import Path
from io import BytesIO
import datetime
import os
from main import process_multiple_images
from typing import Any

async def process_files(files: Any) -> str:
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
    with gr.Blocks(css="body{max-width: 800px;align-self: center;}") as app:
        app.title = "ItemGlow 商品照片優化器 - 榛果繽紛樂"
        app.head = "<meta name=\"description\" content=\"ItemGlow 是一個專為電商賣家設計的商品照優化器，可自動去背、調整色溫、飽和度、亮度，快速製作商品照片。\">"
        with gr.Row(variant="panel"):
            gr.Button("前往 榛果繽紛樂", link="https://hazelnut-paradise.com", icon="https://src.hazelnut-paradise.com/HazelnutParadise-logo.png")
            gr.Button("更多 Web Apps", link="https://apps.hazelnut-paradise.com")
        
        gr.HTML(
            "<div style=\"margin: 0;padding: 0;display: flex; flex-direction: column; align-items: center;\">"
                "<div style=\"margin-bottom: 20px;\">"
                    "<h1 style=\"margin: 0;padding: 0;text-align: center;font-size: clamp(50px, 7vw, 100px);word-wrap: break-word;max-width: 100%;\">ItemGlow</h1>"
                    "<h2 style=\"margin: 0;padding: 0;margin-bottom: 50px;text-align: center;font-size: clamp(30px, 3vw, 40px);\">商品照片優化器</h2>"
                    "<h3 style=\"text-align: center;font-size: clamp(20px, 2vw, 40px);\"><i>～讓您的商品閃閃發光～</i></h3>"
                "</div>"
                "<img src=\"https://src.hazelnut-paradise.com/ItemGlow-logo.png\" style=\"height: auto;width:clamp(150px, 100%, 200px);\">"
            "</div>"
        ),
        gr.Button("View on GitHub", link="https://github.com/HazelnutParadise/ItemGlow", variant="huggingface"),
        gr.HTML("""
<div id="div-onead-draft"></div>

        """),
        gr.HTML(
            "<div style=\"margin: 0;\">"
                "<h3 style=\"text-align: start;font-size: clamp(20px, 2vw, 40px);\">專屬於電商賣家</h3>"
                "<p style=\"text-align: start;font-size: clamp(16px, 2vw, 20px);\">ItemGlow 可自動將商品照片去背，並調整至合適的色溫、飽和度、亮度，幫助電商賣家輕鬆製作商品照片。</p>"
            "</div>"
        )
        
        with gr.Row():
            file_input = gr.File(
                label="拖曳上傳照片",
                file_types=["image", ".webp"],
                file_count="multiple",
                type="filepath"
            )
            output = gr.File(label="下載處理後的照片")
        
        file_input.upload(
            fn=lambda x: asyncio.run(process_files(x)),
            inputs=file_input,
            outputs=output
        )
    
    app.launch(server_name="0.0.0.0", server_port=7860, favicon_path="favicon.ico")

if __name__ == "__main__":
    launch_ui()