import gradio as gr
import tempfile
import shutil
import zipfile
import asyncio
from pathlib import Path
from io import BytesIO
from main import process_multiple_images

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
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.write(memory_zip.getvalue())
        temp_zip.close()
        
        return temp_zip.name

def launch_ui():
    with gr.Blocks() as app:
        gr.Markdown("## 圖片去背處理工具")
        with gr.Row():
            file_input = gr.File(
                label="拖曳上傳圖片或資料夾",
                file_types=["image", "directory"],
                file_count="directory"
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