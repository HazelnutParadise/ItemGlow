FROM python:3.12-slim
# 設定工作目錄
WORKDIR /app

# 複製需要的檔案
COPY . .

# 安裝 Python 套件
RUN pip3 install --no-cache-dir -r requirements.txt

# 容器啟動指令
CMD ["python", "wbeui.py"]
