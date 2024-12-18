# 使用 NVIDIA CUDA 的基礎映像
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# 安裝 Python 3.12 和必要的工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        python3.12 \
        python3.12-dev \
        python3-pip \
        && \
    rm -rf /var/lib/apt/lists/*
# 設定工作目錄
WORKDIR /app

# 複製需要的檔案
COPY . .

# 安裝 Python 套件
RUN pip3 install --no-cache-dir -r requirements-gpu.txt

# 設定環境變數
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 容器啟動指令
CMD ["python3", "wbeui.py"]
