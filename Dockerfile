# Sử dụng base image nhẹ
FROM python:3.11-slim

# Thiết lập môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Cài các gói hệ thống cần thiết (cho OpenCV và EasyOCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy file dependency
COPY requirements.txt .

# Cài torch stack bản CPU (nhẹ, tương thích mọi hệ điều hành)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source
COPY . .

EXPOSE 5678

# Chạy ứng dụng bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5678"]
