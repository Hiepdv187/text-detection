# Lightweight Python base image
FROM python:3.11-slim

# Avoid interactive prompts and enable UTF-8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5678 \
    HOST=0.0.0.0

# Install OS packages required by EasyOCR/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt


COPY main.py ./

COPY static/ ./static/

RUN mkdir -p uploads output

EXPOSE 5678

CMD ["sh", "-c", "uvicorn main:app --host $HOST --port ${PORT:-5678}"]
