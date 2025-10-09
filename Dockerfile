# ======================================
# 🐍 Stage 1: Builder
# ======================================
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Cài torch (CPU build, GPU sẽ override ở runtime nếu có)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ======================================
# 🚀 Stage 2: Runtime
# ======================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    FORCE_CPU=false \
    FORCE_GPU=false \
    USE_GPU_AUTO=true \
    CLEANUP_AFTER_SECONDS=600 \
    LLM_CORRECTION_ENABLED=true \
    LLM_PROVIDER=groq \
    LLM_TIMEOUT=30

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libgomp1 git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY main.py .
COPY run_server.py .
COPY .env .
COPY static/ ./static/

RUN mkdir -p uploads output static

EXPOSE 5678

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5678/ocr/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5678", "--workers", "1"]
