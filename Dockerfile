# ======================================
# 🐍 Stage 1: Builder
# ======================================
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Các lib tối thiểu để build paddle/torch mà không bị lỗi
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    swig \
    python3-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libharfbuzz-dev \
    libjbig2dec0-dev \
    libleptonica-dev \
    libz-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    poppler-utils \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy file yêu cầu
COPY requirements.txt .

# ⚡ Giữ cache pip để lần build sau nhanh hơn
RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-deps --no-cache-dir paddleocr==2.8.1

# ======================================
# 🚀 Stage 2: Runtime
# ======================================
FROM --platform=$TARGETPLATFORM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_GPU_AUTO=true \
    LLM_CORRECTION_ENABLED=true \
    LLM_PROVIDER=groq \
    CLEANUP_AFTER_SECONDS=300 \
    LLM_TIMEOUT=30

# Lib runtime cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libgomp1 fonts-dejavu-core  curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy từ builder sang
COPY --from=builder /usr/local /usr/local
COPY main.py ./
RUN mkdir -p uploads output static

# Copy static files
COPY static/ ./static/

EXPOSE 5678

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:5678/ocr/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5678", "--workers", "1"]
