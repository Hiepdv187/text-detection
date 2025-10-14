# ======================================
# 🐍 Multi-platform OCR Service
# ======================================
FROM python:3.11-slim

# Build arguments for multi-platform
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime environment variables
ENV USE_GPU_AUTO=true \
    LLM_CORRECTION_ENABLED=true \
    LLM_PROVIDER=groq \
    CLEANUP_AFTER_SECONDS=300 \
    LLM_TIMEOUT=30

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    python3-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libharfbuzz-dev \
    libz-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py ./
COPY static/ ./static/
RUN mkdir -p uploads output

EXPOSE 5678

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:5678/ocr/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5678", "--workers", "1"]
