#!/bin/bash

# Script để chạy OCR Text Detection with LLM Correction

echo "🚀 Starting OCR Text Detection with LLM Correction..."

# Kiểm tra virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Cài đặt dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Kiểm tra environment variables
if [ -z "$GROQ_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: No LLM API keys found!"
    echo "   - For Groq: Set GROQ_API_KEY environment variable"
    echo "   - For OpenAI: Set OPENAI_API_KEY environment variable"
    echo "   LLM correction will be disabled."
fi

# Chạy ứng dụng
echo "🎯 Starting server..."
echo "   - API docs: http://localhost:5678/ocr/docs"
echo "   - Web UI: http://localhost:5678/"
echo ""
echo "Press Ctrl+C to stop the server"

uvicorn main:app --host 0.0.0.0 --port 5678 --reload
