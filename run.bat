@echo off
REM Script để chạy OCR Text Detection with LLM Correction trên Windows

echo 🚀 Starting OCR Text Detection with LLM Correction...

REM Kiểm tra virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
pip install --upgrade pip

REM Cài đặt dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Kiểm tra environment variables
if "%GROQ_API_KEY%"=="" if "%OPENAI_API_KEY%"=="" (
    echo ⚠️  WARNING: No LLM API keys found!
    echo    - For Groq: Set GROQ_API_KEY environment variable
    echo    - For OpenAI: Set OPENAI_API_KEY environment variable
    echo    LLM correction will be disabled.
)

REM Chạy ứng dụng
echo 🎯 Starting server...
echo    - API docs: http://localhost:5678/ocr/docs
echo    - Web UI: http://localhost:5678/
echo.
echo Press Ctrl+C to stop the server

uvicorn main:app --host 0.0.0.0 --port 5678 --reload

pause
