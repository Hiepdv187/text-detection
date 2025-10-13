import os
import io
import re
import uuid
import asyncio
import torch
import time
import traceback
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import httpx

# ===========================================
# ⚙️ Config & Init
# ===========================================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

use_gpu_auto = os.getenv("USE_GPU_AUTO", "true").lower() == "true"
force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
force_gpu = os.getenv("FORCE_GPU", "false").lower() == "true"
if force_cpu:
    USE_GPU = False
elif force_gpu:
    USE_GPU = True
else:
    USE_GPU = use_gpu_auto and torch.cuda.is_available()
LLM_CORRECTION_ENABLED = os.getenv("LLM_CORRECTION_ENABLED", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
CLEANUP_AFTER_SECONDS = int(os.getenv("CLEANUP_AFTER_SECONDS", "300"))  # 5 phút

app = FastAPI(title="OCR Service", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================
# 🧩 OCR Engines
# ===========================================
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU đầu tiên

ocr_paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=True)

vietocr_config = Cfg.load_config_from_name("vgg_transformer")
vietocr_config["device"] = "cuda" if USE_GPU else "cpu"
vietocr = Predictor(vietocr_config)

# ===========================================
# 🔧 Helper Functions
# ===========================================
def detect_language(text: str) -> str:
    """Nhận diện tiếng Việt"""
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]", text, re.IGNORECASE):
        return "vi"
    return "en"

def merge_paddle_results(results):
    return "\n".join([line[1][0] for line in results if len(line) >= 2])

async def llm_correct_text(text: str) -> str:
    """Gọi API LLM để sửa lỗi OCR"""
    if not LLM_CORRECTION_ENABLED or not LLM_API_KEY or len(text.strip()) < 10:
        return text

    prompt = f"""
    Bạn là chuyên gia sửa lỗi OCR. Hãy phục hồi chính tả, thêm dấu tiếng Việt nếu thiếu,
    và giữ nguyên các ký tự đặc biệt, số, đơn vị tiền tệ.
    Văn bản OCR gốc:
    {text}
    """

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "Bạn là chuyên gia xử lý văn bản OCR."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return text

async def cleanup_file(filepath: str, delay: int = CLEANUP_AFTER_SECONDS):
    """Tự xóa file sau X giây"""
    await asyncio.sleep(delay)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"🧹 Đã xóa file: {filepath}")
        except Exception as e:
            print(f"⚠️ Không thể xóa {filepath}: {e}")

# ===========================================
# 🚀 API Endpoints
# ===========================================
@app.get("/ocr/health")
async def health():
    return {
        "status": "ok",
        "gpu": USE_GPU,
        "llm": LLM_CORRECTION_ENABLED,
        "cleanup_after_seconds": CLEANUP_AFTER_SECONDS
    }

@app.post("/ocr")
async def ocr_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload ảnh -> Nhận dạng text -> Tự xóa sau 5 phút
    """
    file_id = str(uuid.uuid4())
    filepath = ""
    output_path = ""
    
    try:
        print(f"[DEBUG] Starting OCR processing for file: {file.filename}")
        
        # --- Lưu file tạm ---
        file_ext = os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
        
        print(f"[DEBUG] Saving file to: {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        file_content = await file.read()
        with open(filepath, "wb") as f:
            f.write(file_content)
        print(f"[DEBUG] File saved successfully: {os.path.exists(filepath)}")

        # --- Đọc ảnh ---
        print("[DEBUG] Opening image...")
        try:
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            print("[DEBUG] Image opened successfully")
        except Exception as img_err:
            print(f"[ERROR] Failed to open image: {str(img_err)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_err)}")

        # --- PaddleOCR ---
        print("[DEBUG] Running PaddleOCR...")
        try:
            paddle_result = ocr_paddle.ocr(image, cls=True)
            paddle_text = merge_paddle_results(paddle_result[0]) if paddle_result else ""
            print(f"[DEBUG] PaddleOCR completed. Detected text length: {len(paddle_text)}")
        except Exception as ocr_err:
            print(f"[ERROR] PaddleOCR failed: {str(ocr_err)}")
            raise HTTPException(status_code=500, detail=f"OCR processing error: {str(ocr_err)}")

        # --- VietOCR ---
        print("[DEBUG] Running language detection...")
        try:
            lang = detect_language(paddle_text)
            print(f"[DEBUG] Detected language: {lang}")
            
            if lang == "vi" or len(paddle_text.strip()) < 5:
                print("[DEBUG] Using VietOCR for Vietnamese text...")
                viet_text = vietocr.predict(image)
                raw_text = viet_text
                engine_used = "vietocr"
                print(f"[DEBUG] VietOCR completed. Text length: {len(raw_text)}")
            else:
                raw_text = paddle_text
                engine_used = "paddleocr"
                print("[DEBUG] Using PaddleOCR results directly")
        except Exception as lang_err:
            print(f"[ERROR] Language detection/processing failed: {str(lang_err)}")
            raise HTTPException(status_code=500, detail=f"Language processing error: {str(lang_err)}")

        # --- Hậu xử lý LLM ---
        print("[DEBUG] Running LLM correction...")
        try:
            corrected_text = await llm_correct_text(raw_text)
            print("[DEBUG] LLM correction completed")
        except Exception as llm_err:
            print(f"[WARNING] LLM correction failed, using raw text. Error: {str(llm_err)}")
            corrected_text = raw_text

        # --- Ghi file kết quả ---
        try:
            output_path = os.path.join(OUTPUT_DIR, f"{file_id}.txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(corrected_text)
            print(f"[DEBUG] Results saved to: {output_path}")
        except Exception as file_err:
            print(f"[ERROR] Failed to save results: {str(file_err)}")
            # Continue even if file save fails

        # --- Tự động dọn rác ---
        if background_tasks:
            if os.path.exists(filepath):
                background_tasks.add_task(cleanup_file, filepath)
            if output_path and os.path.exists(output_path):
                background_tasks.add_task(cleanup_file, output_path)

        return JSONResponse({
            "id": file_id,
            "filename": file.filename,
            "language": lang,
            "engine_used": engine_used,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "file_deleted_after_seconds": CLEANUP_AFTER_SECONDS
        })

    except HTTPException as http_err:
        print(f"[HTTP ERROR] {http_err.status_code}: {http_err.detail}")
        raise http_err
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def home():
    """Serve trang chủ với giao diện upload"""
    return FileResponse("static/index.html", media_type="text/html")

@app.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve các file static khác"""
    return FileResponse(f"static/{filename}")
