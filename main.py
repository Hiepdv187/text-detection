import os
import re
import uuid
import asyncio
import torch
import time
import traceback
import io
import numpy as np
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import httpx
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
# 🧩 OCR Engines - Khởi tạo lazy
# ===========================================
_ocr_engines_initialized = False
ocr_paddle = None
vietocr = None

def initialize_ocr_engines():
    """Khởi tạo các OCR engines một lần duy nhất"""
    global _ocr_engines_initialized, ocr_paddle, vietocr

    if _ocr_engines_initialized:
        return

    try:
        if not USE_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU đầu tiên

        print("[DEBUG] Initializing PaddleOCR...")
        ocr_paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=True)
        print(f"[DEBUG] PaddleOCR initialized. Models will be downloaded to: {os.path.expanduser('~/.paddleocr/')}")

        print("[DEBUG] Initializing VietOCR...")
        vietocr_config = Cfg.load_config_from_name("vgg_transformer")
        vietocr_config["device"] = "cuda" if USE_GPU else "cpu"
        vietocr = Predictor(vietocr_config)
        print("[DEBUG] VietOCR initialized successfully")

        _ocr_engines_initialized = True
        print("[DEBUG] All OCR engines initialized successfully")

    except Exception as e:
        print(f"[ERROR] Failed to initialize OCR engines: {e}")
        raise

# Khởi tạo ngay khi import (nhưng được bảo vệ)
try:
    initialize_ocr_engines()
except Exception as e:
    print(f"[WARNING] OCR engines initialization failed, will retry later: {e}")
    _ocr_engines_initialized = False

def ensure_ocr_engines():
    """Đảm bảo OCR engines đã được khởi tạo"""
    if not _ocr_engines_initialized:
        print("[DEBUG] OCR engines not initialized, initializing now...")
        initialize_ocr_engines()

# ===========================================
# 🔧 Helper Functions
# ===========================================
def detect_language(text: str) -> str:
    """Nhận diện tiếng Việt cải tiến"""
    text_lower = text.lower()
    
    # Kiểm tra ký tự đặc biệt tiếng Việt
    viet_chars = r"[àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]"
    if re.search(viet_chars, text, re.IGNORECASE):
        return "vi"
    
    # Từ vựng tiếng Việt phổ biến (không dấu)
    viet_words = ["la", "va", "vao", "ra", "di", "den", "tu", "voi", "cho", "nguoi", "nam", "nuoc", "thi", "nay", "hoac", "nhung", "nhieu", "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi"]
    viet_word_count = sum(1 for word in viet_words if word in text_lower)
    
    # Kiểm tra tỷ lệ ký tự tiếng Việt (dựa trên mẫu)
    total_chars = len(text)
    if total_chars > 10:  # Chỉ áp dụng cho văn bản dài
        viet_char_count = sum(1 for c in text if c.lower() in "àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ")
        viet_ratio = viet_char_count / total_chars
        if viet_ratio > 0.1 or viet_word_count > 2:  # Ngưỡng phát hiện
            return "vi"
    
    return "en"

def merge_paddle_results(results):
    return "\n".join([line[1][1] for line in results if len(line) >= 2 and len(line[1]) >= 2])

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

async def cleanup_old_files():
    """Xóa tất cả file cũ trong uploads và output (tồn tại quá CLEANUP_AFTER_SECONDS)"""
    current_time = time.time()
    cleaned_count = 0

    # Kiểm tra thư mục uploads
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > CLEANUP_AFTER_SECONDS:
                try:
                    os.remove(filepath)
                    print(f"🧹 Đã xóa file cũ: {filepath}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Không thể xóa {filepath}: {e}")

    # Kiểm tra thư mục output
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > CLEANUP_AFTER_SECONDS:
                try:
                    os.remove(filepath)
                    print(f"🧹 Đã xóa file cũ: {filepath}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Không thể xóa {filepath}: {e}")

    return cleaned_count

async def cleanup_file(filepath: str):
    """Xóa file sau một khoảng thời gian"""
    await asyncio.sleep(CLEANUP_AFTER_SECONDS)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🧹 Đã xóa file: {filepath}")
    except Exception as e:
        print(f"⚠️ Không thể xóa {filepath}: {e}")

# ===========================================
# 🚀 API Endpoints
# ===========================================
@app.get("/ocr/cleanup")
async def get_cleanup_status():
    """Kiểm tra trạng thái các file trong uploads và output"""
    current_time = time.time()
    files_info = {
        "uploads": [],
        "output": []
    }

    # Kiểm tra thư mục uploads
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            files_info["uploads"].append({
                "filename": filename,
                "age_seconds": int(file_age),
                "is_old": file_age > CLEANUP_AFTER_SECONDS,
                "size": os.path.getsize(filepath)
            })

    # Kiểm tra thư mục output
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            files_info["output"].append({
                "filename": filename,
                "age_seconds": int(file_age),
                "is_old": file_age > CLEANUP_AFTER_SECONDS,
                "size": os.path.getsize(filepath)
            })

    old_files_count = sum(1 for file in files_info["uploads"] if file["is_old"]) + \
                     sum(1 for file in files_info["output"] if file["is_old"])

    return JSONResponse({
        "cleanup_after_seconds": CLEANUP_AFTER_SECONDS,
        "old_files_count": old_files_count,
        "files": files_info
    })

@app.post("/ocr/cleanup")
async def manual_cleanup():
    """Kiểm tra và xóa tất cả file cũ (tồn tại quá 5 phút)"""
    try:
        cleaned_count = await cleanup_old_files()
        return JSONResponse({
            "status": "success",
            "message": f"Đã xóa {cleaned_count} file cũ",
            "cleanup_after_seconds": CLEANUP_AFTER_SECONDS
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

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

        # Đảm bảo OCR engines đã được khởi tạo
        ensure_ocr_engines()

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
            image = np.array(image)  # Chuyển đổi sang numpy array cho PaddleOCR
            print("[DEBUG] Image opened and converted successfully")
        except Exception as img_err:
            print(f"[ERROR] Failed to open image: {str(img_err)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_err)}")

        # --- PaddleOCR ---
        print("[DEBUG] Running PaddleOCR...")
        try:
            if ocr_paddle is None:
                raise Exception("PaddleOCR not initialized")
            paddle_result = ocr_paddle.ocr(image, cls=True)
            if paddle_result is None:
                print("[ERROR] PaddleOCR returned None")
                raise Exception("PaddleOCR returned None result")

            print(f"[DEBUG] PaddleOCR raw result: {paddle_result}")
            if paddle_result[0]:
                print(f"[DEBUG] First few results: {paddle_result[0][:3]}")

            paddle_text = merge_paddle_results(paddle_result[0]) if paddle_result[0] else ""
            print(f"[DEBUG] PaddleOCR completed. Detected text length: {len(paddle_text)}")
            print(f"[DEBUG] PaddleOCR detected text: '{paddle_text}'")
        except Exception as ocr_err:
            print(f"[ERROR] PaddleOCR failed: {str(ocr_err)}")
            print(f"[ERROR] PaddleOCR error type: {type(ocr_err)}")
            print(f"[ERROR] PaddleOCR traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"OCR processing error: {str(ocr_err)}")

        # --- VietOCR ---
        print("[DEBUG] Running language detection...")
        try:
            lang = detect_language(paddle_text)
            print(f"[DEBUG] Detected language: {lang}")
            print(f"[DEBUG] Sample text from PaddleOCR: '{paddle_text[:100]}...'")  # In mẫu văn bản để kiểm tra
            
            if lang == "vi" or len(paddle_text.strip()) < 5:
                print("[DEBUG] Using VietOCR for Vietnamese text...")
                if vietocr is None:
                    raise Exception("VietOCR not initialized")
                # Convert numpy array back to PIL Image for VietOCR
                pil_image = Image.fromarray(image)
                viet_text = vietocr.predict(pil_image)
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

@app.get("/ocr/health")
async def health_check():
    """Health check endpoint for Docker"""
    return JSONResponse({
        "status": "healthy",
        "ocr_engines_initialized": _ocr_engines_initialized,
        "gpu_available": torch.cuda.is_available(),
        "using_gpu": USE_GPU
    })

@app.get("/")
async def home():
    """Serve trang chủ với giao diện upload"""
    return FileResponse("static/index.html", media_type="text/html")

@app.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve các file static khác"""
    return FileResponse(f"static/{filename}")
