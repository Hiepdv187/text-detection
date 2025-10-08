from paddleocr import PaddleOCR
import os
import asyncio
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import torch
import httpx
from dotenv import load_dotenv
from datetime import datetime

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
STATIC_DIR = "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

tags_metadata = [
    {"name": "OCR", "description": "Các API nhận diện văn bản (upload ảnh, health check)."},
    {"name": "Web", "description": "Trang web giao diện test (HTML)."},
]

app = FastAPI(
    title="OCR Text Recognition API with LLM Correction (PaddleOCR)",
    description="Nhận diện văn bản từ ảnh (Python + PaddleOCR) với tính năng tự động sửa lỗi bằng LLM",
    version="3.0.0",
    openapi_tags=tags_metadata,
    docs_url="/ocr/docs",
    redoc_url="/ocr/redoc",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
load_dotenv()

CLEANUP_AFTER_SECONDS = int(os.getenv("CLEANUP_AFTER_SECONDS", "600"))


async def delete_file_after(path: str, delay_seconds: int = CLEANUP_AFTER_SECONDS) -> None:
    try:
        await asyncio.sleep(max(1, int(delay_seconds)))
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def cleanup_expired_in_dir(dir_path: str, ttl_seconds: int = CLEANUP_AFTER_SECONDS) -> int:
    removed = 0
    try:
        if not os.path.isdir(dir_path):
            return 0
        now = datetime.now().timestamp()
        for name in os.listdir(dir_path):
            path = os.path.join(dir_path, name)
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > ttl_seconds:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return removed


@app.on_event("startup")
async def startup_cleanup():
    cleanup_expired_in_dir(UPLOAD_DIR, CLEANUP_AFTER_SECONDS)
    cleanup_expired_in_dir(OUTPUT_DIR, CLEANUP_AFTER_SECONDS)


# ============================================
# 🔧 PaddleOCR setup (auto CPU / GPU detection)
# ============================================
force_cpu = os.getenv("FORCE_CPU", "").lower() == "true"
force_gpu = os.getenv("FORCE_GPU", "").lower() == "true"
use_gpu_auto = os.getenv("USE_GPU_AUTO", "true").lower() == "true"

USE_GPU = False
reason = "auto"

if force_cpu:
    USE_GPU = False
    reason = "forced_cpu"
elif force_gpu:
    try:
        has_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        USE_GPU = bool(has_cuda or has_mps)
        reason = "forced_gpu_available" if USE_GPU else "forced_gpu_unavailable_fallback_cpu"
    except Exception:
        USE_GPU = False
        reason = "forced_gpu_error_fallback_cpu"
elif use_gpu_auto:
    try:
        has_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        USE_GPU = bool(has_cuda or has_mps)
        reason = "auto_gpu" if USE_GPU else "auto_cpu"
    except Exception:
        USE_GPU = False
        reason = "auto_error_cpu"
else:
    USE_GPU = False
    reason = "env_disabled_cpu"

print(f"[PaddleOCR] Device: {'GPU' if USE_GPU else 'CPU'} (mode={reason})")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',  # PaddleOCR không có sẵn model 'vi', nhưng tiếng Việt đọc được khá tốt
    use_gpu=USE_GPU
)

# ==========================
# ⚙️ LLM correction settings
# ==========================
LLM_ENABLED = os.getenv("LLM_CORRECTION_ENABLED", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not LLM_MODEL:
    LLM_MODEL = "gpt-4o-mini" if LLM_PROVIDER == "openai" else "llama-3.3-70b-versatile"

if LLM_ENABLED:
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("[LLM] WARNING: OpenAI API key not found, disabling correction")
        LLM_ENABLED = False
    elif LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        print("[LLM] WARNING: Groq API key not found, disabling correction")
        LLM_ENABLED = False

print(f"[LLM] {'Enabled' if LLM_ENABLED else 'Disabled'} with provider={LLM_PROVIDER}, model={LLM_MODEL}")


async def correct_text_with_llm(text: str) -> Dict[str, any]:
    result = {"original_text": text, "corrected_text": None, "corrected": False, "provider": None}
    if not LLM_ENABLED or not text.strip():
        return result

    system_prompt = """Bạn là một trợ lý AI chuyên sửa lỗi văn bản từ OCR (Optical Character Recognition).

Nhiệm vụ:
1. Sửa lỗi chính tả và ký tự OCR.
2. Giữ nguyên ngữ nghĩa và định dạng cơ bản.
3. Hỗ trợ cả tiếng Việt và tiếng Anh.
4. Không thêm hoặc bớt thông tin."""

    user_prompt = f"Hãy sửa lỗi chính tả và ngữ pháp cho văn bản sau:\n\n{text}\n\nChỉ trả về văn bản đã sửa:"

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            headers = {"Content-Type": "application/json"}
            json_data = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            if LLM_PROVIDER == "openai":
                headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
                url = "https://api.openai.com/v1/chat/completions"
            else:
                headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
                url = "https://api.groq.com/openai/v1/chat/completions"

            resp = await client.post(url, headers=headers, json=json_data)
            if resp.status_code == 200:
                data = resp.json()
                corrected = data["choices"][0]["message"]["content"].strip()
                result.update({
                    "corrected_text": corrected,
                    "corrected": True,
                    "provider": LLM_PROVIDER
                })
            else:
                print(f"[LLM] Error: {resp.status_code} {resp.text}")

    except Exception as e:
        print(f"[LLM] Correction error: {str(e)}")

    return result


# ==============================
# 🧠 OCR API
# ==============================
@app.get("/ocr/health", tags=["OCR"])
async def ocr_health():
    return {
        "status": "ok",
        "device": "GPU" if USE_GPU else "CPU",
        "llm_enabled": LLM_ENABLED,
        "llm_provider": LLM_PROVIDER if LLM_ENABLED else None,
        "llm_model": LLM_MODEL if LLM_ENABLED else None
    }


@app.post("/ocr/recognize", tags=["OCR"])
async def ocr_image(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        result = ocr.ocr(filepath, cls=True)
        lines = []
        for page in result:
            for line in page:
                text, conf = line[1]
                lines.append(text)
        extracted_text = "\n".join(lines).strip()

        correction_result = await correct_text_with_llm(extracted_text)

        output_file = os.path.join(OUTPUT_DIR, f"{timestamp}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        corrected_output_file = None
        if correction_result["corrected"]:
            corrected_output_file = os.path.join(OUTPUT_DIR, f"{timestamp}_corrected.txt")
            with open(corrected_output_file, "w", encoding="utf-8") as f:
                f.write(correction_result["corrected_text"])
            asyncio.create_task(delete_file_after(corrected_output_file))

        asyncio.create_task(delete_file_after(filepath))
        asyncio.create_task(delete_file_after(output_file))

        return JSONResponse({
            "status": "success",
            "filename": filename,
            "text": extracted_text,
            "output_file": output_file,
            "llm_correction": {
                "enabled": LLM_ENABLED,
                "corrected": correction_result["corrected"],
                "corrected_text": correction_result["corrected_text"],
                "provider": correction_result["provider"],
                "output_file": corrected_output_file
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# 🏠 Giao diện HTML test
# ==========================
@app.get("/", response_class=HTMLResponse, tags=["Web"])
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h2>OCR API (PaddleOCR + LLM) đang chạy</h2>")
