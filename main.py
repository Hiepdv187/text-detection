import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import easyocr
import torch
from dotenv import load_dotenv
from datetime import datetime

# ==========================
# 🔧 Cấu hình thư mục
# ==========================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
STATIC_DIR = "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ==========================
# 🚀 Khởi tạo app và OCR
# ==========================
tags_metadata = [
    {"name": "OCR", "description": "Các API nhận diện văn bản (upload ảnh, health check)."},
    {"name": "Web", "description": "Trang web giao diện test (HTML)."},
]

app = FastAPI(
    title="OCR Text Recognition API",
    description="Nhận diện văn bản từ ảnh (Python + EasyOCR, có sắp dòng thông minh)",
    version="1.1.0",
    openapi_tags=tags_metadata,
    docs_url="/ocr/docs",
    redoc_url="/ocr/redoc",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Nạp biến môi trường từ .env (nếu có)
load_dotenv()

# Quy tắc chọn GPU/CPU:
# - FORCE_CPU=true  -> luôn dùng CPU
# - FORCE_GPU=true  -> cố dùng GPU (nếu không có sẽ fallback CPU)
# - USE_GPU_AUTO=true (mặc định) -> tự động phát hiện
force_cpu = os.getenv("FORCE_CPU", "").lower() == "true"
force_gpu = os.getenv("FORCE_GPU", "").lower() == "true"
use_gpu_auto = os.getenv("USE_GPU_AUTO", "true").lower() == "true"

USE_GPU = False
reason = "auto"

if force_cpu:
    USE_GPU = False
    reason = "forced_cpu"
elif force_gpu:
    # Cố gắng dùng GPU, nếu không có sẽ fallback
    try:
        has_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        USE_GPU = bool(has_cuda or has_mps)
        reason = "forced_gpu_available" if USE_GPU else "forced_gpu_unavailable_fallback_cpu"
    except Exception:
        USE_GPU = False
        reason = "forced_gpu_error_fallback_cpu"
elif use_gpu_auto:
    # Tự động phát hiện
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

print(f"[EasyOCR] Device: {'GPU' if USE_GPU else 'CPU'} (mode={reason})")
reader = easyocr.Reader(['en', 'vi'], gpu=USE_GPU)

# ==========================
# 📷 API: Nhận ảnh, trả chữ (prefix: /ocr)
# ==========================
@app.get("/ocr/health", tags=["OCR"])
async def ocr_health():
    return {"status": "ok", "device": "GPU" if USE_GPU else "CPU"}


@app.post("/ocr/recognize", tags=["OCR"])
async def ocr_image(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Lưu ảnh tạm
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # OCR nhận diện
        results = reader.readtext(filepath)

        # ==========================
        # 🧠 Sắp xếp lại kết quả theo dòng
        # ==========================
        def sort_key(item):
            (bbox, text, conf) = item
            y_mean = sum([p[1] for p in bbox]) / 4
            x_min = min([p[0] for p in bbox])
            return (round(y_mean / 25), x_min)

        results_sorted = sorted(results, key=sort_key)

        extracted_text = ""
        last_y = None

        for (bbox, text, conf) in results_sorted:
            y_mean = sum([p[1] for p in bbox]) / 4
            if last_y is None or abs(y_mean - last_y) > 25:
                extracted_text += "\n"
            extracted_text += text + " "
            last_y = y_mean

        extracted_text = extracted_text.strip()

        # Ghi ra file text
        output_file = os.path.join(OUTPUT_DIR, f"{timestamp}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        return JSONResponse({
            "status": "success",
            "filename": filename,
            "text": extracted_text,
            "output_file": output_file
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# 🏠 Trang HTML test upload
# ==========================
@app.get("/", response_class=HTMLResponse, tags=["Web"])
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h2>OCR API đang chạy</h2>")
