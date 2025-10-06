OCR Text Recognition API (Docker)
=================================

Đây là API OCR (Nhận diện văn bản) dùng FastAPI + EasyOCR.
Tài liệu này hướng dẫn build và chạy bằng Docker với cổng 5678.

Lưu ý: Image Docker chỉ bao gồm API. Các file test/tài liệu được loại bỏ nhờ `.dockerignore`.

Yêu cầu
-------
- Cài Docker
- Internet để pull base image và cài package Python

Cấu hình môi trường (.env)
--------------------------
File `.env` ví dụ:

    PORT=5678
    HOST=0.0.0.0
    # Tự động phát hiện GPU (mặc định true). Có thể tắt bằng false.
    USE_GPU_AUTO=true
    # Ép CPU (ưu tiên tuyệt đối nếu đặt true)
    FORCE_CPU=false
    # Ép GPU (nếu không có GPU sẽ tự fallback về CPU)
    FORCE_GPU=false

Ghi chú:
- `main.py` sẽ đọc `.env` (nhờ `python-dotenv`) để quyết định dùng GPU/CPU.
- Nếu không truyền `.env` vào container, Dockerfile có mặc định `PORT=5678`, `HOST=0.0.0.0`.

Build image
-----------
Chạy trong thư mục gốc (nơi có `Dockerfile`):

    docker build -t ocr-api:latest .

Chạy container
--------------

- Chạy cơ bản (mặc định cổng 5678):

    docker run --rm -p 5678:5678 --name ocr-api ocr-api:latest

- Chạy kèm `.env` (khuyến nghị):

    docker run --rm --env-file .env -p 5678:5678 --name ocr-api ocr-api:latest

- Gắn thư mục để lưu dữ liệu (tùy chọn):

    docker run --rm -p 5678:5678 \
      -v %cd%/uploads:/app/uploads \
      -v %cd%/output:/app/output \
      --name ocr-api ocr-api:latest

PowerShell (nếu %cd% không hoạt động):

    docker run --rm -p 5678:5678 `
      -v ${PWD}/uploads:/app/uploads `
      -v ${PWD}/output:/app/output `
      --name ocr-api ocr-api:latest

Sử dụng API
-----------
- Health check (API):

    curl http://localhost:5678/ocr/health

- Gọi OCR (upload ảnh):

    curl -X POST "http://localhost:5678/ocr/recognize" -F "file=@duong_dan_anh.png"

- Swagger UI (nếu bật trong code FastAPI):

    http://localhost:5678/docs

Ghi chú thêm
------------
- Image có cài các thư viện hệ thống tối thiểu cho EasyOCR/OpenCV (`libgl1`, `libglib2.0-0`).
- Lần chạy đầu có thể chậm vì EasyOCR tải model.
- GPU trong Docker cần base image hỗ trợ CUDA + NVIDIA Container Toolkit. Image này mặc định chạy CPU an toàn. Nếu cần GPU, chúng ta có thể tạo biến thể CUDA.

Khắc phục sự cố
---------------
- Trùng cổng: đổi cổng host, ví dụ `-p 8080:5678`.
- Chậm: CPU-only; cân nhắc dùng máy mạnh hơn hoặc image có GPU.
- Upload lỗi: nhớ dùng `multipart/form-data` với field `file`.
