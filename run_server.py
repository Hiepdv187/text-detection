import uvicorn

if __name__ == "__main__":
    # Chạy server FastAPI và phục vụ web UI tại '/'
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5678,
        reload=True
    )
