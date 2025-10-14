#!/usr/bin/env python3
"""
Script kiểm tra và xóa file cũ trong thư mục uploads và output
"""
import os
import time
import sys
import argparse

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
CLEANUP_AFTER_SECONDS = 300  # 5 phút

def cleanup_old_files():
    """Xóa tất cả file cũ trong uploads và output"""
    current_time = time.time()
    cleaned_count = 0

    # Kiểm tra thư mục uploads
    if os.path.exists(UPLOAD_DIR):
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
    if os.path.exists(OUTPUT_DIR):
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

def check_files():
    """Kiểm tra trạng thái các file"""
    current_time = time.time()
    files_info = {
        "uploads": [],
        "output": []
    }

    # Kiểm tra thư mục uploads
    if os.path.exists(UPLOAD_DIR):
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
    if os.path.exists(OUTPUT_DIR):
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

    print("=== TRẠNG THÁI CLEANUP ===")
    print(f"Thời gian cleanup: {CLEANUP_AFTER_SECONDS} giây ({CLEANUP_AFTER_SECONDS/60:.1f} phút)")
    print(f"Số file cũ cần xóa: {old_files_count}")
    print()

    if files_info["uploads"]:
        print("📁 UPLOADS:")
        for file_info in files_info["uploads"]:
            status = "🗑️ CŨ" if file_info["is_old"] else "✅ MỚI"
            age_min = file_info["age_seconds"] / 60
            print(f"  {file_info['filename']} - {age_min:.1f} phút - {file_info['size']} bytes - {status}")
        print()

    if files_info["output"]:
        print("📄 OUTPUT:")
        for file_info in files_info["output"]:
            status = "🗑️ CŨ" if file_info["is_old"] else "✅ MỚI"
            age_min = file_info["age_seconds"] / 60
            print(f"  {file_info['filename']} - {age_min:.1f} phút - {file_info['size']} bytes - {status}")
        print()

    return old_files_count

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra và xóa file cũ")
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không xóa")
    parser.add_argument("--cleanup", action="store_true", help="Xóa file cũ ngay lập tức")

    args = parser.parse_args()

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.check or not args.cleanup:
        old_count = check_files()

        if args.cleanup or old_count > 0:
            if args.cleanup:
                print("🧹 Đang xóa file cũ...")
                cleaned_count = cleanup_old_files()
                print(f"✅ Đã xóa {cleaned_count} file cũ")
            else:
                print(f"⚠️ Có {old_count} file cũ. Chạy với --cleanup để xóa.")

    elif args.cleanup:
        print("🧹 Đang xóa file cũ...")
        cleaned_count = cleanup_old_files()
        print(f"✅ Đã xóa {cleaned_count} file cũ")

if __name__ == "__main__":
    main()
