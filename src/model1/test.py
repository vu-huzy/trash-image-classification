"""Entry point ngắn gọn để chạy toàn bộ Simple NN / Simple CNN Model 1.

Từ thư mục gốc của project:
    python src/model1/test.py

Smoke test nhanh (không dùng để báo cáo kết quả):
    python src/model1/test.py --epochs 1 --max-batches 1
"""

try:  # Chạy được cả ``python src/model1/test.py`` và ``python -m model1.test``.
    from .train import main
except ImportError:
    from train import main


if __name__ == "__main__":
    main()
