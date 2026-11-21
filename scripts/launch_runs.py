"""Giữ cách gọi launcher cũ; phần triển khai dùng source cuối."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project.kaggle import main

if __name__ == '__main__':
    main()
