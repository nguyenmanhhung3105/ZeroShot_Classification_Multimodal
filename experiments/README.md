# Mã thí nghiệm

`ensemble/run_ensemble.py` là phép đối chứng tùy chọn trên dự đoán của nhiều model. Chạy qua `scripts/launch_runs.py` với `kind: standard` và `ensemble.enabled: true` trong `configs/launcher.yaml`.

Các runner nghiên cứu dùng chung code xử lý của dự án vẫn nằm trong `src/research/`: `run_adapted_methods.py`, `run_upstream.py` và `run_relation_ablation.py`. Launcher tạo cấu hình cho từng job và lưu đầu ra mới vào `results/experiments/<job>/<run_id>/`. Báo cáo cũ ở `docs/experiments/`; kết quả lịch sử được giữ nguyên.

`uncertainty/analyze_review.py` thử cờ `needs_review` từ log dự đoán đã tải về, không cần chạy model. Cách chạy và giới hạn của phép thử nằm trong [README uncertainty](uncertainty/README.md).
