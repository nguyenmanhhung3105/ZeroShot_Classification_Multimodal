# Few-shot tầng 1 Fakeddit trên min5

Đã có run full: xem [báo cáo kiểm tra min5](FAKEDDIT_FEWSHOT_TASK1_MIN5_REVIEW.md). Ảnh–tiêu đề đạt accuracy 81,22%, macro-F1 80,16%, tốt hơn hai nhánh riêng ở cả năm seed về accuracy. Đây là baseline few-shot đã đánh giá trên dev; chưa mở test hoặc triển khai tầng hai. Các lệnh dưới đây được giữ để tái hiện khi cần.

Đã có [phân tích lỗi và đối chứng chọn C bằng train-CV](FAKEDDIT_FEWSHOT_TASK1_ERROR_ANALYSIS.md): CV chưa giúp, giữ baseline C=1. Không thay cấu hình/run gốc hoặc chạy thêm GPU trong bước phân tích offline này.

Job `fakeddit_fewshot_task1_min5` chỉ học bộ phân loại **Real/Fake**, chưa triển khai tầng hai. Ba nhánh: ảnh, tiêu đề, nối ảnh–tiêu đề. MetaCLIP2 ViT-H/14 đóng băng; Logistic Regression giữ đúng lượt few-shot cũ: C=1, class_weight=balanced, solver=lbfgs, max_iter=2000, tol=1e-6. Đây là baseline few-shot có dùng nhãn train, không phải zero-shot hoặc phương pháp mới được đề xuất.

## Dữ liệu và support

- Nguồn min5 đã chuẩn bị: 3.000 mẫu train, 3.000 mẫu dev; mỗi manifest được khóa SHA256 trong `configs/fakeddit_fewshot_task1_min5.yaml`. Đưa nhầm cohort v2 cũ hoặc test làm job dừng trước model.
- Lấy **64 mẫu mỗi lớp 6way = 384 mẫu/seed**, seed 0/1/2/3/4, giống thứ tự RNG của implementation cũ. Khi chuyển sang binary, mỗi support có **64 Real + 320 Fake**. Không gọi đây là 64-shot mỗi lớp binary. Năm support có thể chồng lấp; std giữa seed không phải khoảng tin cậy hoặc năm dataset độc lập.
- Mỗi seed, cả ba modality dùng đúng cùng ID train. Chỉ lấy các support từ train; dev dùng đánh giá, không fit/tuning ngưỡng. Không gắn hoặc mở test.
- Chặn trùng ID, hash ảnh và hash tiêu đề giữa toàn bộ train/dev trước khi nạp model. Bộ đọc vẫn kiểm schema/cohort và ảnh của train đầy đủ; encoder chỉ xử lý hợp các support ID đã chọn, không encode toàn bộ train không cần thiết.
- Smoke giữ **đủ 384 support/seed và năm seed**, nhưng chỉ đánh giá năm mẫu dev. Vì vậy smoke vẫn cần mã hóa nhiều ảnh train; không phải tổng cộng chỉ năm ảnh và không dùng metric smoke để báo cáo.

Binary mapping giữ đúng gate few-shot cũ: **0=Real, 1=Fake**, tính bằng `(6way != 0)`. Mapping này ngược mã `2_way_label` gốc của Fakeddit; các artifact ghi rõ tên lớp, xác suất và mapping để tránh đọc nhầm.

## Upload train min5, nếu chưa có

Dev min5 đã được dùng ở job trước. Job này cần thêm dataset **train min5** riêng. Metadata đã chuẩn bị tại `data/processed/fakeddit_splits_min5_v1/train/dataset-metadata.json`; không thay nội dung mẫu/manifest và không upload test.

```sh
venv/bin/kaggle datasets create \
  -p data/processed/fakeddit_splits_min5_v1/train -r zip

venv/bin/kaggle datasets status nguyenhung3105/vlm-fakeddit-train-min5-v1
```

Đợi dataset ready trước khi gửi GPU. Nếu train min5 đã có và ready thì bỏ bước create, không gửi lại hoặc ghi đè dataset.

Nếu `datasets status` trả 403 sau upload, không kết luận dataset bị mất hoặc tạo lại ngay. Đã quan sát trường hợp train min5: status bị từ chối nhưng `datasets files` đọc được TSV và ảnh đã giải nén. Launcher few-shot hiện kiểm tra thay thế bằng file API khi status 403: đọc đủ các trang, yêu cầu manifest.json, fakeddit_sample.tsv và đúng 3.000 ảnh không rỗng. Chỉ sau đó mới cho gửi job; run tiếp tục kiểm hash manifest/các mẫu trước model. Nếu file API cũng 401/403/404 hoặc dữ liệu thiếu, launcher dừng. Kiểm tra đọc thủ công:

```sh
venv/bin/kaggle datasets files nguyenhung3105/vlm-fakeddit-train-min5-v1 \
  --page-size 5 --format json
```

Lệnh thủ công chỉ xem tối đa năm file để chẩn đoán quyền truy cập, không thay thế kiểm tra đủ dataset trong launcher. Không cần upload lại chỉ vì status 403.

## Chạy job

Smoke:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_fewshot_min5.yaml \
  --job fakeddit_fewshot_task1_min5 --max-samples 5 --use-kaggle true --wait
```

Full dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_fewshot_min5.yaml \
  --job fakeddit_fewshot_task1_min5 --use-kaggle true --wait
```

Thêm `--prepare-only` chỉ chuẩn bị offline, không đọc dataset/nạp model/gọi Kaggle. Không gửi GPU tự động trong quá trình triển khai. Kernel riêng `nguyenhung3105/vlm-min5-fakeddit-fewshot-task1-min5`, smoke thêm `-smoke`. Notebook dùng đúng project_min5 và ghi đè riêng hai slug train/dev bằng spec few-shot đã cố định; gắn đúng hai dataset min5, không dùng train v2 cũ. Launcher kiểm cả hai dataset trước upload code/push. Mỗi notebook nạp model một lần, encode train support và dev một lượt chung cho ba nhánh, fit 15 classifier (ba modality × năm seed), không chạy full ngầm sau smoke.

Timeout GPU 7.200 giây là giới hạn cấu hình, chưa phải thời gian chạy đã đo; poll 120 giây. Ctrl+C chỉ dừng local chờ, không hủy GPU. Dùng `--resume-stage` launcher in ra để thu đúng version đã gửi, không gửi trùng.

## Feature, classifier và artifact

Ảnh và tiêu đề được encode thành vector unit. Nhánh ảnh–tiêu đề dùng `concat(image,title)/sqrt(2)`, giống implementation cũ, không tune trọng số. Title dùng tokenizer/context hiện có của MetaCLIP2 (có thể bị cắt theo context). Không Waffle, StatA, CALIP, không fine-tune encoder. Classifier dùng quyết định mặc định từ posterior, không chọn ngưỡng theo dev. Nếu LBFGS chưa hội tụ, run dừng và không ghi completed.

Output local: `results/final/kaggle/fakeddit_fewshot_task1_min5_<id>/run/`.

- `summary.tsv`: accuracy, macro-F1, balanced accuracy trung bình/sample std qua năm seed cho ba modality. Không xuất top-5 trên binary.
- `class_summary.tsv`: recall Real/Fake và số Fake→Real, Real→Fake, mean/std.
- `comparisons.tsv`: metrics/recall/lỗi từng seed, số sửa đúng/làm sai so nhánh ảnh cùng seed.
- `predictions.tsv`, `per_class.json`: dự đoán/xác suất và metrics từng lớp theo method/seed.
- `support.json`: đúng ID/nhãn 6way support mỗi seed, dùng chung ba modality.
- `features.npz`: feature unit train/dev riêng ảnh/title, nhãn 6way train, ID và support index trong cache. Nhãn dev ở `replay_scores.npz` (cùng thứ tự dev_ids), không đưa vào hàm fit/predict.
- `linear_models.npz`: coefficient và intercept từng classifier, không pickle. Có thể tính lại posterior từ feature đã lưu để phân tích offline mà không nạp MetaCLIP2.
- `replay_scores.npz`: ID, binary labels/tên lớp, 15 posterior; đọc mọi NPZ bằng `allow_pickle=False`.
- `cohort.json`, `train_cohort.json`, `frozen_plan.json`, `run.json`, `completed.json`: ID/hash/cohort, classifier, feature formula, mapping nhãn, hash source/specification và checksum tất cả artifact. Không đủ điều kiện tạo lock bằng workflow Fakeddit zero-shot cũ.

Tập trung so ảnh–tiêu đề với từng nhánh riêng trên **cùng min5 cohort/support**. Nếu text không giúp thì phân tích biểu diễn/lỗi trước khi thêm thành phần. Kết quả cũ 78,05% binary few-shot thuộc cohort v2 cũ nên không lấy chênh lệch với min5 làm cải tiến thuật toán. Giữ dev cho phát triển; chỉ khóa và đánh giá test bằng protocol few-shot phù hợp sau khi chọn xong cấu hình.
