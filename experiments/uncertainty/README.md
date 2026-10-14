# Thử cờ `needs_review` từ kết quả đã lưu

`analyze_review.py` đọc `raw_predictions.tsv` bên trong `run_artifacts.zip` của một lượt single-label đã tải về. Nó không nạp model, không gọi Kaggle và không sửa nhãn dự đoán. Quy tắc thử nghiệm `branch_disagreement_v1` đánh dấu `needs_review=true` khi nhánh ảnh và nhánh text chọn khác nhãn; nếu một nhánh thiếu điểm, mẫu cũng được đánh dấu. Quy tắc **không dùng nhãn thật**.

```bash
python -B experiments/uncertainty/analyze_review.py \
  --run-dir results/experiments/adapted_crisismmd_info/03f1882357bb
```

Kết quả nằm tại `results/analysis/uncertainty/<run_id>_branch_disagreement_v1/`:

- `review_flags.tsv`: ID, nhãn gốc, dự đoán gốc, hai dự đoán nhánh, margin và cờ. `y_true`/`correct` chỉ dùng để phân tích hậu nghiệm.
- `summary.json`: độ phủ tự động, số lỗi và tỷ lệ lỗi của hai nhóm, tỷ lệ lỗi được cờ bắt lại.
- `by_class.tsv`: số mẫu và lỗi theo nhãn thật và nhóm.
- `analysis_manifest.json`: ID lượt nguồn, mã băm code/data, tập mẫu và phiên bản quy tắc.

Đây là phân tích thăm dò trên tập đã dùng để chọn MetaCLIP2 + DCLIP. Không diễn giải tỷ lệ lỗi của nhóm tự động thành bảo đảm trên dữ liệu mới. Trước khi bật cờ cho pipeline chính, giữ quy tắc cố định và đánh giá trên tập độc lập; nếu thêm ngưỡng margin, chọn ngưỡng chỉ trên tập hiệu chỉnh.

## Thử model phụ cho mẫu được gắn cờ

`compare_fallback.py` ghép từng mẫu theo ID, nhãn thật, caption, tên ảnh và thứ tự. Nó lấy dự đoán từ model phụ **chỉ khi** `needs_review=true`, rồi đếm số mẫu được sửa đúng (`n_fixed`) và số mẫu bị làm sai (`n_broken`). Script chỉ đọc kết quả đã lưu, không tải model hay gọi Kaggle.

```bash
python3 -B experiments/uncertainty/compare_fallback.py \
  --primary-dir results/experiments/adapted_crisismmd_info/2ff98e4f0871 \
  --flags-dir results/analysis/uncertainty/2ff98e4f0871_branch_disagreement_v1 \
  --candidate-dir results/runs/crisismmd_info/e595fbb7f79a \
  --candidate-model dfn5b_vith14
```

Đổi `--candidate-model` thành `siglip2_so400m` để thử model còn lại trên cùng 1.000 mẫu. Kết quả ghi dưới `results/analysis/fallback/`: `summary.json` gồm Accuracy/Macro-F1 trước và sau chuyển, cùng số sửa đúng/làm sai; `switches.tsv` liệt kê các mẫu đã đổi dự đoán. Các model phụ ở lượt chạy này dùng prompt gốc, nên phép thử đánh giá **cả model và cách đặt prompt**. Đây là so sánh hậu nghiệm trên tập hiện có, chưa phải kiểm chứng trên test độc lập.

## Thử tỷ trọng ảnh/text trên MetaCLIP2 + DCLIP

```bash
venv/bin/python -B experiments/uncertainty/sweep_fusion_weight.py \
  --run-dir results/experiments/adapted_crisismmd_info/03f1882357bb
```

Script kiểm tra rằng điểm hai nhánh tái tạo đúng dự đoán gốc, rồi thử `image_weight` từ `0` đến `1` theo bước `0.05`. Năm fold được ghép theo ID hoặc caption trùng nhau; mỗi fold chọn tỷ trọng bằng Macro-F1 trên bốn fold còn lại rồi chỉ đánh giá trên fold giữ lại. `full_cohort_sweep.tsv` là bảng **thăm dò**, không dùng để chọn tỷ trọng. `folds.tsv` và `summary.json` ghi lựa chọn và kết quả cross-fit. Không chạy inference, không đổi `experiment_config.yaml`.

Lượt 3.000 mẫu chọn `0.55` ở cả năm fold: Accuracy `0.8657 → 0.8703`, Macro-F1 `0.8645 → 0.8691`, sửa đúng `47` và làm sai `33` mẫu. Chênh lệch nhỏ (McNemar exact hai phía thăm dò `p≈0.146`). Cả 80 mẫu đổi nhãn đều có nhánh ảnh/text bất đồng và margin gốc `≤0.10`. Dataset này đã tham gia chọn model/prompt trước đó và manifest chưa xác minh split, nên đây **không phải** bằng chứng cải thiện trên test độc lập. Cần khóa `0.55` rồi chạy trên tập mới trước khi thay cấu hình chính.
