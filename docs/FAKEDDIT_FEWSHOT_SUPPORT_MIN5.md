# Đối chứng phân bổ support cho tầng 1 min5

Đã có run full: xem [báo cáo kiểm tra](FAKEDDIT_FEWSHOT_SUPPORT_MIN5_REVIEW.md). Nhánh ảnh–tiêu đề 192/192 đạt accuracy 82,41%, macro-F1 82,04%, nhưng recall Fake giảm từ 86,91% xuống 80,53%. Giữ cả hai policy để xem trade-off, chưa thay baseline tự động.

Job `fakeddit_fewshot_support_min5` so sánh hai cách chia cùng ngân sách 384 mẫu có nhãn/seed trên toàn bộ pool 3.000 train min5. Đánh giá cùng 3.000 dev min5, cùng MetaCLIP2 đóng băng và ba modality ảnh/tiêu đề/ảnh–tiêu đề. Không dùng pool cache cũ làm pool lấy mẫu; không mở test hoặc chạy tầng hai.

| Policy | Real | Fake | Tổng mỗi seed |
|---|---:|---:|---:|
| `legacy_6way` | 64 | 320 (64 mỗi loại Fake) | 384 |
| `balanced_binary` | 192 | 192 (38–39 mỗi loại Fake) | 384 |

Giữ 5 seed 0–4, C=1, class_weight=balanced, lbfgs/max_iter=2000/tol=1e-6 và quyết định mặc định, không CV hoặc chọn ngưỡng. Nối feature ảnh–tiêu đề vẫn chia sqrt(2). Mỗi policy/seed, ba modality dùng cùng support. Fit tổng cộng 30 linear classifier, không fine-tune encoder.

## Sampler và cách đọc đối chứng

Fake quota 38 mỗi loại và hai suất dư xoay theo seed: [39,39,38,38,38], [38,39,39,38,38], …, [39,38,38,38,39]. Không lấy quota từ kết quả dev.

Dùng cùng permutation train trong từng lớp và cùng thứ tự RNG như sampler baseline. Do đó, mỗi seed hai policy chung đúng 256 ID: 64 Real và 192 Fake. Nhánh mới thay 128 Fake bằng 128 Real, giữ nguyên tổng ngân sách. Trên cohort hiện tại, hợp mọi support có 1.591 ID, thêm 419 Real ngoài cache baseline cũ. Encoder chỉ xử lý hợp này và dev một lượt chung cho tất cả nhánh.

Lưu ý **class_weight=balanced đã cân bằng tổng trọng số loss của hai lớp ở cả hai policy**. Việc đổi support tác động đồng thời đến độ đa dạng Real, độ phủ Fake và trọng số từng mẫu (legacy khoảng 3/0,6; balanced 1/1). Thí nghiệm này đo hiệu ứng phân bổ ngân sách, không chứng minh riêng một nguyên nhân. Đây là hai protocol support khác nhau cùng tổng budget; không gọi cả hai là 64-shot mỗi lớp binary.

## Lệnh chạy

Dùng lại hai dataset train/dev min5 đã upload, không cần upload ảnh mới. Launcher giữ cơ chế kiểm tra file API nếu dataset status trả 403, hash manifest vẫn được kiểm trước model.

Smoke trước (vẫn đủ hai policy/384 support/5 seed, chỉ giảm dev xuống 5 mẫu):

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_fewshot_min5.yaml \
  --job fakeddit_fewshot_support_min5 --max-samples 5 --use-kaggle true --wait
```

Sau smoke thành công, chạy full dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_fewshot_min5.yaml \
  --job fakeddit_fewshot_support_min5 --use-kaggle true --wait
```

Thêm `--prepare-only` để chỉ chuẩn bị offline. Kernel riêng `nguyenhung3105/vlm-min5-fakeddit-fewshot-support-min5`; smoke thêm `-smoke`. Timeout tối đa 7.200 giây, chưa phải thời gian đo. Poll 120 giây; có thể Ctrl+C phần chờ local và dùng `--resume-stage` launcher in ra để quay lại, không gửi trùng. Notebook chỉ chạy mode đã chọn, không tự chuyển smoke sang full. Job baseline cũ và mọi kết quả trước giữ nguyên.

## Artifact

Output `results/final/kaggle/fakeddit_fewshot_support_min5_<id>/run/`:

- `summary.tsv`: 6 hàng (2 policy × 3 modality), mỗi hàng mean/sample std qua 5 seed. Tên method có prefix policy, ví dụ `balanced_binary_image_title`.
- `class_summary.tsv`: recall Real/Fake và số Fake→Real/Real→Fake của từng policy/modality.
- **`support_comparisons.tsv`**: 15 so sánh balanced − legacy ghép cặp **cùng modality và seed**; delta accuracy/macro-F1/balanced accuracy/recall, sửa sai/làm sai. Đây là bảng đối chiếu chính.
- `comparisons.tsv`: metrics từng method/seed, đối chiếu với `legacy_6way_image`; không nhầm với so sánh policy cùng modality.
- `support.json`: 10 support set, mỗi set có policy/seed/ID/nhãn/quota 6way. `features.npz` chứa index `support_<policy>_s<seed>` tương ứng với train_ids của hợp mới.
- `replay_scores.npz` và `linear_models.npz`: 30 posterior và hệ số head để kiểm tra offline, không pickle.
- `cohort.json`, `train_cohort.json`, `frozen_plan.json`, `run.json`, `completed.json`: cohort, source/specification hashes và checksum artifact. Nhãn binary giữ 0=Real, 1=Fake, ngược coding `2_way_label` gốc; không thay mapping.

Gửi `summary.tsv`, `class_summary.tsv` và `support_comparisons.tsv` khi hoàn tất. Xem balanced có phục hồi recall Real mà giữ recall Fake và macro-F1 hay không. Không chọn winner chỉ bằng accuracy. Kiểm baseline legacy có tái hiện gần 81,22% trước khi quy chênh lệch cho policy; không so một seed mới tốt nhất với mean baseline cũ.

Đây là thí nghiệm phát triển được đề xuất sau khi xem lỗi dev; năm support có thể chồng lấp và std không phải khoảng tin cậy. Nếu kết quả âm, giữ nguyên baseline C=1/legacy; không tự thay runner chính hoặc mở test.

Đã có [script phân tích ngưỡng offline](FAKEDDIT_SUPPORT_THRESHOLDS.md) để so hai policy tại cùng mục tiêu recall Fake: chia selection/audit từ dev, chọn ngưỡng chỉ trên selection và giữ nguyên trên audit. Không cần Kaggle hoặc GPU.
