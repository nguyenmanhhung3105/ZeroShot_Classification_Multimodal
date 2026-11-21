# Pilot WaffleCLIP → StatA một tầng

Đã có kết quả hai pilot: xem [báo cáo kiểm tra](WAFFLE_STATA_REVIEW.md). Waffle–StatA có tín hiệu tốt trên DTD nhưng giảm trên EuroSAT; chưa chọn làm phương pháp chung.

## Lượt tiếp theo: đủ 1.880 ảnh DTD

Job mới `waffle_stata_dtd_full` dùng cấu hình riêng `configs/waffle_stata_full.yaml`, giữ nguyên pilot cũ. Lượt full dùng cùng split/cohort chính thức với DCLIP–StatA full: 1.880 ảnh, 47 lớp. Vẫn có CLIP, CLIP–StatA, Waffle, Waffle unit và Waffle–StatA; 7 seed prompt × 3 seed grouping cho phương pháp kết hợp. Không thêm semantic trong lượt này. Đây vẫn là split đã xem trong phát triển, không phải xác nhận độc lập.

Smoke trước rồi chạy full:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_stata_dtd_full --max-samples 5 --use-kaggle true --wait

venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_stata_dtd_full --use-kaggle true --wait
```

Kernel riêng `nguyenhung3105/vlm-final-dtd-waffle-stata-full`, smoke thêm hậu tố `-smoke`. Output local `results/final/kaggle/waffle_stata_dtd_full_<id>/run/`; gửi `summary.tsv` và `comparisons.tsv`. Mỗi notebook chỉ chạy mode đã chọn; không tự chuyển từ smoke sang full. Không dùng job pilot `waffle_stata_dtd` để chạy full bằng cách đổi số mẫu.

Hai job `waffle_stata_dtd` và `waffle_stata_eurosat` dùng MetaCLIP2 ViT-H/14, mỗi dataset 256 ảnh chọn ngẫu nhiên không theo nhãn, selection seed 0, đủ các lớp ứng viên. Dùng lại dataset Kaggle benchmark hiện có; không cần upload dữ liệu ảnh mới. Đây là pilot phát triển trên tập đánh giá công khai (subset đã phục vụ nghiên cứu), không phải xác nhận độc lập hoặc kết quả full split.

## Lệnh chạy

Kiểm tra luồng DTD 5 ảnh trước; smoke vẫn chạy đủ mọi phương pháp/seed:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_stata_dtd --max-samples 5 --use-kaggle true --wait
```

Sau smoke thành công, chạy tuần tự hai pilot:

```sh
for job in waffle_stata_dtd waffle_stata_eurosat; do
  venv/bin/python -B scripts/launch_runs.py \
    --launcher-config configs/kaggle_launcher.yaml \
    --job "$job" --use-kaggle true --wait || break
done
```

Thêm `--prepare-only` để kiểm tra offline, không nạp model/đọc dataset/gọi Kaggle. Mỗi notebook chạy đúng một lượt: smoke 5 hoặc pilot 256, không tự chạy thêm full split. GPU timeout 7.200 giây là giới hạn, chưa phải thời gian thực tế đo được. Model và ảnh chỉ mã hóa một lần mỗi run, các nhánh dùng chung features. Kernel riêng có hậu tố `-<dataset>-waffle-stata-pilot`; smoke thêm `-smoke`. Mất kết nối sau push thì dùng `--resume-stage` được launcher in ra, không gửi trùng.

## Phương pháp và đối chứng

Waffle dùng bộ sinh prompt tác giả đã pin commit, 15 cặp, prompt seed 0–6. CLIP dùng bộ prompt photo từ cùng API. StatA dùng các tham số solver hiện có: prior scale 100, 10 vòng, 5 vòng trong, 3 láng giềng, nhóm tối đa 75 ảnh không nhãn; chia nhóm seed 0/1/2. Với cùng grouping seed, mọi phương pháp dùng đúng cùng nhóm và thứ tự ảnh; nhóm tối đa 3 ảnh giữ prior.

| Phương pháp | Xử lý | Số lượt/seed |
|---|---|---:|
| CLIP | Cosine với mean prompt prototype | 1 |
| CLIP–StatA | StatA dùng unit CLIP prototype | 3 grouping seed |
| Waffle | Cosine với mean Waffle prototype, giữ aggregation hiện có | 7 prompt seed |
| Waffle unit | Chuẩn hóa lại mean Waffle prototype trước cosine | 7 prompt seed |
| Waffle–StatA | StatA dùng unit Waffle prototype làm prior/khởi tạo | 7 × 3 = 21 tổ hợp |

Waffle–StatA không phải trung bình posterior của Waffle và StatA. Mỗi nhánh ra một argmax trên toàn bộ lớp, không có phân cấp. Nhãn chỉ dùng đánh giá sau khi tạo đủ posterior. Đối chứng Waffle unit cần thiết vì StatA yêu cầu prototype unit; không quy toàn bộ cải thiện so Waffle gốc cho riêng StatA.

`summary.tsv` báo trung bình và sample std ddof=1; riêng Waffle–StatA là biến thiên trên 21 tổ hợp prompt/grouping seed, không phải 21 dataset độc lập hay khoảng tin cậy. Không chọn seed tốt nhất. Cấu hình khóa ở `configs/waffle_stata.yaml`.

## Kết quả cần gửi

Output local: `results/final/kaggle/waffle_stata_<dataset>_<id>/run/`.

- `summary.tsv`: accuracy, macro-F1, balanced accuracy, top-5 của năm phương pháp.
- `metrics_by_seed.tsv`: 39 lượt, seed `pN_gM` ghi rõ hai nguồn ngẫu nhiên.
- `comparisons.tsv`: 21 đối chiếu Waffle–StatA với Waffle/Waffle unit cùng prompt seed và CLIP–StatA cùng grouping seed; delta accuracy và số sửa sai/làm sai.
- `replay_scores.npz`: feature, prototype, posterior, nhóm ảnh và ID/nhãn; kiểm tra offline bằng `allow_pickle=False`.
- `cohort.json`, `frozen_plan.json`, `run.json`, `completed.json`: nguồn, cấu hình, prompt và checksum artifact.

Ưu tiên xem Waffle–StatA có vượt **cả Waffle unit và CLIP–StatA** trên hai dataset, cùng độ ổn định theo seed. Chỉ hơn CLIP chưa đủ chứng minh hai thành phần bổ trợ nhau. Giữ nguyên kết quả giảm nếu có; không chỉnh tham số theo từng dataset sau khi xem điểm rồi gọi đó là xác nhận độc lập.
