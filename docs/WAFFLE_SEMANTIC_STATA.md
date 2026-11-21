# Waffle + semantic + StatA trên đủ DTD

**Cập nhật full EuroSAT:** xem [báo cáo kiểm tra EuroSAT](WAFFLE_SEMANTIC_STATA_EUROSAT_REVIEW.md). Waffle–semantic đạt accuracy 64,42%, còn thêm StatA giảm xuống 61,73%; kết quả không ủng hộ dùng nhánh có StatA làm mặc định chung. Nhận định DTD phía dưới chỉ áp dụng cho DTD.

Đã có kết quả full DTD: xem [báo cáo kiểm tra](WAFFLE_SEMANTIC_STATA_REVIEW.md). Waffle–semantic–StatA đạt accuracy 71,10%, macro-F1 69,95%, tốt nhất về hai metric này trong các phương pháp đã thử trên DTD; chưa xác nhận trên dataset khác. Các lệnh dưới đây được giữ để tái hiện khi cần.

## Bước tiếp theo: EuroSAT đủ 27.000 ảnh

Job `waffle_semantic_stata_eurosat` dùng **toàn bộ 27.000 ảnh RGB, 10 lớp**, theo giao thức EuroSAT của Waffle upstream; không gọi đây là một test split chính thức riêng. Giữ nguyên semantic weight 0,5, 7 prompt seed × 3 grouping seed, solver, batch, backbone và chín đối chứng như DTD. Chỉ đổi nguồn descriptor sang EuroSAT và số ảnh; không tìm tham số theo EuroSAT. Cấu hình riêng `configs/waffle_semantic_stata_eurosat.yaml`, timeout GPU 14.400 giây (giới hạn 4 giờ, không phải thời gian chạy đã đo).

Smoke trước, thành công rồi mới chạy full:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_semantic_stata_eurosat --max-samples 5 --use-kaggle true --wait

venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_semantic_stata_eurosat --use-kaggle true --wait
```

Thêm `--prepare-only` để kiểm tra offline. Không upload ảnh mới: job gắn dataset benchmark đã có. Kernel `nguyenhung3105/vlm-final-eurosat-waffle-semantic-stata-full`, smoke thêm `-smoke`. Output `results/final/kaggle/waffle_semantic_stata_eurosat_<id>/run/`. Gửi `summary.tsv` và `comparisons.tsv` để kiểm tra.

Full EuroSAT vẫn có 71 lượt dự đoán/42 đối chiếu và chỉ mã hóa ảnh một lần. Các công thức/cách đọc bên dưới áp dụng tương tự với descriptor EuroSAT. EuroSAT từng có pilot 256 ảnh phục vụ phát triển; lượt này kiểm tra khả năng chuyển cấu hình từ DTD sang EuroSAT, không phải xác nhận độc lập trên dữ liệu chưa từng xem. Không thay hoặc chạy lại job DTD, không chạy test Fakeddit.

Job mới `waffle_semantic_stata_dtd` chạy đủ **1.880 ảnh DTD test partition 1, 47 lớp**, cùng MetaCLIP2 ViT-H/14. Bao gồm cả Waffle–StatA và các nhánh semantic trong một lượt, dùng chung ảnh, features và nhóm StatA. Job/outputs cũ giữ riêng. Người dùng tự gửi Kaggle; không cần upload lại dataset ảnh.

## Lệnh chạy

Smoke trước:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_semantic_stata_dtd --max-samples 5 --use-kaggle true --wait
```

Smoke thành công thì chạy full:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job waffle_semantic_stata_dtd --use-kaggle true --wait
```

Thêm `--prepare-only` để kiểm tra offline. Kernel riêng `nguyenhung3105/vlm-final-dtd-waffle-semantic-stata-full`, smoke thêm `-smoke`. Mỗi notebook chạy đúng mode được chọn, một lần nạp model và một lần mã hóa ảnh. Timeout 7.200 giây là giới hạn cấu hình, chưa phải thời gian đo. Dùng `--resume-stage` được in khi gửi nếu cần quay lại theo dõi job cũ.

## Semantic và công thức

Semantic ở đây là **mô tả đặc trưng lớp**, không phải cảm xúc hoặc text bài đăng. Dùng descriptor DTD từ repo Waffle đã pin commit qua bộ sinh prompt DCLIP tác giả, giống nguồn của lượt DCLIP–StatA trước. Không sinh mô tả mới từ nhãn test hoặc phân tích lỗi. JSON descriptor được sắp theo thứ tự lớp trước khi sinh prompt.

Gọi `W_s = unit(mean(unit(Waffle prompt embeddings)))` với prompt seed s; `S = unit(mean(unit(semantic prompt embeddings)))` cố định. Prototype kết hợp:

`B_s = unit(0,5 × W_s + 0,5 × S)`.

**Waffle–semantic** dùng argmax cosine với B. **Waffle–semantic–StatA** đưa B vào StatA làm prior/khởi tạo rồi argmax posterior. Không lấy trung bình posterior hoặc dự đoán phân cấp. Trọng số semantic 0,5 được cố định trước lần chạy, không grid search. Cấu hình ở `configs/waffle_semantic_stata.yaml`.

## Các đối chứng trong cùng job

| Method trong artifact | Số lượt | Ý nghĩa |
|---|---:|---|
| `clip` | 1 | Mốc ảnh |
| `clip_stata` | 3 | Thích nghi bằng CLIP prototype |
| `waffle` | 7 | Waffle mean prototype gốc |
| `waffle_unit` | 7 | Waffle prototype unit |
| `waffle_stata` | 21 | Unit Waffle prototype → StatA |
| `semantic_unit` | 1 | Unit semantic prototype (tương ứng đối chứng DCLIP unit) |
| `semantic_stata` | 3 | Semantic prototype → StatA |
| `waffle_semantic` | 7 | Prototype B, không StatA |
| `waffle_semantic_stata` | 21 | Prototype B → StatA |

Tổng 71 lượt, không phải 71 lần mã hóa ảnh. Bảy prompt seed 0–6, ba grouping seed 0/1/2; StatA dùng cùng các nhóm tối đa 75 ảnh không nhãn giữa mọi nhánh. Solver giữ tham số hiện có. Nhãn chỉ được lấy để đánh giá sau khi tạo đủ posterior.

## Kết quả cần gửi

Output `results/final/kaggle/waffle_semantic_stata_dtd_<id>/run/`. Gửi `summary.tsv` và `comparisons.tsv`.

- `summary.tsv`: chín phương pháp, trung bình và sample std ddof=1; phương pháp kết hợp có 21 tổ hợp seed, không coi là khoảng tin cậy hay các cohort độc lập.
- `metrics_by_seed.tsv`, `per_class.json`, `predictions.tsv`: chi tiết theo seed/lớp/mẫu.
- `comparisons.tsv`: 42 dòng, phân biệt bằng cột `method`. 21 dòng `waffle_stata` so Waffle/Waffle unit/CLIP–StatA; 21 dòng `waffle_semantic_stata` so Waffle–semantic cùng prompt seed, Waffle–StatA cùng prompt/grouping seed và semantic–StatA cùng grouping seed. Các cột không áp dụng cho loại dòng được để trống.
- `replay_scores.npz`: features, semantic/Waffle/blended prototypes, posterior, group ID và ID/nhãn. Đọc bằng `allow_pickle=False` để kiểm tra offline không cần model.
- `frozen_plan.json`, `cohort.json`, `run.json`, `completed.json`: prompt semantic nguyên văn, công thức, seed, cohort, source/code/specification hashes và artifact checksums.

Để chứng minh ích lợi kết hợp, cần so Waffle–semantic–StatA với **cả Waffle–StatA và semantic–StatA**, đồng thời so Waffle–semantic để tách hiệu ứng StatA. Không chỉ so với CLIP. Đây là split công khai đã phục vụ phát triển; kết quả chưa chứng minh tổng quát hóa sang dataset khác hoặc xác nhận độc lập.
