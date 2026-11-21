# Fakeddit dev 6way: kiểm chứng hệ thống hai tầng gốc

Job riêng, không sửa nhánh Fakeddit cũ, không mở test và không tạo test lock.
Dùng đúng 3.000 mẫu dev v2 đã có, đủ 6 lớp; smoke chọn 5 mẫu theo loader hiện có.
Backbone MetaCLIP2 ViT-H/14 worldwide, không fine-tune. Chỉ tải model một lần,
mã hóa ảnh/tiêu đề một lần; các nhánh dùng cùng feature, ID và nhãn.

## Thiết kế đối chứng

Không so image-only và multimodal rồi quy toàn bộ lợi ích cho chia tầng. Có hai
nhóm riêng:

1. **Tám đối chứng image-only:** CLIP, EM, StatA, Majority phẳng 6 lớp; và bốn
   phương pháp tương ứng dự đoán binary rồi chọn lớp con bằng CLIP. Dùng cùng
   prototype unit (binary semantic_v1, fine legacy), scale ảnh 100. EM/StatA chỉ
   nhận query không nhãn. Hai tầng thật=1 → True=0, fake=0 → một trong ID 1..5.
2. **Ba đối chứng ảnh–tiêu đề legacy:** `legacy_flat_mm`, `legacy_hierarchy_clip_mm`
   và `original_hierarchy_mm`. Cùng raw mean prototype fine và semantic text,
   fusion ảnh 0,75 + text 0,25, image scale 100 và text scale 25 như code gốc.
   Bản original gọi trực tiếp `methods.majority(features,binary_raw,seed)` để giữ
   EM prior raw mean, CLIP/StatA unit và binary gate image-only của nhánh cũ.

Không chuẩn hóa lại mean của bản original để làm đẹp so sánh. Bản unit chỉ dùng
trong nhóm kiểm soát riêng, tên artifact phản ánh khác biệt này. Không gọi bản
unit là tái lập nguyên điểm lịch sử. Bản legacy cần so với baseline MM legacy,
không kết luận lợi ích multimodal từ đối chiếu với các nhánh image-only.

CLIP tất định một lượt. EM, StatA, Majority và original ba grouping seed 0/1/2;
nhóm 75 query, giống giữa các nhánh. Majority đa lớp nếu ba phiếu khác nhau thì
giữ phiếu CLIP. Binary Majority dùng nguyên hàm gốc. EM giữ 20 vòng/1.000 MM và
softmax(30*cosine); StatA 10 vòng, inner 5, kNN 3, prior scale 100. Không đổi solver.
StatA giữ CLIP khi nhóm <=3 mẫu. Kết quả gồm 11 method, tổng 25 method/seed.

## Lệnh launcher

Smoke trước:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job fakeddit_hierarchy_smoke --use-kaggle true --wait
```

Sau khi smoke hợp lệ, chạy đủ dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job fakeddit_hierarchy_dev --use-kaggle true --wait
```

Một lệnh kiểm tra quota, chặn trùng, gửi job, theo dõi 30 giây/lần, tải ZIP và
kiểm tra checksum. `--prepare-only` chỉ chuẩn bị code/config, không API, dữ liệu
hoặc model. Mất kết nối dùng `--resume-stage` staging đã in, không gửi lại job.
Code whitelist nằm trong notebook; không upload lại Fakeddit hoặc code Dataset.

## Đọc kết quả

Ưu tiên macro-F1 đủ 6 lớp, kèm accuracy, balanced accuracy và recall từng lớp fake.
Không báo top-5 cho các nhánh chỉ có nhãn cứng EM/Majority.

- `summary.tsv`: 11 method với mean/sample std, không chọn seed tốt nhất.
- `metrics_by_seed.tsv`, `per_class.json`, `predictions.tsv`: cùng cohort cho 25 lượt.
- `comparisons.tsv`: corrected/harmed và delta accuracy; image-only so flat_clip_image,
  legacy MM so legacy_flat_mm. Đây là đối chiếu chất lượng, không tự chọn winner.
- `routing.json`: binary accuracy và recall fake/real của gate sau khi predict xong.
- `oracle_diagnostic.json`: dùng **binary label thật**, chẩn đoán trần hiệu quả fine
  cho ảnh và MM. Không xếp oracle vào bảng phương pháp hay dùng nó để định tuyến thật.
- `replay_scores.npz`: feature ảnh/tiêu đề, raw prototype, posterior CLIP, phiếu,
  group IDs và prediction; allow_pickle=False. Nhãn thật chỉ dùng đánh giá sau predict.
- `run.json`/`completed.json`: source cohort, profile/config/code/requirements hash,
  runtime model, checksum đầu ra.

Không chạy DCLIP/Waffle hoặc CALIP trong job này. Không trộn mô tả CIFAR với
taxonomy Fakeddit. Không thay chính sách lọc/chuẩn bị dữ liệu hiện có.
Dev đã dùng phát triển trước đây, nên chưa là xác nhận độc lập trên test.

## Kiểm tra local

Đã kiểm tra: nhãn/tie đa lớp, group giống nhau, original reference khớp hàm predict
Fakeddit cũ trên fixture singleton, end-to-end 5 ảnh tổng hợp với solver giả,
oracle tách khỏi bảng phương pháp, whitelist notebook và launcher không gọi API
khi prepare. Chưa chạy ảnh thật hay solver full trong lượt triển khai.

```sh
venv/bin/python -m unittest discover -s tests -p test_fakeddit_hierarchy.py -v
```
