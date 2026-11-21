# DCLIP → StatA trên đủ DTD

Đã có lượt full: xem [báo cáo kiểm tra](DCLIP_STATA_REVIEW.md). DCLIP–StatA đạt 66,90% accuracy trung bình, chưa hơn CLIP–StatA hoặc mốc Waffle full DTD đã có. Các lệnh dưới đây được giữ để tái hiện khi cần.

Job `dclip_stata_dtd` đánh giá **đủ 1.880 ảnh DTD test partition 1, 47 lớp**, cùng MetaCLIP2 ViT-H/14 và dataset Kaggle benchmark hiện có. Đây là đánh giá phát triển trên split đã xem một phần kết quả trước đó, không phải xác nhận độc lập trên dữ liệu chưa đụng tới. Waffle–StatA pilot 256 ảnh vẫn giữ nguyên và không được so ngang với lượt full này.

## Chạy

Smoke 5 ảnh (đủ năm phương pháp và ba grouping seed):

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job dclip_stata_dtd --max-samples 5 --use-kaggle true --wait
```

Sau khi smoke thành công, chạy full:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job dclip_stata_dtd --use-kaggle true --wait
```

Thêm `--prepare-only` để kiểm tra offline. Không cần upload lại dataset ảnh. Kernel riêng: `nguyenhung3105/vlm-final-dtd-dclip-stata-full`; smoke thêm hậu tố `-smoke`. Mỗi notebook chạy đúng một lượt, chỉ nạp model và mã hóa ảnh một lần; không tự chạy full sau smoke. Timeout 7.200 giây là giới hạn cấu hình, chưa phải thời gian đo. Mất kết nối sau push: dùng `--resume-stage` launcher in ra để theo dõi/tải đúng version đã gửi.

## Phương pháp

| Phương pháp | Cấu hình | Số lượt |
|---|---|---:|
| CLIP | Prompt photo hiện có | 1 |
| CLIP–StatA | StatA nhận unit CLIP prototype | 3 seed |
| DCLIP | Mean embedding descriptor, giữ aggregation gốc | 1 |
| DCLIP unit | Chuẩn hóa lại mean descriptor prototype | 1 |
| DCLIP–StatA | StatA nhận unit DCLIP prototype làm prior/khởi tạo | 3 seed |

DCLIP dùng descriptor DTD từ Waffle repository đã pin commit và bộ sinh prompt DCLIP tác giả, cùng lựa chọn descriptor của benchmark chính trong dự án. JSON được sắp theo thứ tự lớp benchmark trước khi gọi tác giả. DCLIP là bộ prompt cố định, không có bảy seed ngẫu nhiên như Waffle; ký hiệu `p0` chỉ định danh bank duy nhất. Trường `waffle_pairs` trong cấu hình dùng chung API, không sinh Waffle trong job này.

StatA giữ nguyên solver và grouping seed 0/1/2, nhóm tối đa 75 ảnh, cùng features và nhóm ảnh giữa CLIP–StatA/DCLIP–StatA. Chỉ nhận feature/prototype không nhãn; nhãn chỉ dùng tính metrics sau khi tạo mọi posterior. Không fine-tune, không fusion posterior, không thêm semantic hoặc text bài đăng. Cấu hình ở `configs/dclip_stata.yaml`.

## Đọc kết quả

Output: `results/final/kaggle/dclip_stata_dtd_<id>/run/`. Gửi `summary.tsv` và `comparisons.tsv` để phân tích. `metrics_by_seed.tsv` có 9 lượt; các phương pháp StatA báo trung bình/sample std ddof=1 qua ba grouping seed, không chọn seed tốt nhất và không coi std là khoảng tin cậy.

So DCLIP–StatA với **DCLIP unit** để tách hiệu ứng chuẩn hóa, và với **CLIP–StatA** để đo đóng góp descriptor. `comparisons.tsv` lưu ba đối chiếu ghép cặp cùng grouping seed, delta accuracy và số sửa sai/làm sai. `replay_scores.npz` lưu feature, prototype, posterior, group ID và ID/nhãn; đọc bằng `allow_pickle=False` để kiểm tra lại offline. `run.json`/`completed.json` xác minh checksum artifact.

Chưa thêm Waffle–semantic–StatA hoặc chuyển sang EuroSAT trong job này. Nếu DCLIP–StatA không hơn các đối chứng, giữ nguyên kết quả; không tự chỉnh tham số theo điểm test từng dataset.
