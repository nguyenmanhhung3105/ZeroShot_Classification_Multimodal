# Kiểm chứng Fakeddit dev 2way một tầng

Ngưỡng chuẩn bị text mới: `min_text_length=5` trong các script chuẩn bị và cấu hình v2 tại `archive/legacy_20261103/` (trước đây 20). Giữ cách lọc cũ: `len(text.replace(' ', '')) > 5`, không phải số từ/token. Pipeline suy luận hiện tại đọc cohort v2 đã cố định và không áp dụng bộ lọc này lại. Đổi mặc định không bổ sung các mẫu ngắn đã bị loại; cần chuẩn bị cohort mới riêng và cập nhật nguồn Kaggle trước khi đánh giá ảnh hưởng. Chưa tạo lại dữ liệu hoặc thay cohort/kết quả hiện có.

Đã thu lượt đủ 3.000 dev: xem [báo cáo kiểm tra kết quả](FAKEDDIT_TASK1_REVIEW.md). StatA ảnh riêng đạt accuracy trung bình 65,88%, cao hơn fusion ảnh–tiêu đề 65,43%; chưa chứng minh fusion tốt hơn StatA riêng. Các lệnh bên dưới được giữ để chủ động tái hiện.

Job `fakeddit_task1_dev` dùng MetaCLIP2 ViT-H/14 cố định trên 3.000 mẫu dev v2 (cohort con của dự án, không phải toàn bộ Fakeddit). Đây là dữ liệu phát triển, bao gồm 256 mẫu pilot đã xem kết quả. Không gọi là xác nhận độc lập. Chỉ chuẩn bị code/job; người dùng tự gửi Kaggle.

## Chạy từ terminal

Kiểm tra luồng trên 5 mẫu dev trước, vẫn đủ sáu phương pháp và ba seed:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job fakeddit_task1_dev --max-samples 5 --use-kaggle true --wait
```

Khi smoke thành công, chạy đủ cohort:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job fakeddit_task1_dev --use-kaggle true --wait
```

Thêm `--prepare-only` để chỉ đóng gói và kiểm tra offline, không gọi Kaggle hoặc đọc dataset. Notebook private dùng lại `nguyenhung3105/vlm-fakeddit-dev-v2`. Kernel có tên riêng `vlm-final-fakeddit-dev-task1-single-stage`, smoke thêm hậu tố `-smoke`. Mỗi notebook chạy đúng một lần theo mode đã chọn, một lần nạp model và một lần mã hóa chung ảnh–tiêu đề; không chạy lại full ngầm sau smoke. Timeout GPU 7.200 giây là giới hạn cấu hình, không phải dự báo thời gian chạy. Nếu mất kết nối sau push, dùng `--resume-stage` launcher in ra để theo dõi đúng version cũ.

CLI local tương ứng: `run.py fakeddit-task1 --prepare-only`, hoặc bỏ cờ để suy luận khi có dữ liệu/model phù hợp; thêm `--smoke` để chỉ 5 ảnh. CLI này luôn dùng dev.

## Công thức đã cố định trước khi chạy

Ảnh dùng prompt `image_prompts.2way.semantic_v1`; tiêu đề dùng `text_prompts.2way` trong profile hiện tại. Thứ tự lớp `0=fake`, `1=real`. Embedding ảnh, tiêu đề và trung bình embedding prompt đều được chuẩn hóa unit. Đây là các đối chứng mới có kiểm soát, không phải tái hiện baseline legacy dùng prototype chưa chuẩn hóa trong pipeline Fakeddit cũ.

Gọi `C = softmax(100 × image @ image_prototypes.T)`, `T = softmax(25 × title @ text_prototypes.T)` và `S` là posterior StatA trên ảnh. Sáu phương pháp:

| Tên artifact | Xác suất dùng dự đoán | Số lượt |
|---|---|---:|
| `clip_image` | C | 1 |
| `clip_title` | T | 1 |
| `clip_image_title` | 0,75 C + 0,25 T | 1 |
| `stata_image` | S | 3 seed |
| `clip_stata_image` | 0,5 C + 0,5 S | 3 seed |
| `clip_stata_image_title` | 0,375 C + 0,375 S + 0,25 T | 3 seed |

Mỗi phương pháp argmax đúng một lần trên hai lớp. Không phân cấp hoặc gate. StatA chỉ dùng feature ảnh không nhãn, chia nhóm ngẫu nhiên tối đa 75 mẫu với seed 0/1/2; giữ tham số solver hiện có (10 vòng, 5 vòng trong, 3 láng giềng, prior scale 100). Nhóm tối đa 3 ảnh giữ prior CLIP. Nhãn chỉ dùng đánh giá sau khi tính xong mọi posterior. Title dùng cách tokenizer/context hiện có và có thể bị cắt ở context; prompt lớp dùng strict context. Không fine-tune trọng số, không CALIP, không EM, không chạy 6way hoặc test, không tự chọn winner hay tạo test lock.

Cấu hình ở `configs/fakeddit_task1.yaml`: seed, trọng số và scale đã cố định. Không tìm grid theo kết quả lượt này. Smoke chỉ kiểm tra vận hành.

## Đọc kết quả

Launcher tải và xác minh checksum vào `results/final/kaggle/fakeddit_task1_dev_<id>/run/`:

- `summary.tsv`: accuracy, macro-F1, balanced accuracy; trung bình và sample std ddof=1. Baseline tất định tính một lượt, ba phương pháp StatA tính đủ ba seed; không chọn seed tốt nhất. Độ lệch chuẩn giữa seed chia nhóm không phải khoảng tin cậy tổng quát hóa.
- `metrics_by_seed.tsv`, `per_class.json`: từng seed và từng lớp. Không xuất top-5 vì luôn bằng 1 trên hai lớp.
- `comparisons.tsv`: chênh lệch accuracy, số sửa sai/làm sai so với CLIP ảnh và so với baseline ảnh–tiêu đề trên cùng mẫu.
- `predictions.tsv`: ID, nhãn, dự đoán, xác suất fake/real cho mỗi method/seed.
- `branch_diagnostics.json`: confidence, entropy và NLL mỗi method/seed.
- `replay_scores.npz`: feature ảnh/tiêu đề, prototype, posterior, ID/nhãn, group ID mỗi seed; đọc bằng `numpy.load(..., allow_pickle=False)` để kiểm tra offline không cần model.
- `frozen_plan.json`, `cohort.json`, `run.json`, `completed.json`: cấu hình, prompt, cohort, hash code/profile/artifact và thông tin model thực tế. Run này không đủ điều kiện tạo lock bằng `run.py freeze`.

So sánh chính là `clip_stata_image_title` với `clip_image_title`: cần xem cả macro-F1 trung bình, từng seed và từng lớp. Chỉ vượt baseline ảnh chưa đủ chứng minh cải tiến đa phương thức. Nếu không cải thiện, giữ kết quả để phân tích; chưa tự động chuyển sang test hoặc Task 2.
