# Đối chứng chỉ dùng ảnh trên CrisisMMD

Mục tiêu là so **ảnh → prompt lớp → nhãn** trên cùng mẫu CrisisMMD. Caption/tweet của từng mẫu không đóng góp điểm. Prompt lớp vẫn là văn bản được mã hóa bởi text encoder. Lượt dự án dùng MetaCLIP2 và prompt hiện tại; lượt upstream dùng OpenAI CLIP ViT-B/32 với hàm WaffleCLIP/DCLIP của tác giả. DCLIP dùng descriptor JSON của taxonomy CrisisMMD, không phải descriptor nguyên bản của benchmark trong paper.

## Đường dẫn kết quả

```text
results/image_only_baseline/crisismmd/<task>/samples_N/<model>/<lượt>/
results/image_only_adapted_<method>/crisismmd/<task>/samples_N/<model>/<lượt>/
results/image_only_upstream_<method>/crisismmd/<task>/samples_N/openai_clip_vit_b_32/<lượt>/
```

Job `image_only` của standard/adapted khóa `image_weight=1.0` và `image_text_relation.enabled=false`; upstream khóa `outputs=[original_image]` và tắt quan hệ. `summary.tsv` ở nhánh dự án ghi `accuracy`/`macro_f1` của dự đoán ảnh-only. Nhánh upstream ghi `output_variant=original_image`. Không lấy `macro_f1` fusion của lượt `baseline`/`adapted_*` cũ làm điểm ảnh-only; nếu cần dùng lượt cũ, lấy `image_macro_f1` và tái dựng Accuracy từ điểm ảnh trong `raw_predictions.tsv`.

Năm lượt upstream lịch sử mà toàn bộ dòng summary là `original_image` đã được chuyển sang `image_only_upstream_*`. Archive có cả `original_image` và `multimodal` vẫn ở `upstream_suite`; không tách archive hoặc sửa file chứng cứ. Đường dẫn gốc và hiện tại có trong `results/_system/migration_map.json`.

## Lệnh chạy một lượt mỗi lần

Chạy từ thư mục gốc dự án, sau khi kích hoạt venv. Trước tiên có thể thêm `--dry-run` vào từng lệnh để kiểm kế hoạch; ở `data_mode: full_dataset`, bước này vẫn đóng gói dữ liệu local nhưng chưa gửi Kaggle và chưa chạy model.

**Informativeness, 1.000 mẫu:**

```bash
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info_image_only --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_info_image_only --backbone ViT-B/32 --method clip --output original_image --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_info_image_only --backbone ViT-B/32 --method waffle --output original_image --seed 0 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_info_image_only --backbone ViT-B/32 --method dclip --output original_image --max-samples 1000 --use-kaggle true
```

`clip` là đối chứng tên lớp của WaffleCLIP; thêm `--method dclip_labels` nếu muốn đối chứng tên lớp theo template DCLIP. Mỗi lệnh Kaggle chỉ dùng một model, một phương pháp và một output. Chờ lệnh trước hoàn tất rồi mới gửi lệnh sau.

**Humanitarian, 1.000 mẫu:**

```bash
venv/bin/python -B scripts/launch_runs.py --job crisismmd_humanitarian_image_only --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_humanitarian_image_only --backbone ViT-B/32 --method clip --output original_image --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_humanitarian_image_only --backbone ViT-B/32 --method waffle --output original_image --seed 0 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_humanitarian_image_only --backbone ViT-B/32 --method dclip --output original_image --max-samples 1000 --use-kaggle true
```

**Đối chứng cùng backbone MetaCLIP2**, tùy chọn sau khi đã chạy nhóm trên: dùng `--job adapted_crisismmd_info_image_only --model metaclip2_vith14 --method waffle --seed 0` hoặc `--method dclip`, kèm `--max-samples 1000 --use-kaggle true`. Đây là Waffle/DCLIP **adapted** của dự án, không phải code prompt nguyên bản của paper.

## Kiểm tra tính so sánh được

Chỉ so kết quả cùng task, cùng tập mẫu và thứ tự nhãn. CrisisMMD có thể lặp `tweet_id`; cần ghép/kiểm thêm text, tên ảnh và nhãn, không chỉ ID. Báo Accuracy, Macro-F1 và chỉ số từng lớp. Nếu backbone khác nhau, kết quả so **hệ thống hoàn chỉnh**; không quy chênh lệch riêng cho thuật toán prompt. Muốn quy cho prompt, dùng cùng backbone và preprocessing, như nhóm adapted cùng MetaCLIP2, đồng thời ghi rõ đây không phải bản nguyên gốc.

Trong log lịch sử informativeness 1.000 mẫu, bốn lượt dưới đây có cùng dãy `(id, text, image_path, y_true)` gồm 1.000 dòng; 893 `tweet_id` riêng biệt nên kiểm chỉ theo ID sẽ ghép sai. Các số chỉ là đối chứng thăm dò vì split nguồn chưa được xác minh độc lập:

| Hệ thống ảnh-only | Backbone | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Prompt dự án, lấy nhánh ảnh từ baseline cũ | MetaCLIP2 ViT-H/14 | 0,788 | 0,788 |
| CLIP tên lớp upstream | OpenAI CLIP ViT-B/32 | 0,501 | 0,490 |
| Waffle upstream, seed 0 | OpenAI CLIP ViT-B/32 | 0,557 | 0,514 |
| DCLIP upstream | OpenAI CLIP ViT-B/32 | 0,775 | 0,768 |

Điểm ảnh-only của baseline cũ được kiểm tra từ `score_image_*` trong log và khớp `image_macro_f1` của summary. DCLIP ở bảng trên dùng descriptor CrisisMMD của dự án; không diễn giải mức điểm này là điểm tái lập paper trên benchmark gốc.
