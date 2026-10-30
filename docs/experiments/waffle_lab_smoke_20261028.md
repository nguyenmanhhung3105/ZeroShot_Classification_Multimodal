# Chạy thử Waffle lab ngày 28/10/2026

Đã chạy model thật OpenAI CLIP ViT-B/32 trên CPU, seed 0, chỉ ảnh, năm bản ghi đầu CrisisMMD informativeness. Không huấn luyện, không dùng tweet để dự đoán.

18 kiểm thử trong experiments.waffle_lab.tests.test_lab đã đạt. Bốn lượt dùng cùng cohort và giao thức.

| Phương pháp | Accuracy | Macro-F1 | Kết quả |
|---|---:|---:|---|
| clip | 80.00% | 0.7619 | [summary.tsv](../../results/waffle_lab_clip/crisismmd/informativeness/samples_5/openai_clip_vit_b_32/informativeness_2fb1f37b170e/summary.tsv) |
| waffle | 100.00% | 1.0000 | [summary.tsv](../../results/waffle_lab_waffle/crisismmd/informativeness/samples_5/openai_clip_vit_b_32/informativeness_ed1bcc337179/summary.tsv) |
| semantic | 60.00% | 0.3750 | [summary.tsv](../../results/waffle_lab_semantic/crisismmd/informativeness/samples_5/openai_clip_vit_b_32/informativeness_125c7cbb169a/summary.tsv) |
| semantic_waffle | 60.00% | 0.3750 | [summary.tsv](../../results/waffle_lab_semantic_waffle/crisismmd/informativeness/samples_5/openai_clip_vit_b_32/informativeness_24fac0d152e3/summary.tsv) |

Đây là smoke test kỹ thuật, không dùng năm mẫu để chọn phương pháp hoặc kết luận hiệu quả. semantic dùng ba descriptor cố định/lớp; waffle và semantic_waffle dùng 30 prompt/lớp, nên ngân sách chưa cân bằng. Không thay đổi code thuật toán hoặc cấu hình thí nghiệm chính.

## Tái tạo từng lượt

```sh
venv/bin/python -B experiments/waffle_lab/run.py --task informativeness --variant clip --seed 0 --max-samples 5
venv/bin/python -B experiments/waffle_lab/run.py --task informativeness --variant waffle --seed 0 --max-samples 5
venv/bin/python -B experiments/waffle_lab/run.py --task informativeness --variant semantic --seed 0 --max-samples 5
venv/bin/python -B experiments/waffle_lab/run.py --task informativeness --variant semantic_waffle --seed 0 --max-samples 5
```

Bước đánh giá tiếp theo: cố định thiết kế, chạy nhiều seed trên cohort lớn được cho phép và xác nhận trên tập độc lập. Không suy diễn các chỉ số smoke test thành kết quả nghiên cứu.
