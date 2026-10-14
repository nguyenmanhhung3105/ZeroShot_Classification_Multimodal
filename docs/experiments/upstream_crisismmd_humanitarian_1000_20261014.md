# Upstream OpenAI CLIP trên 1.000 mẫu CrisisMMD humanitarian

Lượt Kaggle `9cebbce1f45d` chạy OpenAI CLIP ViT-B/32, chỉ dùng ảnh (`original_image`) để đối chiếu bốn cách tạo prompt của WaffleCLIP/DCLIP trên cùng 1.000 mẫu CrisisMMD humanitarian. Không dùng text của bài đăng để ra quyết định, không huấn luyện hoặc chọn tham số theo nhãn đánh giá. WaffleCLIP dùng seed `0` và 15 cặp prompt ngẫu nhiên; cờ quan hệ ảnh–text tắt.

| Phương pháp | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| CLIP baseline | 0.424 | 0.278 | 0.517 |
| WaffleCLIP | 0.474 | 0.301 | 0.501 |
| DCLIP tên lớp | 0.322 | 0.211 | 0.522 |
| DCLIP mô tả | 0.458 | 0.336 | 0.669 |

Giá trị đầy đủ nằm trong [summary.tsv](../../results/experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/summary.tsv). [run_artifacts.zip](../../results/experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/run_artifacts.zip) chứa frozen plan, seed summary và dự đoán từng mẫu. `run_receipt.json` đã được đối chiếu với gói code/data và run ID của lượt Kaggle này.

Đây là áp dụng các hàm tác giả và OpenAI CLIP lên taxonomy/dataset của dự án, không phải tái lập bảng kết quả paper. Mô tả lớp DCLIP do dự án viết từ định nghĩa nhãn, không phải descriptor GPT-3 gốc. Một seed WaffleCLIP chưa đủ đánh giá độ ổn định; cần chạy nhiều seed và báo trung bình/độ phân tán trước khi kết luận. Không so trực tiếp bảng này với job đa phương thức hoặc job dùng bộ mẫu khác.
