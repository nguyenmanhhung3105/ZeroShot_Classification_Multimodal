# Kiểm tra rerank CIFAR-100 dev — 10/11/2026

Run: `results/final/kaggle/cifar100_rerank_dev_a6781f055ae7/run/`.
Kaggle version: `nguyenhung3105/vlm-cifar100-rerank-dev-v1/1`.

## Kết quả bốn đối chứng

| Phương pháp | Accuracy | Macro-F1 | Top-5 |
|---|---:|---:|---:|
| Flat tên lớp | 86,75% | 86,44% | 97,04% |
| Flat mô tả | **88,38%** | **88,37%** | **97,81%** |
| Top-2 nhóm + tên lớp | 86,76% | 86,45% | 96,70% |
| Top-2 nhóm + mô tả | 88,10% | 88,09% | 96,83% |

Mô tả cải thiện flat 1,63 điểm phần trăm, sửa đúng 479 và làm sai 316 mẫu.
Lọc nhóm với tên lớp chỉ tăng 0,01 điểm phần trăm, sửa 5 và làm sai 4 mẫu.
Top-2 + mô tả tăng 1,35 điểm phần trăm so với tên lớp gốc, nhưng **giảm 0,28
điểm phần trăm so với flat dùng cùng mô tả**, sửa 33 và làm sai 61 mẫu.
Chưa có bằng chứng việc chia tầng cải thiện accuracy; lợi ích hiện đến từ mô tả.

## Phân tích paired trên cùng mẫu

Bootstrap resampling cặp đúng/sai trên 10.000 mẫu dev, 10.000 lần, seed 0;
CI percentile 95% cho chênh lệch accuracy:

| So sánh | Delta (điểm %) | CI 95% (điểm %) |
|---|---:|---:|
| Flat mô tả − flat tên lớp | +1,63 | [+1,08; +2,17] |
| Top-2 tên lớp − flat tên lớp | +0,01 | [−0,05; +0,07] |
| Top-2 mô tả − flat tên lớp | +1,35 | [+0,81; +1,88] |
| Top-2 mô tả − flat mô tả | −0,28 | [−0,47; −0,09] |

McNemar exact, hai phía, p chưa điều chỉnh lần lượt là 8,16e−9; 1,0; 6,47e−7;
0,00508. Đây là phân tích khám phá trên dev đã được dùng phát triển; CI/p không
loại bỏ thiên lệch do lựa chọn phương pháp và không thay thế xác nhận trên test.
Artifact thống kê: `results/final/kaggle/cifar100_rerank_dev_a6781f055ae7/paired_dev_analysis.json`.

Nhánh tên lớp và mô tả có thể bổ sung nhau: oracle chọn đúng nếu một trong hai
đúng đạt 91,54%. Đây chỉ là chẩn đoán dùng nhãn thật, không phải thuật toán thực tế.
Không chọn gate bằng nhãn từng mẫu, không coi oracle là kết quả mô hình.

## Xác minh kỹ thuật

- Completion/run và toàn bộ output_sha256 khớp.
- Đủ 10.000 ID duy nhất mỗi phương pháp, cùng tập ID và nhãn giữa bốn đối chứng.
- Replay từ fine/description logits khớp bốn phân phối lưu và candidate_groups.
- ID/nhãn khớp run cũ; fine logits có chênh lệch tối đa **0**, dự đoán baseline
  khớp **100%**. Thay đổi chất lượng không do đổi cohort hoặc backbone.
- MetaCLIP2 worldwide ViT-H/14, CUDA FP32; batch ảnh 8, batch prompt 64.
- Router top-1 93,42%, top-2 giữ nhóm đúng 97,42%; đây không phải accuracy cuối.

Timer runner: 1.443,26 giây (~24 phút 03 giây), bắt đầu sau khởi tạo model.
Log job kết thúc ở 1.574,56 giây (~26 phút 15 giây), không gồm queue.

## Hướng tiếp theo

Giữ flat mô tả làm baseline mạnh mới. Không tiếp tục dùng lọc top-2 bắt buộc
như một cải tiến đã được chứng minh. Nếu phát triển hai tầng, cần chẩn đoán gate
theo mức bất định và tín hiệu bổ sung ở tầng hai, có baseline cùng prompt và
đối chứng gate; tránh chỉ đổi cách sắp/xóa lớp rồi tuyên bố tăng chất lượng.
Phân tích bổ sung có thể dùng logits/features đã lưu, chưa cần chạy lại ảnh/GPU.
Chưa mở test, chưa sửa prompt/tham số và chưa tự gửi job mới trong lượt review.
