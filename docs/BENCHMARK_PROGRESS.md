# Tiến độ và kiểm tra benchmark tham chiếu

Đã hoàn tất **8/8 dataset**, **160,093 ảnh**, **88 lượt đánh giá**, **1,761,023 dòng dự đoán**.
Model: OpenAI CLIP ViT-L/14, 224px; mỗi dataset có 11 lượt, Waffle dùng đủ 7 seed.

| Dataset | Ảnh | CLIP photo (%) | DCLIP photo (%) | Waffle, 7 seed (%) | Gần khớp 3 dòng Waffle | Gần khớp cả 5 đối chiếu |
|---|---:|---:|---:|---:|---:|---:|
| dtd | 1,880 | 52.979 | 56.223 | 53.822 | 0/3 | 0/5 |
| imagenetv2 | 10,000 | 67.910 | 69.730 | 69.497 | 3/3 | 5/5 |
| imagenet | 50,000 | 73.382 | 75.244 | 75.278 | 3/3 | 3/5 |
| cub | 5,794 | 62.254 | 63.566 | 64.162 | 3/3 | 4/5 |
| food101 | 25,250 | 92.539 | 92.840 | 93.275 | 3/3 | 3/5 |
| pets | 3,669 | 93.186 | 93.895 | 91.921 | 2/3 | 2/5 |
| eurosat | 27,000 | 56.048 | 58.533 | 61.481 | 1/3 | 1/5 |
| places365 | 36,500 | 40.063 | 42.573 | 42.122 | 1/3 | 2/5 |

Ngưỡng gần khớp đặt trước: ≤0,10 điểm phần trăm so CSV đầy đủ hoặc paper khi không có CSV. Tổng 16/24 đối chiếu chính và 20/40 đối chiếu số học đạt ngưỡng.
Hoàn tất chạy đủ dữ liệu không có nghĩa mọi kết quả đã tái lập khớp paper.

## Ba dataset mới

Mỗi ô dưới đây là chênh lệch thực tế trừ CSV tác giả, đơn vị điểm phần trăm.

| Dataset | CLIP photo | DCLIP photo | Waffle |
|---|---:|---:|---:|
| pets | -0.109021 | +0.000003 | -0.062298 |
| eurosat | +0.018518 | -0.185187 | +0.313758 |
| places365 | -0.399999 | -0.024658 | -0.140117 |

Pets: DCLIP photo và Waffle gần khớp; CLIP photo lệch −0,109021 điểm %, vừa vượt ngưỡng.
EuroSAT: chỉ CLIP photo gần khớp trong ba dòng chính.
Places365: chỉ DCLIP photo gần khớp trong ba dòng chính.

## Đã kiểm tra

SHA256 của cả 8 run và tất cả artifact đã được xác minh lại; số tham chiếu khớp registry offline.
Ba run mới được tính lại accuracy, macro-F1, balanced accuracy, top-5, per-class và summary mean/std từ dự đoán lưu.
Sai số lớn nhất là 7,11e-15. Replay top-1/top-5 nhất quán xác suất FP16 có xét đồng hạng.
Pets: đủ 37 lớp, ánh xạ theo tên lớp đúng; tên giống từ filename khớp nhãn native của 3.669 mẫu.
Sinh lại 11 bank prompt của cả ba dataset bằng code tác giả khớp toàn bộ frozen GPU plan.
Không mở ảnh hoặc chạy lại inference trong lượt kiểm tra này.

## Giới hạn và bước còn lại

Một số kết quả vẫn vượt ngưỡng; chưa xác định đầy đủ nguyên nhân. Không đổi prompt hay nới ngưỡng từ test.
EuroSAT dùng toàn bộ 27.000 ảnh theo Waffle; không xác nhận cùng split EuroSATVal trong DCLIP gốc.
DCLIP DTD loader dùng val, dự án dùng test partition 1. Đối chiếu prompt gốc cần giữ riêng giới hạn split.
Kết quả cao hơn paper không tự chứng minh cải tiến của phương pháp.
Đã hoàn tất bộ tham chiếu OpenAI; model chính MetaCLIP2 hiện có DTD và ImageNetV2 (2/8), xem [metaclip2_progress.json](metaclip2_progress.json).
Chưa thấy run Fakeddit bản cuối hoàn tất dưới results/final; cần dev và lock trước test.
Chưa gửi GPU job mới.

Chi tiết: [benchmark_progress.json](benchmark_progress.json), [40 phép đối chiếu](benchmark_comparison.tsv), [mốc paper đã lưu](PAPER_RESULTS.md).

## Báo cáo từng run

- [dtd](../results/final/kaggle/paper_reference_dtd_vitl14_4de91a58e03b/review/result_review.md)
- [imagenetv2](../results/final/kaggle/paper_reference_imagenetv2_vitl14_0e9e6407ce90/review/result_review.md)
- [imagenet](../results/final/kaggle/paper_reference_imagenet_vitl14_ac110d7d823a/review/result_review.md)
- [cub](../results/final/kaggle/paper_reference_cub_vitl14_69338b83481b/review/result_review.md)
- [food101](../results/final/kaggle/paper_reference_food101_vitl14_6f9a5c56259b/review/result_review.md)
- [pets](../results/final/kaggle/paper_reference_pets_vitl14_a706e908983e/review/result_review.md)
- [eurosat](../results/final/kaggle/paper_reference_eurosat_vitl14_748bccf4b96c/review/result_review.md)
- [places365](../results/final/kaggle/paper_reference_places365_vitl14_1a913aff262c/review/result_review.md)
