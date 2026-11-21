# Kiểm tra full few-shot tầng 1 min5

Run `results/final/kaggle/fakeddit_fewshot_task1_min5_4051d18f8c31/run/` hoàn tất 3.000 mẫu dev min5, MetaCLIP2 đóng băng, Logistic Regression C=1 balanced, 384 support/seed (64 mỗi lớp 6way), năm seed 0–4. Nhãn binary trong artifact: 0=Real, 1=Fake. Dev gồm 1.202 Real và 1.798 Fake. Chưa mở test, không chạy tầng hai.

Đã xác minh checksum toàn bộ artifact; support đúng ngân sách, không trùng ID train/dev, support cache khớp ID/nhãn. Replay 15 classifier bằng feature/coef/intercept khớp posterior; metrics từng seed, trung bình/sample std, recall và số sửa sai/làm sai đều khớp. Mapping nhãn dev tái tạo từ 6way đúng. Cohort ID/hash ảnh/thứ tự khớp lượt zero-shot min5 trước. Không fit model, mã hóa ảnh hoặc gửi job trong lần kiểm tra này.

| Modality | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---:|---:|---:|
| Ảnh | 77,83 ± 1,07 | 76,91 ± 1,20 | 76,91 ± 1,30 |
| Tiêu đề | 77,17 ± 0,75 | 75,43 ± 1,14 | 74,88 ± 1,30 |
| Ảnh–tiêu đề | **81,22 ± 0,75** | **80,16 ± 0,88** | **79,81 ± 1,02** |

Ảnh–tiêu đề hơn ảnh accuracy trung bình 3,39 điểm phần trăm; hơn tiêu đề 4,05. Chênh lệch accuracy theo seed so ảnh: +3,00 / +4,30 / +3,30 / +3,67 / +2,67 điểm phần trăm; đều dương. So tiêu đề cũng dương cả năm seed. Đây là kết quả trên cùng support và cohort, cho thấy văn bản bổ trợ ảnh trong cấu hình few-shot này; khác nhận định text yếu của nhánh zero-shot trước.

| Modality | Recall Real (%) | Recall Fake (%) | Fake→Real mỗi seed | Real→Fake mỗi seed |
|---|---:|---:|---:|---:|
| Ảnh | 72,25 | 81,57 | 331,4 | 333,6 |
| Tiêu đề | 63,34 | 86,41 | 244,4 | 440,6 |
| Ảnh–tiêu đề | **72,71** | **86,91** | **235,4** | **328,0** |

Ảnh–tiêu đề giảm trung bình 96 mẫu Fake bị nhận thành Real so ảnh, trong khi recall Real chỉ tăng 0,47 điểm phần trăm trung bình. Không có nghĩa mọi seed đều tăng recall Real: seed 0 và 4 có recall Real thấp hơn ảnh riêng. Cần phân tích đánh đổi Real/Fake thay vì chỉ tối đa accuracy tổng. Năm support có thể chồng lấp; sample std không phải khoảng tin cậy và chưa có kiểm định paired độc lập.

## Quyết định

Giữ ảnh–tiêu đề làm **baseline few-shot tầng 1 min5** mạnh nhất trong ba đối chứng hiện tại. Đây là linear probing tiêu chuẩn, không phải bằng chứng phương pháp mới. Tập trung phân tích lỗi Real/Fake và cải tiến có đối chứng trên dev trước khi triển khai tầng hai; không tự chỉnh ngưỡng hoặc mở test trong lượt kiểm tra này.

Kết quả 78,05% few-shot cũ dùng cohort khác; không quy chênh lệch lên 81,22% cho cải tiến thuật toán. StatA zero-shot 72,02% nằm trên cùng dev min5 nhưng dùng chế độ không nhãn train khác: có thể báo song song với ngân sách nhãn rõ ràng, không coi là so sánh cùng chế độ học.

Hợp support năm seed có 1.172 mẫu train được mã hóa một lần chung cho mọi modality; mỗi classifier vẫn chỉ học từ 384 mẫu của seed đó. Timer encoding ~772 giây, toàn runner ~776 giây, không gồm toàn bộ thời gian queue/setup. Features và linear_models đã lưu để phân tích offline tiếp mà không cần GPU.
