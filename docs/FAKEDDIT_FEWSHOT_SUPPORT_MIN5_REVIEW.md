# Kiểm tra đối chứng phân bổ support tầng 1 min5

Run `results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run/` hoàn tất đủ 3.000 dev min5, 5 seed, hai support policy và ba modality. C=1, class_weight=balanced, frozen MetaCLIP2 và công thức feature giữ nguyên. Chưa mở test, không chạy tầng hai. Không gửi job mới trong lượt kiểm tra.

Đã xác minh checksum toàn bộ artifact, 10 support set đúng quota/384 ID duy nhất, hai policy chung đúng 256 ID mỗi seed. 30 posterior replay từ feature/coef/intercept khớp; metrics, trung bình/sample std, recall, số sửa sai/làm sai đều khớp. Tất cả 15 dự đoán baseline legacy khớp chính xác run baseline trước trên cùng dev. Hợp train được mã hóa là 1.591 ID. Timer runner khoảng 1.066 giây, không phải toàn bộ thời gian queue/setup.

## Kết quả tổng

| Modality | Support | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---|---:|---:|---:|
| Ảnh | 64 Real + 320 Fake | 77,83 ± 1,07 | 76,91 ± 1,20 | 76,91 ± 1,30 |
| Ảnh | 192 Real + 192 Fake | 79,93 ± 0,63 | 79,64 ± 0,62 | 80,77 ± 0,60 |
| Tiêu đề | 64 Real + 320 Fake | 77,17 ± 0,75 | 75,43 ± 1,14 | 74,88 ± 1,30 |
| Tiêu đề | 192 Real + 192 Fake | 78,66 ± 0,57 | 77,95 ± 0,63 | 78,21 ± 0,76 |
| Ảnh–tiêu đề | 64 Real + 320 Fake | 81,22 ± 0,75 | 80,16 ± 0,88 | 79,81 ± 1,02 |
| Ảnh–tiêu đề | 192 Real + 192 Fake | **82,41 ± 0,47** | **82,04 ± 0,43** | **82,87 ± 0,36** |

Nhánh 192/192 tăng accuracy và macro-F1 ở cả năm seed, với cả ba modality. Đối với ảnh–tiêu đề, tăng accuracy trung bình 1,19 điểm phần trăm, macro-F1 1,89 và balanced accuracy 3,06. Std giữa seed là biến thiên trên cùng dev, không phải khoảng tin cậy và năm support có thể chồng lấp.

## Đánh đổi Real/Fake

| Chỉ số ảnh–tiêu đề | 64 Real + 320 Fake | 192 Real + 192 Fake |
|---|---:|---:|
| Recall Real | 72,71% | 85,21% |
| Recall Fake | 86,91% | 80,53% |
| Real bị nhận thành Fake/seed | 328,0 | 177,8 |
| Fake bị nhận thành Real/seed | 235,4 | 350,0 |

Nhánh mới cứu trung bình 150,2 mẫu Real nhưng bỏ sót thêm 114,6 Fake, lợi ích ròng 35,6 mẫu đúng/seed. Mỗi seed đều tăng recall Real và giảm recall Fake. Vì vậy kết quả chưa đạt mục tiêu mạnh hơn ở Real mà vẫn giữ nguyên khả năng phát hiện Fake.

Tỷ lệ nhận ra là Fake trong từng nhóm (không phải accuracy phân loại subtype tầng hai):

| Fake subtype | Mẫu dev | Legacy (%) | Balanced (%) |
|---|---:|---:|---:|
| Satire | 182 | 80,77 | 74,51 |
| False Connection | 572 | 71,22 | 56,08 |
| Imposter | 64 | 71,25 | 59,06 |
| Manipulated | 868 | 99,54 | 99,52 |
| Misleading | 112 | 88,04 | 80,36 |

False Connection giảm rõ nhất về độ lớn và có nhiều mẫu. Không được kết luận chỉ từ support binary cân bằng rằng mọi loại Fake đều tốt hơn. Thay ngân sách đồng thời thay độ đa dạng Real, độ phủ Fake và trọng số mỗi mẫu (dù class_weight=balanced cố định); chưa tách nguyên nhân cụ thể.

## Quyết định

Giữ cả hai làm ứng viên: 192/192 tốt hơn về accuracy, macro-F1 và balanced accuracy ở quyết định mặc định; 64/320 giữ recall Fake tốt hơn. Không tự thay baseline chỉ vì một metric tăng. Đây vẫn là few-shot linear probing và đối chứng ngân sách, chưa phải phương pháp mới hoặc test độc lập.

Bước tiếp theo hợp lý là đối chiếu hai gate ở **cùng mục tiêu recall Fake**, phân tích trade-off ngưỡng từ posterior đã lưu, không cần GPU. Nếu chọn ngưỡng, chỉ chọn trên phần selection của dev và kiểm tra phần audit tách biệt; vì full dev đã được xem, đây chỉ là kiểm tra phát triển nội bộ, không gọi audit là test độc lập. Chưa mở test hoặc chuyển tầng hai.
