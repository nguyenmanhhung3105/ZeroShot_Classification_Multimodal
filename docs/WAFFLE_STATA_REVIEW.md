# Kiểm tra hai pilot Waffle–StatA

Đã xác minh checksum của hai run hoàn tất, tính lại toàn bộ metrics (accuracy/macro-F1/balanced accuracy/top-5), trung bình/sample std và 21 đối chiếu sửa sai/làm sai mỗi run từ posterior đã lưu. Mỗi run đúng 256 ID duy nhất, 39 lượt dự đoán, cùng MetaCLIP2; không phải full DTD/EuroSAT. Không gửi job mới.

| Phương pháp | DTD accuracy (%) | EuroSAT accuracy (%) |
|---|---:|---:|
| CLIP | 67,19 | **64,84** |
| CLIP–StatA | 67,06 ± 0,45 | 60,29 ± 1,63 |
| Waffle | 70,93 ± 0,74 | 63,17 ± 0,89 |
| Waffle unit | 71,54 ± 0,42 | 62,72 ± 1,25 |
| Waffle–StatA | **72,32 ± 1,01** | 59,64 ± 1,93 |

DTD: Waffle–StatA tăng accuracy trung bình 1,40 điểm phần trăm so Waffle và 0,78 so Waffle unit. So Waffle unit cùng prompt seed: thắng 13, hòa 3, thua 5 trên 21 tổ hợp. Macro-F1 tăng từ 68,55% (unit) lên 69,87%, nhưng balanced accuracy gần như không đổi (71,876% so 71,875%) và top-5 giảm từ 97,66% xuống 96,45%. Vì vậy không tốt hơn trên mọi metric.

EuroSAT: Waffle–StatA giảm 3,53 điểm phần trăm so Waffle, giảm 3,09 so Waffle unit và giảm 5,21 so CLIP. Thua Waffle/unit ở 20/21 tổ hợp. Macro-F1 chỉ 53,19%, so CLIP 63,18%. Pilot này không ủng hộ dùng Waffle–StatA làm mặc định chung.

Kết luận: giả thuyết bổ trợ có tín hiệu trên DTD nhưng chưa tổng quát sang EuroSAT. Chưa chốt phương pháp chung hoặc tuyên bố cải thiện benchmark. Nếu xác nhận hướng DTD, cần cấu hình chạy đủ 1.880 ảnh, giữ đối chứng unit và seed đã định trước; job hiện tại vẫn chỉ hỗ trợ 5/256 ảnh. EuroSAT cần phân tích lỗi từng lớp và hành vi StatA trước khi mở rộng, không chỉ tăng số seed hoặc chọn seed tốt nhất.

Std của Waffle–StatA là biến thiên trên 7 prompt × 3 grouping seed cùng một cohort; không phải 21 phép thử độc lập hay khoảng tin cậy. Các subset đã dùng phát triển, không gọi kết quả này là xác nhận độc lập.

Nguồn:

- `results/final/kaggle/waffle_stata_dtd_218418a64888/run/`
- `results/final/kaggle/waffle_stata_eurosat_a1a6f8433ee9/run/`
