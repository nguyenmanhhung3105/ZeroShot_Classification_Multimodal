# Kiểm tra Task 1 trên cohort min5

Run: `results/final/kaggle/fakeddit_task1_dev_min5_b05002bc6b94/run/`. Hoàn tất dev 2way, 3.000 ID duy nhất, MetaCLIP2 ViT-H/14 worldwide FP32/CUDA. Seed chia nhóm 0/1/2, nhóm 75 ảnh và trọng số đúng kế hoạch. Đã xác minh checksum mọi artifact, tính lại metrics cho 12 posterior, trung bình/sample std và số đếm sửa sai/làm sai; đều khớp. Không chạy lại model hoặc gửi job.

Audit cohort min5 ghi `passed`, 9.000 mẫu, ngưỡng 5, seed chọn mẫu 42, kiểm tra chống trùng ID/URL/tiêu đề/ảnh bytes/ảnh pixels. Dev có 1.798 fake và 1.202 real. Baseline đoán toàn fake đạt accuracy 59,93%, balanced accuracy 50%, macro-F1 khoảng 37,47%.

| Phương pháp | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---:|---:|---:|
| CLIP ảnh | 63,87 | 63,03 | 63,37 |
| CLIP tiêu đề | 57,50 | 56,66 | 56,99 |
| CLIP ảnh + tiêu đề | 64,07 | 63,27 | 63,65 |
| StatA ảnh | **72,02 ± 0,27** | **71,29 ± 0,32** | **71,69 ± 0,38** |
| CLIP + StatA ảnh | 71,26 ± 0,32 | 70,54 ± 0,35 | 70,96 ± 0,38 |
| CLIP + StatA ảnh + tiêu đề | 71,17 ± 0,39 | 70,46 ± 0,42 | 70,89 ± 0,46 |

Ba baseline tất định chạy một lượt; các hàng StatA báo trung bình ± sample std trên ba seed, không phải khoảng tin cậy tổng quát hóa.

StatA ảnh riêng tiếp tục tốt nhất về accuracy/macro-F1/balanced accuracy trong các phương pháp đã chạy. Accuracy cao hơn CLIP ảnh 8,16 điểm phần trăm. Fusion đa phương thức hơn baseline ảnh–tiêu đề 7,10 điểm phần trăm nhưng thấp hơn StatA riêng 0,86 điểm phần trăm. Cả ba seed đều cho thứ tự này. Thêm tiêu đề vào CLIP + StatA làm giảm nhẹ accuracy ở cả ba seed; chưa chứng minh lợi ích top-1 từ văn bản trong cách fusion hiện tại. Tuy nhiên NLL của fusion đa phương thức khoảng 0,602–0,607, thấp hơn StatA khoảng 0,992–1,023, nên không kết luận StatA tốt hơn về mọi tiêu chí confidence.

Không quy mức tăng StatA từ 65,88% (cohort cũ) lên 72,02% hoàn toàn cho tiêu đề ngắn: hai cohort chỉ chung 2.098/3.000 ID; 902 ID mỗi phía khác nhau và phân bố lớp cũng thay đổi. Đây là thay đổi dữ liệu, không phải cải tiến thuật toán, và cả hai lượt đều dùng dev.

Bước tiếp theo có thể kiểm tra riêng `0,75 StatA + 0,25 tiêu đề` từ posterior đã lưu, giữ ba seed, không cần GPU. Phép này chưa có trong sáu phương pháp hiện tại; nó giúp tách ảnh hưởng văn bản khỏi việc pha CLIP vào StatA. Chưa chọn test winner hoặc chuyển Task 2.
