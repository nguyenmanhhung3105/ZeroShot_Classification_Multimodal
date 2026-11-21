# Kiểm tra lượt Task 1 đủ 3.000 mẫu dev

Nguồn: `results/final/kaggle/fakeddit_task1_dev_192c58fde5ad/run/`. Run hoàn tất, MetaCLIP2 ViT-H/14 worldwide FP32 trên CUDA, một tầng, 3.000 ID duy nhất; 1.463 fake và 1.537 real. Cấu hình đúng kế hoạch: seed chia nhóm 0/1/2, nhóm 75 ảnh, CLIP–StatA 1/2–1/2, ảnh–tiêu đề 3/4–1/4. Thời gian mã hóa khoảng 666 giây; mã hóa và suy luận khoảng 669,5 giây (không bao gồm khởi tạo model và toàn bộ thời gian notebook).

Đã xác minh checksum toàn bộ artifact; tính lại accuracy, macro-F1 và balanced accuracy cho 12 posterior từ `replay_scores.npz`; kiểm tra trung bình, sample std ddof=1 và mọi số đếm sửa sai/làm sai trong `comparisons.tsv`. Các phép kiểm tra đều khớp. Không chạy lại inference hoặc gửi job mới.

| Phương pháp | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| CLIP ảnh | 62,47 | 62,46 |
| CLIP tiêu đề | 53,77 | 53,75 |
| CLIP ảnh + tiêu đề | 62,57 | 62,56 |
| StatA ảnh | **65,88 ± 0,68** | **65,87 ± 0,69** |
| CLIP + StatA ảnh | 65,46 ± 0,54 | 65,45 ± 0,54 |
| CLIP + StatA ảnh + tiêu đề | 65,43 ± 0,49 | 65,43 ± 0,50 |

Ba baseline tất định tính một lượt; ba phương pháp StatA báo cáo trung bình ± sample std qua ba seed chia nhóm, không chọn seed tốt nhất. Độ lệch chuẩn này không phải khoảng tin cậy tổng quát hóa.

## Kết luận

- Ứng viên đa phương thức tăng accuracy trung bình 2,87 điểm phần trăm so baseline ảnh–tiêu đề; cả ba seed đều tăng, lần lượt 2,53 / 3,43 / 2,63 điểm phần trăm.
- Tuy nhiên, StatA ảnh riêng tốt nhất trong lượt này: accuracy 65,33 / 66,63 / 65,67% theo seed 0/1/2, đều cao hơn ứng viên đa phương thức 65,10 / 66,00 / 65,20%. Trung bình chênh 0,44 điểm phần trăm. So trực tiếp với StatA, fusion sửa 37/26/30 ảnh nhưng làm sai 44/45/44 ảnh tương ứng.
- Bằng chứng cải thiện hiện đến từ StatA trên ảnh. Chưa chứng minh thêm CLIP hoặc tiêu đề cải thiện accuracy so với StatA riêng. Tiêu đề riêng đạt 53,77%, thêm tiêu đề vào CLIP ảnh chỉ tăng 0,10 điểm phần trăm (3/3.000 ảnh đúng). Kết luận này chỉ áp dụng cho prompt/encoder/cách fusion đang thử.
- Fusion đa phương thức có NLL khoảng 0,670–0,676, thấp hơn StatA riêng khoảng 1,190–1,217. Như vậy StatA tốt hơn về top-1 nhưng không tốt hơn trên mọi tiêu chí; posterior quá tự tin cần được phân tích nếu dùng làm confidence. Chưa fit calibration.

Giữ StatA ảnh làm mốc mạnh cho bước nghiên cứu tiếp. Ưu tiên kiểm tra biểu diễn tiêu đề, lỗi theo lớp và đóng góp văn bản trước khi thêm nhánh mới. Chưa chuyển sang Task 2 hoặc đánh giá test. Đây vẫn là dev, bao gồm mẫu pilot đã dùng phát triển; chưa có xác nhận độc lập trên test. Không thay cấu hình chính hoặc tạo test lock trong lần kiểm tra này.
