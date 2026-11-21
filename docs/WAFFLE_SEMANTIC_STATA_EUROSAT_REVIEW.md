# Kiểm tra full EuroSAT: Waffle–semantic–StatA

Run `results/final/kaggle/waffle_semantic_stata_eurosat_664684cd7c96/run/` hoàn tất **27.000 ảnh RGB / 10 lớp** theo giao thức all-images của Waffle upstream, MetaCLIP2, semantic weight 0,5, bảy prompt seed và ba grouping seed. Thời gian mã hóa/suy luận ghi trong run khoảng 4.987 giây (83,1 phút), không phải toàn bộ thời gian notebook/GPU.

Đã kiểm checksum mọi artifact; kiểm tra 27.000 ID duy nhất, shape/xác suất 71 posterior, tính lại accuracy/macro-F1/balanced accuracy/top-5 cho mọi seed và trung bình/sample std. Đã kiểm công thức blended prototype và 42 đối chiếu sửa sai/làm sai. Tất cả khớp. Không chạy lại model hoặc gửi job mới.

| Phương pháp | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---:|---:|---:|
| CLIP | 64,06 | 62,13 | 64,07 |
| CLIP–StatA | 60,69 ± 0,04 | 53,97 ± 0,06 | 60,26 ± 0,05 |
| Semantic unit | 57,20 | 55,21 | 57,71 |
| Semantic–StatA | 57,78 ± 0,06 | 55,68 ± 0,05 | 58,12 ± 0,04 |
| Waffle | 62,66 ± 0,59 | 62,01 ± 0,96 | 62,21 ± 0,57 |
| Waffle unit | 62,95 ± 0,52 | 61,01 ± 0,70 | 63,03 ± 0,63 |
| Waffle–semantic | **64,42 ± 0,41** | **64,08 ± 0,48** | **64,91 ± 0,43** |
| Waffle–StatA | 60,25 ± 0,59 | 53,77 ± 0,50 | 59,81 ± 0,63 |
| Waffle–semantic–StatA | 61,73 ± 0,50 | 57,13 ± 0,64 | 62,07 ± 0,63 |

Std phản ánh seed trên cùng cohort, không phải khoảng tin cậy; 21 tổ hợp có chung prompt/grouping nên không độc lập.

## Kết luận

Thêm StatA vào Waffle–semantic làm giảm accuracy trung bình **2,69 điểm phần trăm**, thua **21/21** đối chiếu ghép cặp cùng prompt seed. Dù Waffle–semantic–StatA hơn Waffle–StatA 1,49 điểm phần trăm và semantic–StatA 3,96, nó vẫn thua phương pháp kết hợp prototype không StatA. Không chọn nhánh ba thành phần làm mặc định chung dựa trên hai dataset hiện có.

Recall trung bình từng lớp cho thấy giảm rõ sau StatA: annual crop land 30,98% → 0,88%; brushland/shrubland 17,19% → 0,89%; pasture land 50,92% → 27,99%. Một số lớp khác tăng (permanent crop land 74,27% → 96,78%, residential 69,68% → 87,56%). Đây là biểu hiện phân phối dự đoán thay đổi và một số lớp mất recall; chưa xác định nguyên nhân chỉ từ metrics, không coi là bằng chứng lỗi ánh xạ nhãn hoặc lỗi code.

## Nhìn chung hai dataset full

| Phương pháp | DTD accuracy (%) | EuroSAT accuracy (%) |
|---|---:|---:|
| CLIP | 66,76 | 64,06 |
| Waffle unit | 69,59 | 62,95 |
| Waffle–semantic | 70,40 | 64,42 |
| Waffle–semantic–StatA | 71,10 | 61,73 |

Waffle–semantic không StatA là ứng viên chuyển giữa hai dataset tốt hơn trong các kết hợp đã thử: hơn Waffle unit 0,81 điểm phần trăm trên DTD và 1,47 trên EuroSAT. So CLIP trên EuroSAT chỉ tăng 0,36 điểm phần trăm; không tuyên bố tăng có ý nghĩa thống kê chỉ từ trung bình seed. Macro-F1 EuroSAT tăng 1,95 điểm phần trăm so CLIP.

Ưu tiên giữ nhánh Waffle–semantic weight 0,5 để kiểm chứng trên dataset tiếp theo cùng giao thức, thay vì mặc định thêm StatA. StatA có thể là lựa chọn riêng cho từng miền (như kết quả Fakeddit dev), chưa chứng minh lợi ích chung. Giữ nguyên các kết quả âm; không chỉnh trọng số theo từng dataset rồi coi là xác nhận độc lập. DTD và pilot EuroSAT đã phục vụ phát triển, chưa có bằng chứng trên benchmark khác ngoài hai bộ này cho phương pháp kết hợp.
