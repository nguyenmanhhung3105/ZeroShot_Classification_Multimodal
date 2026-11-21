# Kiểm tra full DTD: Waffle–semantic–StatA

Đã kiểm tra hai run:

- `results/final/kaggle/waffle_stata_dtd_full_c86ed2a97772/run/`
- `results/final/kaggle/waffle_semantic_stata_dtd_08236ee40262/run/`

Cả hai hoàn tất đủ 1.880 ảnh / 47 lớp DTD test partition 1, cùng MetaCLIP2, cohort ID/nhãn/hash ảnh/thứ tự khớp. Checksum toàn bộ artifact hợp lệ. Đã tính lại bốn metrics cho 39 và 71 posterior tương ứng; trung bình/sample std và các số đếm ghép cặp khớp. Công thức prototype semantic 1/2–1/2 được tính lại đúng. 39 posterior chung giữa hai run khớp trong dung sai atol=1e-6, rtol=1e-5. Không chạy inference hoặc gửi job mới.

| Phương pháp | Accuracy (%) | Macro-F1 (%) | Top-5 (%) |
|---|---:|---:|---:|
| CLIP | 66,76 | 64,39 | 90,59 |
| CLIP–StatA | 66,95 ± 0,08 | 64,84 ± 0,22 | 88,74 ± 0,27 |
| Semantic unit (DCLIP unit) | 66,01 | 63,59 | 92,55 |
| Semantic–StatA | 66,90 ± 0,13 | 65,15 ± 0,11 | 90,23 ± 0,20 |
| Waffle | 68,94 ± 0,23 | 66,95 ± 0,25 | 93,53 ± 0,25 |
| Waffle unit | 69,59 ± 0,16 | 67,71 ± 0,19 | 93,66 ± 0,12 |
| Waffle–StatA | 69,25 ± 0,33 | 67,78 ± 0,38 | 91,59 ± 0,24 |
| Waffle–semantic | 70,40 ± 0,23 | 68,65 ± 0,26 | **94,95 ± 0,04** |
| Waffle–semantic–StatA | **71,10 ± 0,24** | **69,95 ± 0,23** | 93,11 ± 0,32 |

Các nhánh Waffle không StatA dùng 7 prompt seed; kết hợp StatA dùng 7 prompt × 3 grouping seed, các nhánh semantic/CLIP–StatA dùng 3 grouping seed. Sample std là biến thiên trên cùng cohort, không phải khoảng tin cậy tổng quát hóa hay 21 thử nghiệm độc lập. DTD có support đều nên balanced accuracy bằng accuracy.

## Kết luận trên DTD

Ứng viên Waffle–semantic–StatA đạt accuracy cao nhất trong các phương pháp đã thử trên full DTD, hơn CLIP 4,34 điểm phần trăm, Waffle unit 1,51, Waffle–StatA 1,85 và Waffle–semantic 0,69. Macro-F1 cũng cao nhất.

Trong 21 đối chiếu cùng prompt/grouping seed, ứng viên mới hơn Waffle–semantic 21/21, Waffle–StatA 21/21 và semantic–StatA 21/21 về accuracy. Đây là bằng chứng bổ trợ có kiểm soát trên DTD, chưa phải kết luận có ý nghĩa thống kê độc lập hoặc vượt paper dùng backbone/giao thức khác.

Tách thành phần: thêm semantic vào unit Waffle tăng khoảng 0,81 điểm phần trăm; thêm StatA vào prototype kết hợp tăng tiếp khoảng 0,69. Trong khi đó StatA áp dụng riêng lên Waffle làm giảm accuracy so unit Waffle khoảng 0,34 điểm phần trăm trên full split, khác tín hiệu pilot 256 ảnh. Không suy rộng kết luận từ pilot.

Top-5 của Waffle–semantic–StatA thấp hơn Waffle–semantic (93,11% so 94,95%), nên không tuyên bố phương pháp mới tốt hơn về mọi tiêu chí.

## Bước tiếp theo

Giữ nguyên mô tả, công thức semantic weight 0,5 và seed; kiểm chứng trên dataset khác, ưu tiên EuroSAT vì StatA trước đó làm giảm điểm. Cần chạy cả Waffle–semantic có/không StatA và các baseline trên cùng cohort để tách hiệu ứng. Chưa chốt phương pháp chung, chưa dùng test Fakeddit, chưa chuyển Task 2. DTD đã phục vụ phát triển nên không gọi lượt này là xác nhận độc lập.
