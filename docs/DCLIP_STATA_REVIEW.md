# Kiểm tra DCLIP–StatA trên đủ DTD

Run `results/final/kaggle/dclip_stata_dtd_993bc656c03d/run/` hoàn tất **1.880 ảnh / 47 lớp**, MetaCLIP2, ba grouping seed 0/1/2. Đã xác minh checksum artifact, tính lại metrics từ chín posterior, kiểm tra trung bình/sample std và ba hàng đối chiếu sửa sai/làm sai; đều khớp. Cohort, thứ tự ID, nhãn và hash ảnh khớp lượt full benchmark DTD cũ. Không gửi job hoặc chạy lại model.

| Phương pháp | Accuracy (%) | Macro-F1 (%) | Top-5 (%) |
|---|---:|---:|---:|
| CLIP | 66,76 | 64,39 | 90,59 |
| CLIP–StatA | 66,95 ± 0,08 | 64,84 ± 0,22 | 88,74 ± 0,27 |
| DCLIP | 64,79 | 62,08 | 92,77 |
| DCLIP unit | 66,01 | 63,59 | 92,55 |
| DCLIP–StatA | 66,90 ± 0,13 | 65,15 ± 0,11 | 90,23 ± 0,20 |

DCLIP–StatA tăng accuracy 2,11 điểm phần trăm so DCLIP gốc, trong đó chuẩn hóa prototype riêng đã tăng 1,22 điểm phần trăm. Phần thích nghi StatA tăng thêm 0,89 điểm phần trăm so DCLIP unit, dương ở cả ba seed (1,01 / 0,90 / 0,74 điểm phần trăm).

Tuy nhiên, DCLIP–StatA chỉ hơn CLIP 0,14 điểm phần trăm và thấp hơn CLIP–StatA 0,05 điểm phần trăm trung bình (khoảng một ảnh đúng mỗi lượt). So trực tiếp CLIP–StatA cùng grouping seed: thêm 1 ảnh đúng ở seed 0, mất 2 ảnh ở seed 1 và 2. Macro-F1 của DCLIP–StatA tốt hơn CLIP–StatA khoảng 0,31 điểm phần trăm, nên không kết luận phương pháp nào tốt hơn trên mọi metric. Top-5 của StatA giảm so DCLIP/unit.

Mốc Waffle trên cùng full cohort DTD từ lượt benchmark cũ: Waffle 68,94 ± 0,23%, Waffle unit 69,59 ± 0,16% (bảy prompt seed). DCLIP–StatA vẫn thấp hơn lần lượt 2,04 và 2,69 điểm phần trăm. CLIP và DCLIP không thích nghi tái hiện đúng accuracy của lượt cũ; không so điểm full DCLIP–StatA với Waffle–StatA pilot 256 ảnh để xếp hạng.

Kết luận: descriptor DCLIP có lợi khi thêm chuẩn hóa và StatA so chính DCLIP gốc, nhưng chưa tạo ứng viên vượt các baseline mạnh về accuracy. Chưa chọn DCLIP–StatA làm phương pháp chung. Waffle unit hiện vẫn là mốc full DTD tốt nhất đã có. Nếu tiếp tục StatA, bước xác nhận phù hợp là Waffle–StatA đủ 1.880 ảnh, giữ các đối chứng và seed cố định trước khi thêm semantic. Đây vẫn là split đã phục vụ phát triển, không phải xác nhận độc lập.
