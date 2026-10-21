# Báo cáo giữa kỳ: kết quả cohort chính 3.000 mẫu

- Tệp chính: `bao_cao_giua_ky.tex`.
- Hình cần để biên dịch: toàn bộ thư mục `images/` đi kèm trong gói Overleaf.
- Gói upload: `bao_cao_giua_ky_overleaf.zip`; chọn compiler **pdfLaTeX**.
- Không có PDF báo cáo trong bản bàn giao. Sơ đồ hệ thống và biểu đồ kết quả
  được vẽ trực tiếp bằng TikZ/PGFPlots trong tệp `.tex`.

Tại máy local có thể biên dịch từ thư mục `docs/` hoặc từ thư mục gốc dự án.
Chạy pdfLaTeX ba lần để cập nhật mục lục, danh mục và trích dẫn. Ảnh lý thuyết
được giữ nguyên nội dung từ repo chính thức của ViT, CLIP và DCLIP; nguồn,
checksum và ghi chú được lưu tại `images/theory/sources.json`. Logo bìa lấy
từ website chính thức của Đại học Tôn Đức Thắng.

## Phạm vi số liệu

Báo cáo chỉ dùng 28 lượt có biên nhận của cohort chính: 20 baseline và tám
DCLIP adapted, mỗi lượt đủ 3.000 dự đoán. Hai lượt CrisisMMD cũ dù cũng có
3.000 mẫu không được đưa vào các bảng. Không xóa dữ liệu hoặc kết quả lịch sử.

Mỗi task dùng cùng ID, nhãn, hash văn bản và thông tin ảnh giữa các cấu hình.
Các chỉ số đã được tính lại từ 84.000 dòng dự đoán, đối chiếu summary và
kiểm tra quy tắc quyết định nhãn. Bộ mẫu có nguồn validation Fakeddit,
train chung hai task CrisisMMD và test MM-IMDb; báo cáo không gọi cả ba
là test chính thức độc lập.

Biên bản đầy đủ: `experiments/ket_qua_3000_kiem_tra.json`.
Bảng số liệu: `experiments/ket_qua_3000.tsv`.
Có thể tái kiểm tra từ thư mục gốc dự án bằng:

```sh
venv/bin/python -B scripts/analyze_report_3000.py
```

Lệnh này chỉ đọc kết quả đã lưu, không đọc dữ liệu nguồn và không nạp mô hình.
