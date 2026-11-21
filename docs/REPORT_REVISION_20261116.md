# Biên tập và bố trí báo cáo ngày 16/11/2026

## Bộ tài liệu bàn giao

Điểm vào là `report/main.tex`. Nội dung được tách thành `frontmatter.tex`,
sáu tệp trong `chapters/`, `references.tex` và `assets/`. Bộ bàn giao chỉ gồm
LaTeX và hình PNG; không kèm PDF theo yêu cầu của người dùng.

Nguồn trước khi biên tập và bản PDF đã có từ trước được sao lưu trong
`backups/report_revision_20261116/`, cùng manifest SHA256. Không thay thế
hoặc xóa các snapshot lịch sử khác.

## Bố trí source đã rà soát

| Nhóm trách nhiệm | Module/điểm vào hiện tại |
|---|---|
| CLI và I/O | `run.py`, `cli.py`, `io.py` |
| Bộ mã hóa và prompt | `backbone.py`, `methods.py`, `paper.py` |
| Đọc và nhận dạng cohort | `data.py`, `shared_data.py` |
| Đánh giá và thích nghi | `evaluate.py`, `adaptation.py`, `diagnostic.py` |
| Đối chứng prototype | `waffle_stata.py`, `waffle_semantic_stata.py` |
| Few-shot binary/support | `fakeddit_fewshot_task1.py`, `fakeddit_fewshot_support.py` |
| Few-shot cascade | `fakeddit_fewshot_cascade.py` |
| Protocol và runner | `min5_protocol.py`, `cascade_protocol.py`, các test runner tương ứng |
| Đóng gói và thực thi Kaggle | `packaging.py`, `kaggle.py`, các entry point trong `scripts/` |
| Tiền xử lý | `src/preprocessing/data_fakeddit/` |

Graph AST hiện có hỗ trợ định vị backbone và few-shot support; cascade và một số
protocol mới hơn graph nên được đọc trực tiếp. Không chuyển vị trí module Python:
giữ luồng import, cấu hình và sự đối chiếu source với artifact lịch sử. Thư mục
báo cáo tách riêng phần trình bày khỏi mã thực nghiệm.

## Nội dung biên tập

- Chương 1 chuyển từ lịch sử giữa kỳ sang bối cảnh, mục tiêu, nghiên cứu liên quan
  và đóng góp thực nghiệm.
- Chương 2 thống nhất thuật ngữ và cách diễn giải các chỉ số.
- Chương 3 mô tả đặc trưng, support L/B, quy tắc hard gate, R87 và phân rã lỗi.
- Chương 4 xác định dữ liệu, nguồn giám sát, ma trận đối chứng và chỉ số chính.
- Chương 5 giữ kết quả đo đã có, thêm đủ sáu nhánh cascade, ghép cặp, gate,
  subtype, oracle chẩn đoán và audit dev. Hai biểu đồ được tạo từ TSV đã lưu.
- Chương 6 kết luận theo phạm vi và chỉ số đã đo; bỏ các đoạn bỏ ngỏ và kế hoạch
  chưa có kết quả khỏi nội dung báo cáo.

Các số min5 được gọi đúng là kết quả dev. Macro-F1 cao nhất của sáu nhánh thuộc
phẳng B; Accuracy cao nhất thuộc cascade B mặc định. Không đổi kết luận thành
khẳng định một nhánh thắng theo mọi chỉ số.

## Phần đầu theo mẫu người dùng

Đối chiếu trực tiếp XML của `/Users/hung/Documents/MyProject/template_report.docx`:
hai bìa, lời cảm ơn, cam đoan, tên đề tài và tóm tắt tiếng Việt,
mục lục, danh mục hình/bảng/chữ viết tắt. Trang Abstract tiếng Anh được bỏ theo yêu cầu cập nhật của người dùng. Tên đề tài, sinh viên,
giảng viên và tên học phần được giữ theo dự án thực tế.

Cỡ chữ bìa khai báo tường minh: đề tài 24 pt; dòng báo cáo 22 pt;
Tổng Liên đoàn và trường 14 pt. Logo lấy từ mẫu Word. Phần đầu dùng lề trái/trên
3,5 cm, phải 2 cm, dưới 3 cm theo mẫu; nội dung chương dùng lề của báo cáo.

## Kiểm tra

- Chạy XeLaTeX ba lượt với `-no-pdf -no-shell-escape`: chỉ sinh XDV/tệp trung
  gian tại `/private/tmp/vlm-report-texcheck/`, không xuất PDF.
- Tham chiếu và ký tự hợp lệ; không có overfull box. Một số underfull box ở
  cột bảng hẹp và URL là cảnh báo khoảng trắng, không phải tràn dòng.
- Tính lại metric cascade từ 90.000 prediction: 30 tổ hợp, 3.000 ID, 18 artifact
  được xác minh checksum; trạng thái `passed`.
- 14 mục tham khảo đều là tài liệu nghiên cứu trên web. Sửa tên tác giả StatA
  thành Clément Fuchs; đối chiếu survey phân cấp với nhà xuất bản để lấy đúng
  trang 31–72 và DOI, thay metadata sai ở trang tổng hợp của trường.
- Kiểm tra tĩnh toàn bộ LaTeX: tệp input/hình tồn tại; nhãn/citation đầy đủ;
  không dẫn source/Markdown local trong tài liệu tham khảo; không chứa đoạn
  kết quả bỏ ngỏ; không còn PDF trong `report/`.

## Tự đánh giá theo agent-self-evaluation

| Tiêu chí | Điểm | Bằng chứng và điểm có thể cải thiện |
|---|---:|---|
| Chính xác | 4/5 | Đối chiếu source, tính lại metric cascade, kiểm checksum; kết quả lịch sử kế thừa audit đã lưu. |
| Đầy đủ | 4/5 | Có thư mục riêng, sáu chương, tham khảo web, phần đầu theo mẫu và cỡ chữ; bố cục được kiểm bằng XDV theo yêu cầu không xuất PDF. |
| Rõ ràng | 4/5 | Tách nhị phân/sáu lớp, dev/cohort cũ, phương pháp/oracle và giải thích L/B/R87. |
| Khả dụng | 4/5 | `main.tex` nhập đầy đủ nội dung/hình; có hướng dẫn giữ cấu trúc thư mục và tái tạo biểu đồ. |
| Súc tích | 4/5 | Bỏ lịch sử thao tác và kế hoạch khỏi báo cáo; giữ bảng đối chứng cần cho lập luận. |

Trung bình 4,0/5. Khi có chỉnh sửa nội dung hoặc yêu cầu định dạng mới, kiểm lại
LaTeX ở chế độ không xuất PDF và cập nhật chỉ những bảng bị ảnh hưởng.
