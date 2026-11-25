# Giao diện báo cáo thực nghiệm

## Cấu trúc

Tên hiển thị: **VLM Classifier**. Mô tả: **Phân loại nội dung ảnh–tiêu đề**.
Biểu tượng SVG gồm hai đầu vào ảnh/văn bản nối với một khối quyết định. Không dùng
khẩu hiệu hoặc số liệu hiệu năng trong phần giới thiệu.

Điều hướng có năm vùng độc lập; tại một thời điểm chỉ một vùng được hiển thị:

| Vùng | Nội dung |
|---|---|
| Bài toán | Đầu vào, luồng xử lý, hệ nhãn Fakeddit, phân loại phẳng/phân cấp |
| Cấu hình suy luận | Upload, tiêu đề, bài toán, pipeline, backbone, nhóm phương pháp |
| Kết quả suy luận | Đầu vào đang đối chiếu, trạng thái, thời gian, bảng dự đoán, chi tiết |
| Kết quả thực nghiệm | Biểu đồ từ artifact đã kiểm tra, tách khỏi dự đoán một ảnh |
| Lịch sử | Yêu cầu của phiên server và liên kết mở lại kết quả |

Các vùng giữ trạng thái khi chuyển; polling không tự chuyển vùng hoặc đặt lại form.
Submit thành công và thao tác mở lịch sử mới chuyển sang kết quả. Hash URL hỗ trợ
Back/Forward. Tab bàn phím, nhãn điều khiển, tiêu đề dialog và trạng thái focus được khai báo.

## Nhóm phương pháp

Biểu đồ và phạm vi thống kê: [DEMO_EXPERIMENT_CHARTS.md](DEMO_EXPERIMENT_CHARTS.md).

Zero-shot, thích nghi không nhãn, few-shot và đối chứng bổ sung được trình bày riêng.
Legacy, Waffle unit và CALIP nằm trong vùng đối chứng thu gọn. Không đổi công thức,
seed, head, prompt hay mã phương pháp trong request gửi worker.

Kết quả mặc định hiển thị mọi nhóm cùng một bảng, có dòng phân cách theo nhóm.
Bộ lọc chỉ thay phạm vi nhìn, không gửi lại yêu cầu hoặc đổi kết quả. Chi tiết policy/seed
ở dialog riêng. Không dùng nhãn “tốt nhất” hoặc xếp hạng từ điểm của một ảnh.

Thời gian worker, mã hóa đầu vào và chi phí nhánh được phân biệt. Chi phí dưới 0,01 giây
hiển thị `< 0,01 s` thay vì làm tròn thành 0,00 s. Nhánh chưa thực thi không hiển thị
nhãn dự đoán. Lỗi tải lịch sử không được báo thành mất kết nối API local.

## Nguồn tham khảo thiết kế

- [Carbon — Tabs](https://carbondesignsystem.com/components/tabs/usage/): tách các vùng
  nội dung theo ngữ cảnh; không đặt các kết quả cần so sánh trực tiếp vào các tab riêng.
- [Carbon — Data table](https://carbondesignsystem.com/components/data-table/usage/):
  toolbar, bộ lọc, tiêu đề cột và chi tiết bổ sung.
- [NN/g — Progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/):
  đưa các đối chứng và chi tiết ít dùng vào vùng mở rộng.

Chỉ tham khảo nguyên tắc tổ chức; không thêm CDN, font ngoài hoặc dependency frontend.

## Kiểm tra

- RED: kiểm thử cấu trúc và kiểm thử Chrome thất bại vì thiếu điều hướng/vùng nội dung;
  bản cũ còn tiêu đề khẩu hiệu. Các test chạy trước khi thay mã giao diện.
- GREEN: 20 test `test_demo*.py` PASS, gồm hai test cấu trúc mới và API đã cập nhật tên.
- `node --check src/vlm_project/demo_web/app.js`, `git diff --check`: PASS.
- Docker build/up thành công; web healthy, HTTP 200, kiểm tra HTML thực tế xác nhận
  bốn vùng báo cáo và loại bỏ khẩu hiệu cũ.
- Chrome regression đã bổ sung kiểm tra giữ trạng thái form, nhóm phương pháp/kết quả,
  bộ lọc, dialog, download, lịch sử, hủy yêu cầu, trạng thái API và ba viewport.
  Chưa ghi nhận kết quả cuối lượt Chrome tại thời điểm lập tài liệu; không tính là PASS.
- Dùng dữ liệu tổng hợp, không mở dataset/test hoặc chạy inference GPU trong lượt sửa UI.
- Không commit vì chưa được yêu cầu; giữ nguyên các thay đổi demo đã có trước lượt này.

Tự đánh giá (accuracy/completeness/clarity/actionability/conciseness): 4/4/4/4/4,
trung bình 4,0/5. Nội dung đối chiếu với mã chạy và schema hiện tại; giới hạn còn lại
là kiểm chứng đầy đủ các thao tác và bố cục qua Chrome trước khi kết luận QA hoàn tất.

## Phạm vi đầu vào chính

Giao diện chọn phương pháp và biểu đồ chính chỉ trình bày chỉ ảnh hoặc ảnh + tiêu đề. Các cấu hình chỉ tiêu đề vẫn được giữ trong nghiên cứu và snapshot gốc, không hiển thị ở danh sách lựa chọn chính. Few-shot nhị phân chọn sẵn cả ảnh và ảnh + tiêu đề để đối chiếu. Không suy rộng kết quả này thành đối chứng few-shot chỉ ảnh ở sáu lớp/cascade: kết quả hiện có cho phần đó dùng ảnh + tiêu đề.

Danh sách demo chính giữ CLIP ảnh/fusion, DCLIP, WaffleCLIP, StatA và few-shot phù hợp bài toán. Majority, EM-Dirichlet, CLIP–StatA và các biến thể tương ứng nằm trong mục đối chứng bổ sung thu gọn.
