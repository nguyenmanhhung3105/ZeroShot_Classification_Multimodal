# Quy tắc làm việc

## Đọc dự án
- Được đọc đầy đủ mã nguồn, README, cấu hình và khai báo thư viện.
- Được đọc đầy đủ code xử lý dữ liệu.
- Không đọc, tìm kiếm bên trong hoặc chạy chương trình truy cập dataraw/.
- Với data_process/: chỉ xem schema/tên cột và tối đa 5 bản ghi
  đầu mỗi file cần thiết.
- Đọc dữ liệu có giới hạn ngay từ đầu; không nạp toàn bộ file
  rồi mới lấy head().
- Bỏ qua môi trường ảo, thư viện cài sẵn, cache, checkpoint
  và trọng số mô hình.
- Không đọc file chứa mật khẩu, token hoặc khóa bí mật.

## Sửa code
- Hiểu luồng xử lý và các nơi gọi hàm trước khi sửa.
- Ưu tiên sửa nhỏ, đúng nguyên nhân; tránh viết lại cả dự án.
- Giữ nguyên nghiệp vụ, định dạng đầu vào/đầu ra và cấu hình
  thí nghiệm, trừ khi nhiệm vụ yêu cầu thay đổi.
- Không ghi đè thay đổi sẵn có của người dùng.
- Chạy kiểm tra phù hợp với phần sửa; không chạy toàn bộ
  dataset, train hoặc inference lớn nếu chưa được yêu cầu.
- Báo cáo đã sửa gì, tại sao, đã kiểm tra gì và còn hạn chế gì.

## Git
- Kiểm tra git status và diff trước khi thao tác.
- Chỉ commit, push hoặc deploy khi người dùng yêu cầu.
- Không force push hoặc xóa thay đổi để giải quyết xung đột.
- Trả lời bằng tiếng Việt.