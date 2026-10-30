# Hướng few-shot CLIP để quay lại sau

## Trạng thái

Người dùng yêu cầu lưu hướng few-shot để cân nhắc sau. Hiện chưa triển khai,
chưa tạo support set và chưa gửi job Kaggle cho hướng này. Việc lưu ghi chú
không thay đổi thiết lập zero-shot của các thí nghiệm hiện tại.

## Phương pháp ưu tiên: Tip-Adapter

Nguồn chính thức: [Tip-Adapter, ECCV 2022](https://github.com/gaopengcuhk/Tip-Adapter).

Tip-Adapter dùng một tập nhỏ ảnh có nhãn để tạo bộ nhớ gồm embedding ảnh và
nhãn tương ứng. Khi phân loại ảnh mới, kết hợp điểm CLIP từ prompt với điểm
truy hồi bộ nhớ. Bản Tip-Adapter gốc không cần tối ưu trọng số bằng gradient;
Tip-Adapter-F có bước tinh chỉnh bộ nhớ. Dù không huấn luyện backbone, sử dụng
support có nhãn vẫn thuộc **few-shot**, không phải zero-shot.

## Giả thuyết cho dự án

Một số ví dụ có nhãn đúng miền CrisisMMD có thể bổ sung thông tin mà mô tả
lớp khó thể hiện. Đây là giả thuyết cần đánh giá, chưa phải kết quả cải thiện.
Có thể giữ backbone MetaCLIP2 và so với phương pháp hai tầng hiện có để
giảm thay đổi đồng thời nhiều yếu tố. Cần kiểm tra tương thích trước khi chạy.

## Khi quay lại triển khai

1. Thống nhất việc mở rộng phạm vi sang few-shot với người dùng; ghi rõ trong báo cáo.
2. Lấy support từ train, chọn trước số mẫu mỗi lớp theo khả năng dữ liệu;
   không lấy nhãn dev/test để xây bộ nhớ.
3. Kiểm tra tách biệt support và tập đánh giá theo ID, tweet và hash ảnh;
   không mặc định các split đã hoàn toàn độc lập.
4. Giữ cùng backbone và cùng tập đánh giá khi so với zero-shot. Nếu thử nhiều
   cách chọn support, báo cáo trung bình và độ biến thiên, không chọn seed tốt nhất.
5. Chọn tham số trên dev, khóa phương án trước khi đánh giá test.

## Phân biệt với transductive zero-shot

Few-shot dùng ví dụ **có nhãn** làm support. Transductive zero-shot tận dụng
một nhóm ảnh cần dự đoán **không có nhãn** để suy luận chung. Hai khái niệm
khác nhau; một phương pháp cũng có thể vừa few-shot vừa transductive.
