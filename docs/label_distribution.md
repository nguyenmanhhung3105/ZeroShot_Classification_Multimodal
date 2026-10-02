# Xem số mẫu từng lớp, không chạy model

Chạy từ thư mục gốc dự án:

```bash
venv/bin/python src/run/test_label_distribution.py
```

Script đọc tuần tự **toàn bộ file processed** của ba dataset trong YAML.
Không đọc raw, không mở ảnh, không chạy model, không sửa hay cân bằng dữ liệu.
Nó thống kê cả Fakeddit 2way/6way, CrisisMMD informativeness/humanitarian và
23 thể loại MM-IMDb, bất kể task/enabled đang chọn để inference.

Mỗi lần chạy tạo thư mục mới trong `results/label_distribution/`:

- `report.html`: mở bằng trình duyệt, có bảng và thanh tỷ lệ từng lớp.
- `class_counts.tsv`: dataset, task, ID/tên lớp, số mẫu, % số dòng.
- `dataset_overview.tsv`: file nguồn, cột nhãn, tổng số dòng, thiếu/lạ/lặp nhãn.
- `unknown_labels.tsv`: các giá trị nhãn ngoài danh sách hợp lệ để kiểm tra.

Chỉ đọc thử tối đa 5 bản ghi đầu mỗi file:

```bash
venv/bin/python src/run/test_label_distribution.py --max-rows 5
```

Bản xem trước được ghi rõ `first_rows_preview`, không đại diện toàn dataset.
Có thể đổi `--config` hoặc `--output-dir`; đường dẫn trong YAML tính từ thư mục
đang chạy, giống các runner. Hỗ trợ TSV/CSV/Parquet, đọc streaming theo dòng/batch.

## Cách hiểu bảng

Đây là **phân bố file trước lọc**, không áp dụng `max_samples`, độ dài text,
lọc ảnh hỏng hoặc thiếu ảnh của inference. Vì vậy số mẫu có thể khác summary
chạy model. Muốn phân tích đúng cohort đã chạy, cần thống kê nhãn thật trong
raw_predictions.tsv của lượt đó, không lấy bảng toàn file thay thế.

Mẫu là một dòng; script không gộp các dòng cùng ID. Với MM-IMDb, một phim
có nhiều thể loại: mỗi thể loại được đếm một lần/phim, tổng số lượt nhãn
có thể lớn hơn số phim, tổng phần trăm có thể vượt 100%. Mẫu thiếu nhãn vẫn
nằm trong mẫu số tỷ lệ và được báo riêng. Các nhãn hợp lệ trong dòng có
thêm nhãn lạ vẫn được đếm. Nhãn lặp không làm tăng số mẫu của lớp.

Lớp 0 mẫu vẫn xuất hiện. Nếu thiếu file/cột, báo N/A và status=error,
không giả định là 0 mẫu; các dataset khác vẫn được báo cáo, script exit=1.
Đây là công cụ kiểm chứng trước khi quyết định điều chỉnh; không tự bỏ lớp
hiếm, đổi nhãn hay chỉnh tập validation để tăng metric.

Các kiểm tra tự động chỉ dùng dữ liệu giả. Chưa xác minh số mẫu thực tế trên
toàn bộ dữ liệu của người dùng; việc chạy đầy đủ do người dùng thực hiện.
