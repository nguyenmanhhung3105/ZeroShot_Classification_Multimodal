# Smoke loader cho bốn benchmark mới

Hai Input đã chuẩn bị:

- `nguyenhung3105/vlm-cifar100-sources-v1` (API báo ready).
- `nguyenhung3105/vlm-shared-benchmarks-v1` (ba bộ chung đã xác thực nguồn).

Loader riêng `src/vlm_project/shared_data.py` đọc cấu trúc Kaggle đã giải nén,
kể cả thư mục lồng hai cấp của CIFAR/Aircraft. Cars đọc parquet theo batch nhỏ.
Flowers hỗ trợ cả ảnh đã giải nén và archive `102flowers.tgz` được Kaggle giữ
nguyên. Với archive, duyệt luồng tar và chỉ giải mã 5 ảnh validation đã chọn,
không ghi file nguồn ra Input/Output hay đổi ID để né ảnh thiếu.
Không thay đổi loader/CLI benchmark chính hoặc các job Fakeddit hiện tại.

## Gửi smoke bằng CLI

Từ thư mục dự án, tạo staging rồi gửi **một lần**:

```sh
TASK_STAGE=$(mktemp -d)
cp notebooks/smoke_shared_datasets.ipynb "$TASK_STAGE/"
cp configs/kaggle_shared_data_smoke.json "$TASK_STAGE/kernel-metadata.json"
venv/bin/kaggle kernels push -p "$TASK_STAGE"
```

Notebook độc lập chứa loader; không cần dataset code hoặc Internet. Job CPU,
không model, không train/inference. Chỉ decode tối đa 5 ảnh mỗi bộ từ train/val,
không mở ảnh test. MAT Flowers chứa toàn bộ metadata nhãn/split nhỏ, không chứa ảnh;
chỉ chọn 5 ID validation trước khi mở ảnh. Không quét/hash toàn bộ dataset.

Theo dõi mỗi hai phút:

```sh
while true; do
  date
  venv/bin/kaggle kernels status nguyenhung3105/vlm-shared-data-smoke-v1
  sleep 120
done
```

Sau khi complete, lấy Output nhỏ (20 PNG và summary; không tải dataset nguồn):

```sh
venv/bin/kaggle kernels output nguyenhung3105/vlm-shared-data-smoke-v1 \
  -p results/final/shared_data_smoke
```

Chỉ đạt khi có bốn log `SMOKE OK` và `shared_data_smoke/summary.json` ghi
`status=data_smoke_passed`, mỗi bộ đúng 5 ảnh. Mở các ảnh PNG để kiểm tra hình.

## Kiểm tra và giới hạn

- CIFAR: record binary planar RGB đúng, nhãn coarse/fine khớp mapping nguồn.
- Flowers: dùng ID validation, đổi nhãn 1..102 về 0..101; chưa có tên lớp được
  xác thực nên `class_name=null`. Chưa thể dùng tên lớp để tạo prompt inference.
- Cars: giữ thứ tự 196 lớp của mirror, chỉ decode 5 ảnh trong batch đầu.
- Aircraft: giữ label variant và ảnh thô; chưa crop banner 20 pixel ở bước smoke.
- Output chỉ kiểm chứng đọc dữ liệu và nhãn; không có dự đoán/accuracy và không
  chứng minh lợi ích hai tầng. Cần giao thức, tên lớp và baseline trước khi inference.
- CIFAR/Aircraft cần cấu trúc đã giải nén trên Kaggle; Flowers hỗ trợ thêm `.tgz`.
  Smoke không tự giải nén tất cả dataset.
- Cần Pillow, numpy, scipy, pyarrow; các job chuẩn bị trước đã dùng các thư viện
  này trên Kaggle. Local chưa cài pyarrow nên test parquet tích hợp được skip.

Kiểm tra local bằng `venv/bin/python -m unittest discover -s tests -p test_shared_data.py -v`:
test ảnh tổng hợp xác thực planar RGB, ánh xạ coarse/fine, ID/split Flowers,
nhãn Aircraft có khoảng trắng, nguồn trùng/thiếu, chặn thư mục cấm và limit >5.
Test Flowers `.tgz` chứa ảnh ngoài subset bị hỏng để kiểm tra không decode chúng;
kiểm tra đủ 5 ID được chọn, không giải nén JPEG ra đĩa và báo lỗi nếu thiếu ID.
Chưa xác nhận smoke dữ liệu thật cho đến khi chạy job Kaggle thành công.

Ngày 10/11/2026, đã kiểm tra danh sách file Dataset qua API: Flowers có
`102flowers.tgz` (344.862.509 byte) cùng hai MAT, không có JPEG đã giải nén.
Cars vẫn có đủ bốn shard parquet train/test. Lỗi `found 0` trong smoke đầu tiên
do loader chỉ tìm JPEG trực tiếp; đã cập nhật notebook độc lập theo loader sửa.
Gửi lại cùng lệnh push để tạo version mới của smoke, không chạy lại job tải nguồn.
