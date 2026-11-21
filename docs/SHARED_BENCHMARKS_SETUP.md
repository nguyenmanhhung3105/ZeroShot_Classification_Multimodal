# Tải ba benchmark dùng chung trên Kaggle

Ba bộ: Flowers102, Stanford Cars, FGVC Aircraft. Notebook riêng, không đổi
notebook CIFAR-100 hay các pipeline thí nghiệm hiện tại.

## Gửi bằng CLI từ thư mục dự án

Kiểm tra trạng thái trước; không push phiên mới khi job cũ còn queued/running:

```sh
venv/bin/kaggle kernels status nguyenhung3105/vlm-prepare-shared-benchmarks-v1
```

Nếu chưa từng tạo job, hoặc chủ động chạy lại sau khi đã xem kết quả phiên trước:

```sh
TASK_STAGE=$(mktemp -d)
cp notebooks/download_shared_benchmarks.ipynb "$TASK_STAGE/"
cp configs/kaggle_shared_benchmarks_kernel.json "$TASK_STAGE/kernel-metadata.json"
venv/bin/kaggle kernels push -p "$TASK_STAGE"
```

Job private, CPU, Internet bật. Không tải dữ liệu xuống Mac; không dùng GPU,
không train/inference. Nguồn tải cỡ 5–6 GB; không giải nén toàn bộ ảnh.
Notebook cần scipy, numpy và pyarrow có sẵn trên Kaggle; nếu thiếu sẽ báo import error.

Theo dõi tại https://www.kaggle.com/code/nguyenhung3105/vlm-prepare-shared-benchmarks-v1
và đọc log. Chỉ hoàn tất sau ba log `Verified` và version có trạng thái complete.

## Nguồn và kiểm tra

| Bộ | Nguồn | Train | Val | Test |
|---|---|---:|---:|---:|
| Flowers102 | Oxford, archive + MAT | 1.020 | 1.020 | 6.149 |
| Stanford Cars | Mirror tanganke/stanford_cars trên Hugging Face | 8.144 | Không có | 8.041 |
| FGVC Aircraft 2013b | Oxford, archive | 3.334 | 3.333 | 3.333 |

Flowers kiểm tra MD5 của ba file, số nhãn, ảnh và split không giao nhau.
Aircraft chấp nhận MD5 Oxford `32eca553f897e747706144f3bb444df0` hoặc bản lưu trữ
datasets-mila `d4acdd33327262359767eeaa97a4f732`, ghi hash thực tế, đủ 10.000 ảnh,
nhãn variant/family/manufacturer, split không giao nhau và ánh xạ cha nhất quán.
Giữ nguyên ảnh nguồn; khi đánh giá phải xử lý banner 20 pixel cuối ảnh Aircraft
theo giao thức đã chốt, không tự thay preprocessing trong bước tải.

Cars khóa revision `9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`, kiểm tra SHA256
của bốn shard train/test, thứ tự 196 lớp và giới hạn nhãn. Không tải các split ảnh
corruption. Mirror chưa được đối chiếu từng ảnh với nguồn Stanford gốc; cần ghi rõ
điều này khi báo cáo. Bộ tải giữ định dạng parquet chứa ảnh, không mã hóa JPEG lại.

## Lưu thành Kaggle Dataset

1. Mở version hoàn tất → Output → Create Dataset.
2. Chọn private, tên `VLM Shared Benchmarks V1`, slug đề xuất `vlm-shared-benchmarks-v1`.
3. Giữ thư mục `shared_benchmarks` với ba thư mục con và manifest.
4. Đợi xử lý xong; ghi ID thực tế trên URL. Không tạo dataset từ output lỗi/tải dở.

Job CLI thí nghiệm sau này thêm ID vào `dataset_sources` (cùng ID dataset code nếu có):

```json
"dataset_sources": ["nguyenhung3105/vlm-shared-benchmarks-v1"]
```

Nguồn được mount dưới `/kaggle/input/`; kiểm tra đường dẫn thực tế. Không tải lại
qua Internet trong từng job thí nghiệm. File đã tải đúng checksum được dùng lại
trong cùng thư mục/session; session mới không tự mang theo `/kaggle/working` cũ.

## Khôi phục job lỗi checksum Aircraft

Notebook đã bổ sung: log số byte/Content-Length/MD5 thực tế, tối đa ba lần thử tải,
dùng lại `.part` nếu checksum đầy đủ hợp lệ, và cell ID để tránh cảnh báo nbformat.
Hai checksum Aircraft có nguồn; không tự chấp nhận checksum lạ hay bỏ xác thực.

Nếu notebook gốc đã lưu Output từ run lỗi, tạo job phục hồi khác ID và gắn Output cũ:

```sh
TASK_STAGE=$(mktemp -d)
cp notebooks/download_shared_benchmarks.ipynb "$TASK_STAGE/"
cp configs/kaggle_shared_benchmarks_recovery.json "$TASK_STAGE/kernel-metadata.json"
venv/bin/kaggle kernels push -p "$TASK_STAGE"
venv/bin/kaggle kernels status nguyenhung3105/vlm-recover-shared-benchmarks-v1
```

Không push đè notebook gốc trước khi phục hồi, để giữ Output cũ làm nguồn.
Bộ tải chỉ sao chép file từ Output gắn vào khi checksum đúng. Nếu Output cũ không
được Kaggle giữ/mount, dữ liệu không thể được khôi phục bằng code: Kaggle có thể
từ chối nguồn Input; khi đó dùng cấu hình job chuẩn để tải mới. Nếu mount tồn tại
nhưng không có file hợp lệ, bộ tải tự tải file còn thiếu. Chưa xác nhận phục hồi thật
cho đến khi thấy log `Reusing verified source` và cả ba bộ `Verified` trên Kaggle.

Nguồn checksum Aircraft bổ sung:
https://raw.githubusercontent.com/datasets-mila/datasets--fgvcaircraft/master/md5sums

Chưa hỗ trợ trực tiếp trong loader/launcher benchmark hiện tại (`runner_ready=false`).
Cần thêm loader, tên lớp Flowers và giao thức so paper trước khi inference. Flowers
test chính thức 6.149 ảnh khác cohort 818 ảnh của CascadeVLM, không so điểm trực tiếp.
Fakeddit và CIFAR-100 không tham gia job này.

## Kiểm tra local đã thực hiện

Kiểm tra cú pháp notebook độc lập, xác thực split/ánh xạ/ảnh Aircraft bằng archive
giả, từ chối archive hỏng; xác nhận Cars chỉ chọn shard train/test. Chưa xác nhận
tải đầy đủ cho đến khi job Kaggle hoàn tất và có đủ manifest thực tế.

Nguồn:
- https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- https://huggingface.co/datasets/tanganke/stanford_cars
- https://github.com/pytorch/vision/issues/7670 (nguồn Cars cũ lỗi)
