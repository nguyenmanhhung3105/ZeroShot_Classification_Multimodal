# Chuẩn bị dữ liệu hai tầng trên Kaggle

Notebook: `notebooks/download_two_stage_datasets.ipynb`.
Script nguồn tương ứng: `scripts/download_two_stage_datasets.py`.

## Thao tác

1. Tạo notebook Kaggle, import file `.ipynb` ở trên bằng chức năng import/upload notebook.
2. Bật Internet, chọn Accelerator **None** (không cần GPU).
3. Muốn làm CIFAR-100 trước: sửa `DATASETS = ("cifar100",)` ở cuối cell code.
   Mặc định tải cả CIFAR-100 và Flowers102.
4. Chạy cell. Chỉ hoàn tất khi log có `Verified` cho từng bộ đã chọn.
5. Save Version và chọn Save & Run All để lưu Output. Đợi version hoàn tất.
6. Mở trang xem version đã hoàn tất (không phải màn hình Edit), vào **Output**.
7. Chọn **Create Dataset** từ Output (vị trí nút có thể khác theo giao diện).
   Tạo dataset private, đặt tên `VLM Two Stage Sources V1`, slug đề xuất
   `vlm-two-stage-sources-v1`. Giữ cả thư mục `two_stage_sources` và manifest.
   Đây là một dataset Kaggle chứa hai bộ nguồn, không gộp các mẫu/split với nhau.
8. Đợi dataset xử lý xong, mở trang Files và kiểm tra có đủ hai
   `download_manifest.json` cùng các archive. Ghi lại ID thực tế trên URL,
   dự kiến `nguyenhung3105/vlm-two-stage-sources-v1` nếu slug chưa được dùng.
   Không tạo dataset từ Output của version lỗi hoặc còn tải dở.

## Dùng dataset lâu dài với Kaggle CLI

Sau khi tạo dataset, kiểm tra từ Terminal trên Mac (tại thư mục dự án):

```sh
venv/bin/kaggle datasets status nguyenhung3105/vlm-two-stage-sources-v1
venv/bin/kaggle datasets files nguyenhung3105/vlm-two-stage-sources-v1
```

Thay ID nếu Kaggle tạo slug khác. Không cần tải dataset xuống Mac.
Job sau này gắn ID vào `kernel-metadata.json`, ví dụ phần cấu hình:

```json
{
  "dataset_sources": ["nguyenhung3105/vlm-two-stage-sources-v1"]
}
```

Đây chỉ là đoạn trích, không phải metadata đầy đủ để push. Nếu job có dataset code,
giữ ID dataset code trong cùng danh sách. Notebook thí nghiệm sẽ đọc dữ liệu tại
`/kaggle/input/`; kiểm tra đường dẫn mount thực tế và thư mục lồng trong trang Files.
Không chạy lại notebook tải nguồn trong từng job thí nghiệm.
Gửi một kernel đã chuẩn bị đầy đủ bằng `venv/bin/kaggle kernels push -p <thu_muc_job>`.

Launcher hiện tại chưa hỗ trợ hai benchmark mới; không chỉ thêm tên job rồi gửi.
Cần bổ sung loader, cấu hình nguồn và notebook chạy thí nghiệm sau khi dataset sẵn sàng.
Khi cập nhật nguồn, tạo version của dataset hiện có để giữ ID, thay vì tạo nhiều
dataset trùng tên; ghi nhận version/checksum thực tế trong kết quả thí nghiệm.

Không cần tải ảnh xuống Mac hoặc upload lại ảnh từ Mac. Nếu tài khoản không bật được
Internet, cần hoàn thành điều kiện xác minh mà Kaggle hiển thị trước khi chạy.

## Đầu ra

Trong `/kaggle/working/two_stage_sources/`:

- `cifar100/cifar-100-binary.tar.gz`: ảnh binary gốc, giữ cả nhãn coarse/fine.
- `cifar100/download_manifest.json`: checksum, số mẫu, tên lớp và ánh xạ 100→20.
- `flowers102/102flowers.tgz`, `imagelabels.mat`, `setid.mat`: ảnh, nhãn và split chính thức.
- `flowers102/download_manifest.json`: checksum, số mẫu, quy ước nhãn/ID.

Kiểm tra mong đợi:

| Bộ | Train | Validation | Test |
|---|---:|---:|---:|
| CIFAR-100 | 50.000 | Chưa tách | 10.000 |
| Flowers102 | 1.020 | 1.020 | 6.149 |

CIFAR-100: 100 lớp con, 20 nhóm, mỗi nhóm 5 lớp; nhãn bắt đầu từ 0.
Flowers102: 102 lớp; nhãn và image ID trong nguồn bắt đầu từ 1.
Notebook kiểm tra MD5 trước khi đọc archive; lưu thêm SHA256, kiểm tra ánh xạ
CIFAR và tính đầy đủ/không giao nhau của các split Flowers.
Chạy lại sẽ dùng file đã tải nếu checksum đúng; file `.part` chưa hoàn tất sẽ tải lại.
Nếu lỗi checksum của file cuối cùng, dùng thư mục Output mới rồi chạy lại.

## Giới hạn và bước tiếp theo

Đây là **nguồn đã tải và kiểm tra**, chưa phải dataset chạy được với CLI hiện tại
(`runner_ready=false`). Chưa chuyển ảnh CIFAR thành PNG, chưa tạo manifest theo loader
dự án, chưa thêm tên lớp Flowers102 và chưa khai báo job benchmark mới.
Flowers102 có taxonomy phẳng; không tự chia nhóm hai tầng.

Cần bổ sung loader riêng và giao thức trước khi chạy thí nghiệm. Với CIFAR-100,
tách dev từ train bằng seed cố định; giữ nguyên test. Với Flowers, dùng validation
chính thức để chọn tham số. Split Flowers này có 6.149 ảnh test, không đồng nhất
với cohort 818 ảnh được báo cáo trong CascadeVLM; chỉ so số paper sau khi khớp giao thức.

Fakeddit: dùng lại nguồn v2 đã có trong `configs/kaggle_launcher.yaml` và cấu hình
dự án; notebook này không đọc hay thay đổi Fakeddit, không tạo lại cohort.

Nguồn chính thức:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- Flowers102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Chưa chạy tải thật trên Kaggle hoặc xác minh khả năng truy cập từ tài khoản.
Việc chuẩn bị này không huấn luyện, không inference và không gửi GPU job.
