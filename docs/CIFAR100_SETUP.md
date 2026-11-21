# Job tải riêng CIFAR-100

Đợt kiểm tra 10/11/2026 chưa tìm thấy Output/manifest CIFAR-100 được xác thực.
Các ID dataset đề xuất `vlm-two-stage-sources-v1` và `vlm-cifar100-sources-v1`
trả 403; điều này không chứng minh dữ liệu không tồn tại ở mọi vị trí.
Các notebook tải cũ được kiểm tra không có Output truy cập được.

Job riêng chỉ gọi `main(..., datasets=("cifar100",))`, không tải lại Flowers,
Cars, Aircraft hoặc Fakeddit. CPU, Internet bật, khoảng 0,17 GB archive nguồn.

Từ thư mục dự án, kiểm tra job chưa chạy để tránh trùng:

```sh
venv/bin/kaggle kernels status nguyenhung3105/vlm-prepare-cifar100-v1
```

Nếu chưa có job hoặc chủ động chạy lại sau khi job trước dừng:

```sh
TASK_STAGE=$(mktemp -d)
cp notebooks/download_cifar100.ipynb "$TASK_STAGE/"
cp configs/kaggle_cifar100_kernel.json "$TASK_STAGE/kernel-metadata.json"
venv/bin/kaggle kernels push -p "$TASK_STAGE"
```

Theo dõi mỗi hai phút; Ctrl+C chỉ dừng theo dõi:

```sh
while true; do
  date
  venv/bin/kaggle kernels status nguyenhung3105/vlm-prepare-cifar100-v1
  sleep 120
done
```

Chỉ hoàn tất khi job Complete và log `Verified cifar100` báo train=50.000,
test=10.000. Bộ tải kiểm tra MD5 archive binary chính thức, kích thước/số bản ghi,
giới hạn coarse/fine và ánh xạ 100 lớp→20 nhóm (5 lớp mỗi nhóm).
Giữ nguồn binary; chưa chuyển sang loader benchmark, chưa tách dev.

Output nằm ở `cifar100_sources/cifar100/`: archive cùng `download_manifest.json`.
Trên trang version thành công: Output → menu ⋮ cạnh Output → **New dataset**,
chọn Private, tên `VLM CIFAR100 Sources V1`, slug đề xuất `vlm-cifar100-sources-v1`.
Không cần gộp với dataset ba benchmark đã có.

## Kết quả kiểm tra các bộ còn lại

- Flowers102, Stanford Cars, FGVC Aircraft: đã tải manifest `sources_verified`
  từ Output `nguyenhung3105/vlm-recover-shared-benchmarks-v1`.
- `nguyenhung3105/vlm-shared-benchmarks-v1`: trạng thái Dataset `ready`.
- Fakeddit dev-v2 và test-v2: trạng thái Dataset `ready`; lần kiểm tra này chỉ
  xác minh trạng thái, không đọc lại 3.000 mẫu hay hash toàn bộ ảnh.
- Fakeddit train-v2: API status trả 403, chưa xác nhận truy cập/tồn tại. Không
  tự tạo lại hoặc đổi cohort. Nghiên cứu không fine-tune có thể dùng dev để phát
  triển và test để xác nhận; cần làm rõ vai trò train trước khi chuẩn bị thêm.

Đã kiểm tra cú pháp notebook độc lập, JSON metadata và cấu hình chỉ chọn CIFAR.
Chưa gửi job CIFAR trong lượt chuẩn bị này; người dùng chạy lệnh phía trên.
