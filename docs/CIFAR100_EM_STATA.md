# Kiểm chứng luồng hai tầng CLIP + EM + StatA trên CIFAR-100

Đây là thích nghi đa lớp từ luồng Majority Fakeddit, không phải giữ nguyên một
classifier binary cho 20 nhóm. Backbone, prompt tên lớp và dev 10.000 ID seed 0
giữ như trước. Tầng 1 có ba phiếu CLIP, EM-Dirichlet và StatA cho 20 nhóm;
hai phiếu giống nhau thắng, ba phiếu khác nhau dùng phiếu CLIP. Tầng 2 chọn
argmax CLIP trong năm lớp con của nhóm đã chọn, không huấn luyện expert mới.

## Đối chứng và tham số

| Phương pháp | Phân loại phẳng | Phân cấp |
|---|---|---|
| CLIP | 100 lớp | Chọn nhóm CLIP rồi lớp con CLIP |
| EM | EM trên 100 lớp | EM trên 20 nhóm rồi lớp con CLIP |
| StatA | StatA trên 100 lớp | StatA trên 20 nhóm rồi lớp con CLIP |
| Majority | CLIP/EM/StatA trên 100 lớp | CLIP/EM/StatA trên 20 nhóm rồi lớp con CLIP |

Các nhánh dùng cùng feature và cùng partition query 75 ảnh, seed 0/1/2. CLIP
tất định chạy một lượt, sáu phương pháp có adaptation chạy ba seed. Tổng 20 lượt
metric, gom thành tám dòng mean ± sample std. Không chọn seed tốt nhất.

EM dùng softmax(30*cosine), 20 vòng, tối đa 1.000 MM mỗi vòng; dùng Hungarian
như implementation sẵn có để gán cụm theo xác suất CLIP, không dùng nhãn thật.
StatA giữ 10 vòng, inner 5, neighbors 3 và prior scale 100. Nhóm <=3 ảnh giữ
CLIP cho StatA. Không sửa solver `adaptation.py`. Torch CPU threads=1 trong
adaptation, khôi phục sau khi hoàn tất hoặc lỗi.

Đây là transductive: mẫu phụ thuộc các query không nhãn cùng nhóm. Flat EM dùng
100 thành phần, EM coarse dùng 20; đây là khác biệt cấu trúc cần ghi khi so sánh,
không xem chúng là cùng bài toán fit. Những nhóm có N<K vẫn dùng solver hiện tại,
không tự loại lớp hoặc đổi nhãn để chạy được. Solver lỗi thì dừng, không bỏ mẫu.

## Lệnh chạy một bước

Smoke trước:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job cifar100_em_stata_smoke --use-kaggle true --wait
```

Sau khi smoke thành công, full dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job cifar100_em_stata_dev --use-kaggle true --wait
```

Launcher kiểm tra quota, chặn trùng, gửi job GPU, theo dõi mỗi 30 giây, tải ZIP
và kiểm tra checksum. Model chỉ nạp một lần; ảnh chỉ mã hóa một lần trong job.
EM/StatA chạy CPU sau đó, nên full có thể chậm hơn các lượt CLIP trước; chưa có
dự báo thời gian. Timeout vẫn là 4 giờ cấu hình, không phải dự báo. Theo dõi log
`EM/StatA seed=... group ...`; không gửi lại khi job vẫn running/queued.

## Đầu ra và giới hạn

Ngoài metric/prediction/checksum chuẩn, có `adaptation_details.json` (MM steps,
objective đầu/cuối từng nhóm), `adaptation_groups.npz` (group ID, phiếu fine/coarse
mỗi seed), và `routing.json` (accuracy nhóm EM/StatA/Majority sau dự đoán).
Replay lưu feature, prototype và logits để không phải mã hóa ảnh lại khi phân tích.
Không báo top-5 từ one-hot của EM/Majority vì đó không phải phân phối xếp hạng lớp.

CALIP chưa nằm trong job này: nó cần global/patch feature và ablation riêng.
Không gọi ba phiếu hiện tại là CLIP/StatA/CALIP; đây là **CLIP/EM/StatA**.
Chưa thử trên test, không thay cohort Fakeddit và không ghi đè kết quả cũ.

Local đã kiểm tra majority đa lớp/tie, group dùng chung, tham số truyền solver,
fallback nhóm nhỏ, khôi phục thread, 8 method/3 seed, end-to-end backend/solver
giả và notebook whitelist. Chưa chạy full solver trên ảnh thật trong triển khai.
