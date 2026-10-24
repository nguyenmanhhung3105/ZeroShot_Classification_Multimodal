# Dữ liệu chính và bản v1

Từ lần chuyển dữ liệu ngày 20/10/2026, cấu hình chính tiếp tục dùng các tên
thư mục `pro_fakeddit`, `pro_crisismmd`, `pro_mmimdb`. Mỗi bộ chính có 3.000
bản ghi và 3.000 ảnh đã kiểm tra giải mã; bản trước được giữ nguyên trong
các thư mục cùng tên có hậu tố `_v1`.

```text
data/processed/
  pro_fakeddit/       fakeddit_sample.tsv, images/, manifest.json, experiment.yaml
  pro_crisismmd/      pre_crisismmd.tsv, images/, manifest.json, experiment.yaml
  pro_mmimdb/         mmimdb_sample.tsv, images/, manifest.json, experiment.yaml
  pro_fakeddit_v1/    bản dữ liệu trước khi chuyển
  pro_crisismmd_v1/   bản dữ liệu trước khi chuyển
  pro_mmimdb_v1/      bản dữ liệu trước khi chuyển
```

`configs/experiment_config.yaml` trỏ đến dữ liệu chính, giới hạn 3.000 mẫu mỗi
bộ và bật sample lock ở `results/_system/sample_locks/processed_v2`.
`configs/experiment_config_v1.yaml` giữ lựa chọn cũ và trỏ sang các thư mục v1.
Không đổi hoặc tính lại kết quả cũ trong `results/`.

## Nguồn nhãn và cách dùng

| Bộ | Nguồn | Cột nhãn runner sử dụng |
|---|---|---|
| Fakeddit | Validation nguồn, quota theo tỷ lệ sáu lớp sau lọc text | `2_way_label` cho 2way; `6_way_label` cho 6way |
| CrisisMMD | Phần **train chung** của hai task, nhãn ảnh | `label_informative` hoặc `label_humanitarian` |
| MM-IMDb | Test chính thức | `genres`, danh sách nhiều nhãn |

Mỗi TSV còn giữ `y_true` để đối chiếu. Với CrisisMMD dùng chung hai task,
`y_true` là JSON chứa hai khóa `informativeness` và `humanitarian`; runner
đọc cột riêng của task để tính metric, không đưa JSON đó vào bộ phân loại
một nhãn. Các cột `y_true_informative` và `y_true_humanitarian` cũng được lưu.

## CrisisMMD chạy được cả hai task trên cùng 3.000 cặp

Đã lấy giao của train informativeness và train humanitarian: 10.433 cặp nguồn
hợp lệ. Chọn 3.000 cặp bằng seed 42, phân tầng theo cặp nhãn rồi theo sự kiện;
giữ nguyên cả hai nhãn ảnh. Không ghép train của task này với test của task kia.

Trong `configs/experiment_config.yaml`, chỉ đổi:

```yaml
task: informativeness
```

thành:

```yaml
task: humanitarian
```

Hai cấu hình `label_col_info` và `label_col_human` đã đúng; không thay data_path
hoặc tạo bộ ảnh khác. Có thể dùng hồ sơ `configs/launcher_3000.yaml` chọn từng
job/task và model. Các job cũ trong hồ sơ khác vẫn có giới hạn mẫu riêng;
muốn chạy đủ 3.000 cần đặt `max_samples: 3000` hoặc dùng hồ sơ 3.000.

Phân bố informativeness: Informative 1.551; Not Informative 1.449. Có đủ bảy
sự kiện. Humanitarian có tám lớp với support tương ứng: Affected Individuals
86; Infrastructure and Utility Damage 603; Injured or Dead People 22;
Missing or Found People 2; Not Humanitarian 1.449; Other Relevant Information
404; Rescue, Volunteering or Donation Effort 382; Vehicle Damage 52.
Đây là phân bố gần nguồn, không cân bằng đều tám lớp; lớp chỉ có hai mẫu
chưa cho ước lượng F1 ổn định. Bộ này phục vụ khảo sát trên train, không gọi là
kết quả test chính thức 2.237 cặp của mỗi task.

## Bổ sung ảnh Fakeddit

Đã có 1.973 cặp local hợp lệ theo bộ lọc bỏ dấu cách và độ dài `>20`.
Công cụ tải bù từng lớp từ chính URL annotation nguồn, xác thực ảnh trước
khi nhận và thay ứng viên cùng lớp khi URL lỗi. Quota cuối cho 3.000 mẫu:

| 6way | Nhãn | Số mẫu |
|---|---|---:|
| 0 | True (Thật) | 1.537 |
| 1 | Satire/Parody (Châm biếm/nhại) | 204 |
| 2 | False Connection (Liên kết sai) | 602 |
| 3 | Imposter Content (Nội dung mạo danh) | 92 |
| 4 | Manipulated Content (Nội dung bị chỉnh sửa) | 407 |
| 5 | Misleading Content (Nội dung gây hiểu lầm) | 158 |

Nhãn nhị phân tương ứng: Real 1.537 và Fake 1.463. Cùng một TSV dùng được
cho cả hai cách phân loại. Không đổi ID nhãn: 0 của 2way là Fake, 0 của 6way
là True.

Lệnh tải bù và chuẩn bị được dùng trước khi chuyển bản chính:

Tổng cộng đã nhận thêm **1.036 ảnh hợp lệ** qua 1.120 lượt thử URL; 84 lượt
không hợp lệ được bỏ qua. Trong đó có 9 ảnh thay cho placeholder Imgur đã
có trong dữ liệu cũ. Placeholder được nhận diện bằng checksum đã kiểm tra
bằng mắt, bổ sung ngoài kiểm tra ảnh có mở được. Không dùng ảnh thông báo
"image no longer available" làm ảnh của mẫu đánh giá.

Ba nghìn bản ghi Fakeddit có ID và đường dẫn ảnh riêng. Có hai cặp bản ghi
dùng cùng nội dung ảnh (2.998 checksum ảnh khác nhau); đây là các ảnh logo
nguồn, được giữ cùng từng caption và nhãn annotation, không phải mẫu được
nhân bản để bù số lượng.

```sh
venv/bin/python -B -u scripts/prepare_labeled_cohort.py \
  --dataset fakeddit --task 6way --split validate --samples 3000 --seed 42 \
  --download-missing --max-download-attempts 6000 \
  --output-dir data/processed/.prepared_fakeddit
```

Khi tạo thành công, lệnh chuyển giữ nguyên dữ liệu trước đó ở `_v1` và cập
nhật cấu hình, đường dẫn ảnh cùng checksum TSV:

```sh
venv/bin/python -B scripts/promote_processed_data.py \
  --dataset fakeddit --source data/processed/.prepared_fakeddit
```

Đây là các bước đã thực hiện cho lần chuyển này; không cần chạy lại trên bản
chính đang đủ mẫu. Công cụ từ chối ghi đè thư mục v1 có sẵn. Ảnh tải mới được
lưu ở cache `data/raw/fakeddit/images`; `pro_fakeddit/images` chứa bản sao của
đúng 3.000 ảnh đầu ra. Thiếu quota sẽ làm bước chuẩn bị dừng; không chuyển
một bộ thiếu mẫu thành dữ liệu chính.

## Đối chiếu và tái lập

`manifest.json` của từng bộ ghi nguồn/hash, seed, text policy, nhãn từng ID,
hash ảnh, support, cohort ID và hash TSV. Khối `promotion` ghi hash TSV cũ và
đường dẫn v1 để kiểm tra bản lưu không thay đổi. Log đánh giá dùng `source_split`
riêng của dataset, tránh gọi chung cả train/test/validation là `val`.

Chưa chạy suy luận mới trên ba bộ chính này. Các kết quả cũ và báo cáo giữa kỳ
vẫn mô tả dữ liệu v1; không ghép kết quả đó với cohort mới để tính mức cải thiện.

Kết quả kiểm tra sau chuyển:
[du_lieu_chinh_3000_kiem_tra.json](du_lieu_chinh_3000_kiem_tra.json).
Đã đối chiếu 9.000 ID/đường dẫn/nhãn thật, checksum mọi ảnh, checksum TSV và
hash TSV của ba bản v1. Đã đạt 18 kiểm thử cohort/chuyển dữ liệu và 9 kiểm thử
audit, gồm kiểm tra nguồn split riêng theo dataset và hoàn tác khi chuyển lỗi.
