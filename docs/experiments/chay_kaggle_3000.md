# Kiểm tra Kaggle và chạy thí nghiệm trên cohort chính 3.000 mẫu

Chạy các lệnh từ thư mục gốc dự án. Hồ sơ sử dụng là
`configs/launcher_3000.yaml`, không dùng giới hạn 1.000 mẫu của hồ sơ cũ.

## 1. Dataset và điều kiện nhận kết quả

| Dataset trên Kaggle | Nội dung đúng của bản mới |
|---|---|
| `nguyenhung3105/vlm-runs-data-fakeddit` | 3.000 ảnh + `fakeddit_sample.tsv` + manifest cohort |
| `nguyenhung3105/vlm-runs-data-crisismmd` | 3.000 ảnh + `pre_crisismmd.tsv` có hai nhãn + manifest cohort |
| `nguyenhung3105/vlm-runs-data-mmimdb` | 3.000 ảnh + `mmimdb_sample.tsv` + manifest cohort |
| `nguyenhung3105/vlm-runs-code` | Mã suy luận/đánh giá, prompt/descriptor và khai báo thư viện hiện tại |

Mỗi gói dữ liệu có đúng **3.002 tệp**. Không upload thư mục `_v1`.
Dataset chung `vlm-runs-data` là đường dùng cũ; hồ sơ mới dùng ba dataset
riêng với `data_mode: full_dataset`.

Các cohort ID phải khớp:

| Dữ liệu | Cohort ID | Nguồn |
|---|---|---|
| Fakeddit | `455fc8feb8733f1d394c` | Validation, dùng chung 2way/6way |
| CrisisMMD | `0451b45a4274038830ed` | Train chung hai task, nhãn ảnh |
| MM-IMDb | `c308fdf53c1f0b1ff369` | Test chính thức |

Kiểm tra Kaggle ban đầu ngày 20/10/2026 cho thấy các dataset dữ liệu vẫn có
`lastUpdated` ngày 14/10 và mã ngày 16/10, trước lần chọn lại dữ liệu.
Metadata MM-IMDb trả kích thước 0 nên không được coi là đã đồng bộ xong.
Tên dataset giữ nguyên không chứng minh nội dung đúng; phải đối chiếu bản mới.

## 2. Đồng bộ lại dữ liệu và mã, không khởi chạy GPU

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml --sync-data all

venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml --sync-code
```

Hai lệnh luôn kiểm tra lại từ xa dù state local trùng SHA. Khi nội dung khác,
launcher tạo version dataset mới, không xóa các version trước. Mỗi gói có
manifest và checksum. Bulk sync dữ liệu kiểm tra tên/kích thước để tránh tải
ZIP lớn còn trỏ bản cũ; bước CPU phía dưới xác minh hash nội dung thực sự mount.
Kết quả kiểm tra ghi tại
`results/_system/launches/dataset_verification.json`:

- `verification: sha256`: hash cây file tải về khớp local.
- `verification: file_manifest`: chỉ khớp tên/kích thước; chưa là bằng chứng
  hash nội dung. Cần bước kiểm chứng mount phía dưới để kết luận mạnh hơn.

Khi đồng bộ lại bản không đổi, không cần cố tạo version mới. Không dùng
`--delete-old-versions` và không đổi dataset thành public.

## 3. Kiểm chứng dữ liệu mount trên Kaggle bằng CPU

Trạng thái ghi nhận ngày 20/10/2026: bốn gói đã đồng bộ và khớp danh sách
tên/kích thước từ API. Lượt CPU version 1 chuyển `cancel_acknowledged`, không
trả log/biên nhận; version 2 đang chạy. **Chưa xác nhận SHA nội dung mount.**
Biên bản trạng thái: [kaggle_3000_sync_status.json](kaggle_3000_sync_status.json).
Notebook chẩn đoán CPU không gắn dataset đã hoàn tất và có biên nhận Python
thực thi thành công. Kết quả đó không xác nhận nội dung dataset. Tiến trình
chờ local đã dừng; notebook dữ liệu version 2 vẫn được ghi nhận đang chạy,
chưa có kết luận về bước mount hay checksum.
Để tiếp tục đúng lượt version 2 đang chạy, dùng lệnh sau trước khi chạy GPU:

```sh
venv/bin/python -B scripts/audit_kaggle_cohorts.py \
  --launcher-config configs/launcher_3000.yaml \
  --resume-stage results/_system/launches/cohort_checks/check_wmq8l6mz
```

Khi đã có biên nhận hợp lệ, dùng trạng thái trong biên nhận thay cho ghi nhận
đang chờ ở trên. Không chạy lại `--submit` khi lượt cũ còn đang chạy.
Lệnh dưới đây chỉ dùng để tạo một lượt kiểm tra CPU mới khi cần:

```sh
venv/bin/python -B scripts/audit_kaggle_cohorts.py \
  --launcher-config configs/launcher_3000.yaml --submit
```

Lệnh gửi notebook **private, CPU, không internet, không nạp model** tại
`nguyenhung3105/vlm-data-check-3000`, không thay notebook thí nghiệm `vlm-runs`.
Notebook mount cả bốn dataset, đối chiếu SHA cây file với local, rồi kiểm tra
ID, `y_true`, split và số lượng cho cả năm task. Kết quả phải báo
`[VERIFIED] 4 gói và 5 task trên Kaggle CPU`.

Biên nhận sau thành công:
`docs/experiments/kaggle_3000_remote_check.json`. Trong đó `gpu_used=false`,
`models_loaded=false`; mỗi task có `cohort_samples=3000`, `requested_samples=3000`.
Chỉ ghi "đã xác minh hash dữ liệu trên Kaggle" khi có biên nhận khớp audit ID
và toàn bộ SHA kỳ vọng. File chưa tồn tại hoặc một kiểm tra thất bại không được
diễn giải là đã PASS.

Nếu notebook CPU còn chạy khi hết thời gian chờ hoặc mất kết nối, tiếp tục
đúng `stage=` đã in, không gửi thêm lượt bằng `--submit`:

```sh
venv/bin/python -B scripts/audit_kaggle_cohorts.py \
  --launcher-config configs/launcher_3000.yaml \
  --resume-stage results/_system/launches/cohort_checks/THU_MUC_CUA_LUOT_DA_GUI
```

Lệnh này không tạo notebook mới; biên nhận tải về vẫn phải khớp audit ID và
version của lượt cũ.

## 4. Ma trận thí nghiệm phục vụ báo cáo

| Nhóm | Task | Backbone | Lượt |
|---|---:|---:|---:|
| Baseline ảnh + text | Fakeddit 2way/6way, CrisisMMD info/humanitarian, MM-IMDb genres: 5 task | 4 | 20 |
| DCLIP adapted ảnh + text | CrisisMMD info/humanitarian: 2 task | 4 | 8 |
| **Tổng** | | | **28** |

Bốn backbone: `metaclip2_vith14`, `dfn5b_vith14`, `siglip2_so400m`,
`openai_clip_vit_b_32`. Mỗi lệnh chỉ chạy một job/model/phương pháp.
Các phương pháp dùng cùng TSV và nhãn trong mỗi task; tắt quan hệ ảnh–text
bổ sung, giữ fusion/scale/ngưỡng hiện tại.

Chênh lệch DCLIP adapted gồm thay descriptor và quy tắc tổng hợp nhánh ảnh;
không gọi đây là tác động riêng của nội dung mô tả. Có thể bổ sung đối chứng
`--method current_mean_sim` trên các job adapted để tách yếu tố tổng hợp,
nhưng các lượt đó nằm ngoài ma trận chính 28 lượt.

Toàn bộ lệnh riêng đã sinh và kiểm tra:
[lenh_kaggle_3000.sh](lenh_kaggle_3000.sh). File là danh sách để chọn lệnh;
chạy tuần tự, chờ một lệnh hoàn tất trước lệnh tiếp theo vì dùng chung kernel.
Sinh lại danh sách bằng:

```sh
venv/bin/python -B scripts/list_kaggle_commands.py \
  --launcher-config configs/launcher_3000.yaml --configured-only
```

## 5. Xem trước, kiểm tra nhỏ rồi chạy đủ 3.000

Ví dụ MetaCLIP2 informativeness. Lệnh xem trước không gọi API Kaggle và
không chạy model:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml \
  --job crisismmd_info --model metaclip2_vith14 \
  --use-kaggle true --dry-run
```

Nếu cần kiểm tra môi trường GPU và tải trọng số trước lượt dài, chạy 5 mẫu:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml \
  --job crisismmd_info --model metaclip2_vith14 \
  --max-samples 5 --use-kaggle true
```

Đây là lượt kiểm tra nhỏ, không đưa điểm số vào bảng 3.000 mẫu. Gói dữ liệu
mount vẫn là cohort 3.000; `requested_samples` của lượt này là 5.

Chạy đầy đủ baseline:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml \
  --job crisismmd_info --model metaclip2_vith14 --use-kaggle true
```

Chạy đối chứng DCLIP cùng backbone/task/cohort:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml \
  --job adapted_crisismmd_info --model metaclip2_vith14 \
  --method dclip --use-kaggle true
```

Với các task baseline khác, dùng job `fakeddit_2way`, `fakeddit_6way`,
`crisismmd_humanitarian` hoặc `mmimdb_genres`. DCLIP humanitarian dùng
`adapted_crisismmd_humanitarian`. Giữ các flag còn lại theo file 28 lệnh.

## 6. Tải kết quả và đưa vào báo cáo

Launcher đợi kernel hoàn tất rồi tải `summary.tsv`, `run_artifacts.zip` và
`run_receipt.json` vào cây `results/<method>/<dataset>/<task>/samples_3000/<model>/...`.
Worker kiểm tra lại hash dữ liệu và nhãn trước suy luận; launcher từ chối
receipt sai run ID, hash mã/dữ liệu hoặc cohort/task/số mẫu yêu cầu.

Sau mỗi lượt, kiểm tra:

1. Receipt đúng cohort ID, task, `requested_samples=3000` và GPU thực tế.
2. Summary/raw predictions có **3.000 mẫu thực sự đánh giá**, không chỉ cấu
   hình yêu cầu 3.000. Các ID trong cùng đối chứng phải giống nhau.
3. Bài toán một nhãn: Accuracy, Macro-F1, Balanced Accuracy và F1/support
   từng lớp. MM-IMDb: Micro-F1, Macro-F1, Samples-F1, Hamming Accuracy.
4. Điểm được lấy từ output, không điền theo dự kiến. `sample_set_id` có thể
   khác giữa môi trường do metadata thời gian tệp; đối chiếu thêm cohort ID,
   nhãn, ID mẫu và checksum nội dung để kết luận cùng đầu vào.

CrisisMMD là khảo sát trên train nguồn; Fakeddit là validation. Chỉ MM-IMDb
của ma trận này lấy từ test chính thức. Không gọi cả ba là test độc lập và
không ghép kết quả v1 vào bảng mới dù cùng N=3.000. Bản LaTeX giữa kỳ giữ
các điểm v1 đã có và ghi rõ cohort mới chưa có kết quả GPU.

Nếu mất kết nối sau khi đã gửi job, dùng đường dẫn `stage=` của đúng lượt:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/launcher_3000.yaml \
  --resume-stage results/_system/launches/staging/THU_MUC_CUA_LUOT_DA_GUI
```

Lệnh resume không tạo GPU mới. Nếu API báo cần đăng nhập lại, dùng
`venv/bin/kaggle auth login --force`, hoàn tất trên máy rồi resume; không
gửi token trong cuộc trò chuyện.

Đối chiếu local trước upload:
[kaggle_3000_local_check.json](kaggle_3000_local_check.json): 28 cấu hình hợp
lệ và bảy gói job được kiểm tra; mỗi gói có 3.002 tệp và không chứa v1.
