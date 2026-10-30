# Tạo lại cohort có y_true để đánh giá zero-shot

**Dữ liệu chính hiện dùng tên thư mục `pro_*`, bản cũ ở `pro_*_v1`.** Xem
[trạng thái và cách dùng dữ liệu chính 3.000 mẫu](du_lieu_chinh_3000.md).
Các ví dụ `cohorts_v2/` bên dưới mô tả cách tạo cohort riêng, không cần chạy
lại để thay thế dữ liệu chính.

Công cụ: `scripts/prepare_labeled_cohort.py`. Chạy từ thư mục gốc bằng Python
trong môi trường đã cài thư viện của dự án. Các lệnh chuẩn bị dưới đây không
huấn luyện, không chạy model và không gửi dữ liệu lên Kaggle.

## Nhãn thật và đầu ra

Nhãn đều được lấy từ annotation có sẵn, không dùng dự đoán model làm nhãn:

| Dataset/task | Nguồn y_true | Kiểu trong TSV |
|---|---|---|
| Fakeddit 2way | `2_way_label` | `0` Fake, `1` Real |
| Fakeddit 6way | `6_way_label` | ID `0`–`5`, giữ nguyên nghĩa nhãn gốc |
| CrisisMMD | `label_image` mặc định; có thể chọn `label_text` hoặc `label` chính thức | Khóa lớp tiếng Anh như `informative` |
| MM-IMDb | Các genre hợp lệ trong JSON gốc | Mảng JSON, ví dụ `["Drama", "Short"]` |

Mỗi bộ xuất ra thư mục **mới**, gồm:

- `samples.tsv`: có `y_true`, cột nhãn gốc tương ứng, nguồn tệp/bản ghi,
  `source_split`, `label_source`, text và đường dẫn ảnh.
- `images/`: bản sao ảnh của đúng các mẫu được chọn; không đổi pixel hoặc resize.
- `manifest.json`: N, seed, nguồn/hash, phân bố nhãn/sự kiện, lớp ít mẫu, mẫu bị
  loại, nhãn thật từng ID, hash text/ảnh và mã cohort. Với MM-IMDb còn có số nhãn
  trên phim, đồng xuất hiện và sai lệch tần suất nhãn.
- `experiment.yaml`: sao chép cấu hình thí nghiệm hiện tại, chỉ bật dataset/task
  đang chuẩn bị, trỏ đến dữ liệu mới, dùng **cột `y_true`** để tính metric và
  bật sample lock. Prompt, backbone, fusion, chunking và ngưỡng giữ từ cấu hình
  gốc. Đường dẫn cấu hình gốc đổi được bằng `--base-config`.

Công cụ từ chối xuất bộ mẫu khi thiếu N mẫu hợp lệ, nhãn không đúng điều kiện,
hoặc vi phạm kiểm tra định danh/split. Dòng có text/nhãn/ảnh không hợp lệ được
loại và đếm theo lý do; lỗi định danh/giao nhau split làm dừng. Công cụ không
lặp mẫu hoặc tự sinh y_true để bù số lượng. Đường dẫn đầu ra đã tồn tại sẽ bị
từ chối, không có tùy chọn ghi đè.

Ảnh được kiểm tra cấu trúc và thử giải mã RGB trước khi chọn; ID phim được giữ
dạng chuỗi để không mất số 0 đầu. Text được chuẩn hoá Unicode NFC, giải mã HTML
và gộp khoảng trắng; giữ ngôn ngữ, nội dung và hashtag. Quy tắc text mới được
ghi trong manifest; không coi cohort mới là đầu vào giống hệt thực nghiệm cũ.

## 1. CrisisMMD: 3.000 mẫu có nhãn cho thực nghiệm phát triển

Chọn từ **train chính thức**, nguồn nhãn ảnh, phân tầng theo nhãn rồi theo sự
kiện trong từng nhãn. Không lấy 3.000 dòng đầu. Dùng đúng `tweet_id` trong bảng
split, kiểm tra khớp `image_id`, kiểm tra tweet không giao nhau giữa các split.

Informativeness:

```sh
venv/bin/python -B scripts/prepare_labeled_cohort.py \
  --dataset crisismmd --task informativeness --split train \
  --label-source image --samples 3000 --seed 42 \
  --output-dir data/processed/cohorts_v2/crisismmd_info_train3000_s42
```

Humanitarian:

```sh
venv/bin/python -B scripts/prepare_labeled_cohort.py \
  --dataset crisismmd --task humanitarian --split train \
  --label-source image --samples 3000 --seed 42 \
  --output-dir data/processed/cohorts_v2/crisismmd_human_train3000_s42
```

Hai task có danh sách split khác nhau nên hai cohort có thể gồm các ID khác
nhau. So sánh backbone/baseline/DCLIP **trong cùng task** phải dùng cùng một
TSV và manifest; không tạo lại cohort riêng theo từng model.

Đây là **3.000 mẫu từ train dùng để khảo sát zero-shot**, có y_true để tính
Accuracy/F1; việc chuẩn bị này không huấn luyện trọng số. Không gọi nó là test
chính thức hoặc bằng chứng tổng quát hoá cuối cùng. Chọn `--label-source text`
hoặc `official` sẽ thay định nghĩa tác vụ; khi đổi phải dùng thư mục mới và
đánh giá lại mọi phương pháp trên cùng nguồn nhãn đó.

Để đánh giá trên toàn bộ test chính thức (trước lọc có 2.237 cặp/task):

```sh
venv/bin/python -B scripts/prepare_labeled_cohort.py \
  --dataset crisismmd --task informativeness --split test \
  --label-source image --samples 2237 --seed 42 \
  --output-dir data/processed/cohorts_v2/crisismmd_info_test2237_s42
```

Với humanitarian, thay `--task humanitarian` và dùng thư mục đầu ra khác.
Nếu ảnh/text lỗi khiến không đủ 2.237, công cụ dừng; cần kiểm tra nguyên nhân
và công bố N hợp lệ thực tế. Không ghép dev vào test để bù. Các mẫu từng tham
gia thực nghiệm cũ không tự trở thành một holdout mới chỉ vì được chọn lại.

## 2. MM-IMDb: 3.000 phim trong test chính thức

```sh
venv/bin/python -B scripts/prepare_labeled_cohort.py \
  --dataset mmimdb --task genres --split test \
  --samples 3000 --seed 42 \
  --output-dir data/processed/cohorts_v2/mmimdb_genres_test3000_s42
```

Chỉ đọc nội dung JSON của ID trong split đã chọn. Giữ 23 genre của hệ thống,
gồm Short; lọc nhãn không thuộc danh sách đó và loại nhãn lặp trong cùng phim.
Chọn plot dạng chuỗi hợp lệ đầu tiên, fallback sang `plot outline`; không ép
`null` thành chuỗi `"None"`. Không thêm bộ lọc `kind=movie`: membership chính
thức và annotation genre quyết định pool, tránh tự loại phim ngắn khỏi bài toán.

Phân tầng lặp bậc một chia pool hợp lệ thành 3.000 mẫu và phần còn lại, ưu
tiên giữ gần support từng genre. Không ép mọi genre bằng nhau và không chọn
một phim nhiều lần. Đây không phải bảo đảm tối ưu mọi cặp đồng xuất hiện;
manifest có thống kê để kiểm tra thêm.

## 3. Fakeddit: chuẩn bị thêm ảnh rồi chốt 3.000 mẫu

Theo kiểm tra trước đó, ảnh local hiện chưa đủ. Công cụ tìm đồng thời trong
`data/processed/pro_fakeddit/images` và `data/raw/fakeddit/images`, vì downloader
cũ ghi vào thư mục raw. Nếu ảnh ở nơi khác, truyền `--image-dir` một hoặc nhiều
lần; khi truyền, danh sách này thay thế hai thư mục mặc định.

Ví dụ dưới đây dùng **validation** để tiếp tục khảo sát, không gọi là test.
Nếu muốn public test, thay `--split test_public` trong cả bước lập kế hoạch và
bước tạo cohort, đồng thời đổi tên thư mục đầu ra.

Bước 1: tạo danh sách ứng viên tải, tối đa 6.000 mẫu trước khi loại những ID đã
có tệp ảnh local. Bước này chỉ ghi kế hoạch, không phải bộ đánh giá hoàn chỉnh:

```sh
venv/bin/python -B scripts/prepare_labeled_cohort.py \
  --dataset fakeddit --task 6way --split validate \
  --samples 3000 --seed 42 --plan-downloads \
  --output-dir data/processed/cohort_plans/fakeddit_validate3000_s42
```

Bước 2: tải các URL trong kế hoạch bằng downloader hiện có:

```sh
venv/bin/python -B src/preprocessing/data_fakeddit/image_downloader.py \
  data/processed/cohort_plans/fakeddit_validate3000_s42/download_candidates.tsv
```

Lệnh này truy cập mạng và ghi ảnh vào `data/raw/fakeddit/images`. URL có thể
không còn hoạt động; số ứng viên không bảo đảm số ảnh tải thành công. Downloader
cũ bỏ qua tệp đã tồn tại; ảnh local hỏng cần được thay bằng bản tải hợp lệ
trước khi thử lại, không chỉ chạy lại downloader trên cùng tệp hỏng.

Bước 3: xác thực text/ảnh và chọn cohort chính thức cho lần thử:

```sh
venv/bin/python -B scripts/prepare_labeled_cohort.py \
  --dataset fakeddit --task 6way --split validate \
  --samples 3000 --seed 42 --min-text-length 20 \
  --output-dir data/processed/cohorts_v2/fakeddit_6way_validate3000_s42
```

Quota sáu lớp được tính theo tỷ lệ pool raw hợp lệ, không theo phần ảnh local
có sẵn. Thiếu ảnh của một lớp sẽ được báo đúng số thiếu; không âm thầm lấy
thêm lớp khác để đủ N. Nếu còn thiếu ảnh, tạo kế hoạch ở thư mục mới với
`--candidate-multiplier 3` hoặc lớn hơn để thêm ứng viên, rồi chạy lại bước 2–3.
Không cần đổi seed của cohort để bù URL lỗi. Giữ cùng split nguồn.

Với 2way, thay `--task 2way` và thư mục đầu ra. Cách chọn vẫn phân tầng sáu
nhãn để tránh làm méo thành phần fake, nhưng `y_true` và config dùng nhãn nhị
phân. Không chạy lại các script cũ để ghi đè `fakeddit_sample.tsv`.

## 4. Kiểm tra trước khi chạy model

Chỉ sử dụng đầu ra khi chương trình báo `status: ready`, manifest có
`selected.samples = 3000` và support đúng số mẫu mong muốn. Với MM-IMDb, tổng
support lớn hơn N là bình thường vì một phim có nhiều nhãn. Đọc thêm
`low_support_labels`, sai lệch tỷ lệ nhãn, phân bố sự kiện và số mẫu bị loại.

`status: download_plan_only` không có `samples.tsv`/`experiment.yaml` và chưa
thể dùng tính metric.

Sau khi chuẩn bị xong, ví dụ lệnh baseline đọc cohort informativeness mới:

```sh
venv/bin/python -B src/run_experiment.py \
  --config data/processed/cohorts_v2/crisismmd_info_train3000_s42/experiment.yaml
```

Lệnh trên **có tải/chạy model** theo danh sách trong cấu hình. Với các cohort
khác, trỏ đến `experiment.yaml` tương ứng. Hồ sơ `configs/launcher_3000.yaml`
cũ vẫn đọc TSV cũ; không dùng hồ sơ đó để khẳng định đã chạy cohort v2.

Nếu dùng launcher/Kaggle sau này, cần trỏ `experiment_config` của một hồ sơ mới
đến bản cấu hình cohort đặt trong `configs/`, và đóng gói `samples.tsv` cùng
`images/` mới. Không sửa tên split trong YAML để biến một cohort train thành test.

Các kiểm thử tự động dùng fixture tổng hợp, không đọc bộ dữ liệu thật:

```sh
venv/bin/python -B -m unittest discover -s tests -p test_prepare_labeled_cohort.py -v
```

Kết quả kiểm tra triển khai: 11 kiểm thử cohort mới và 4 kiểm thử cấu hình đều
đạt. Bộ `test_runtime_fixes.py` cũ còn lỗi mock `never-read` ở các trường hợp
entry point; các file runner và bộ test đó không bị sửa trong lần triển khai
này. Chưa chạy chuẩn bị 3.000 mẫu trên dữ liệu thật, tải ảnh hoặc suy luận;
tính sẵn sàng của dữ liệu thật chỉ được xác nhận sau khi lệnh chuẩn bị hoàn
tất với `status: ready`.
