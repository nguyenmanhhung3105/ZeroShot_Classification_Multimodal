# Rà soát dữ liệu gốc, tiền xử lý và phương án chọn mẫu

Ngày kiểm tra: 20/10/2026. Phạm vi: Fakeddit, CrisisMMD và MM-IMDb.

Triển khai bổ sung sau rà soát: công cụ mới `scripts/prepare_labeled_cohort.py`;
xem [lệnh tạo cohort có y_true](tao_cohort_co_y_true.md). Chưa chạy lại dữ liệu
thật; các mô tả "chưa sửa script" bên dưới là trạng thái tại lần rà soát ban đầu.

## Kết luận

**Cần chuẩn bị lại cohort đánh giá trước khi mở rộng thực nghiệm.** Nguyên nhân
chính là nguồn split, điều kiện hợp lệ và định danh; không phải cứ mất cân bằng
lớp là phải chia đều số mẫu.

| Bộ dữ liệu | Bằng chứng kiểm tra | Phương án đề nghị |
|---|---|---|
| Fakeddit | Có 2.841 ảnh local; 1.973 ID có tiêu đề gốc qua điều kiện độ dài của runner. Bộ tải và bộ tiền xử lý dùng hai thư mục ảnh khác nhau. | Đồng bộ điều kiện lọc và đường dẫn ảnh; bổ sung ảnh hợp lệ; chọn cohort mới sau kiểm tra ảnh. Có thể đặt mục tiêu 3.000 nhưng hiện chưa đủ. |
| CrisisMMD | 3.000 dòng hiện tại trộn train/dev/test, chỉ gồm ba sự kiện; 470 tweet ID không còn là chuỗi chữ số. Test chính thức của mỗi task có 2.237 cặp. | Dựng lại từ split chính thức, bảo toàn ID và nguồn nhãn. Dùng test 2.237 cặp/task trước kiểm tra ảnh, không ghép tập để ép đủ 3.000. |
| MM-IMDb | 3.000 phim hiện tại gồm 1.819 train, 284 dev và 897 test. Test nguồn có 7.799 cặp JSON–ảnh. | Chọn 3.000 phim hợp lệ trong riêng test; giữ gần phân bố nhiều nhãn của tập nguồn. |

Đây là phương án cho vòng đánh giá tiếp theo. Các kết quả cũ vẫn là kết quả thăm
dò trên cohort cũ; không đổi tên chúng thành kết quả trên test chính thức.
Trộn split không tự chứng minh đã có huấn luyện bằng nhãn hay rò rỉ nhãn vào
trọng số. Tuy nhiên, cohort đó không thể được mô tả là một split test độc lập
theo cách chia của bộ dữ liệu.

## Phương pháp kiểm tra và giới hạn

- Đã đọc mã nguồn của ba script `preprocess_*.py`, bộ tải ảnh Fakeddit và các
  nơi đọc dữ liệu tại inference. Không import/chạy các script tiền xử lý.
- Theo yêu cầu kiểm tra raw mới của người dùng, đã đọc README/split metadata
  ở `data/raw/`. TSV nguồn được đọc theo chunk 1.000 dòng, chỉ các cột cần cho
  thống kê. Riêng tiêu đề Fakeddit chỉ được dùng tính độ dài, không ghi lại
  nội dung trong kết quả kiểm tra.
- Với processed, đọc tối đa 3.000 dòng metadata mỗi tệp ngay từ đầu bằng
  `usecols` và `nrows`; không đọc cột text/plot. Kiểm tra JSON–ảnh MM-IMDb dựa
  trên tên tệp và danh sách ID split, không duyệt nội dung toàn bộ JSON.
- Không tải thêm ảnh, không giải mã ảnh, không chạy model, không ghi đè dữ liệu.
  Tệp tồn tại không bảo đảm ảnh đọc được. Số lượng nguồn đủ lớn chưa bảo đảm
  đủ số mẫu sau mọi bước kiểm tra chất lượng.
- Số 1.973 của Fakeddit tính từ **tiêu đề raw của cùng các ID**, không phải một
  lượt inference mới hay phép kiểm chứng toàn bộ nội dung tiêu đề processed.

Số liệu đối chiếu: [ra_soat_raw_preprocessing_3000.json](ra_soat_raw_preprocessing_3000.json).
Kiểm tra metadata processed trước đó:
[kiem_tra_cohort_3000.md](kiem_tra_cohort_3000.md).

## 1. Fakeddit

### 1.1. Dữ liệu có sẵn

| Nguồn | Số dòng raw | ID của cohort hiện tại có trong nguồn |
|---|---:|---:|
| `multimodal_validate.tsv` | 59.342 | 2.841/2.841 |
| `multimodal_test_public.tsv` | 59.319 | 0/2.841 |

Cohort hiện tại thuộc validation theo đối chiếu ID. Thư mục ảnh processed có
2.841 tệp JPG, không có kho ảnh Fakeddit tương ứng ở `data/raw/fakeddit/images`
tại thời điểm rà soát. Đối chiếu tiêu đề raw với `len(title.replace(" ", "")) > 20`
cho kết quả:

| 6-way ID | Nhãn | Có ảnh local và đạt lọc hiện tại của preprocessor | Có ảnh local và đạt lọc độ dài runner, theo tiêu đề raw |
|---|---|---:|---:|
| 0 | True (Thật) | 1.400 | 1.198 |
| 1 | Satire/Parody (Châm biếm/nhại) | 134 | 101 |
| 2 | False Connection (Liên kết sai) | 430 | 297 |
| 3 | Imposter Content (Nội dung mạo danh) | 54 | 53 |
| 4 | Manipulated Content (Nội dung bị chỉnh sửa) | 744 | 249 |
| 5 | Misleading Content (Nội dung gây hiểu lầm) | 79 | 75 |
| | **Tổng** | **2.841** | **1.973** |

Chênh lệch là 868 mẫu. Theo tiêu đề raw, tập qua lọc có 1.198 True và 775 Fake,
khác phân bố gần cân bằng nhị phân của 2.841 dòng trước lọc.

Nguồn validation còn 39.005 dòng thỏa nhãn, URL, `hasImage` và độ dài runner,
**chưa kiểm tra ảnh tải được hay đọc được**. Vì vậy, thiếu mẫu hiện tại là thiếu
kho ảnh hợp lệ tại local, không phải nguồn raw chỉ có 2.841 dòng. Nếu giữ toàn
bộ 1.973 ứng viên hiện có, còn thiếu ít nhất 1.027 mẫu để đạt 3.000; chọn lại
theo phân bố nguồn có thể cần thay nhiều hơn và bổ sung nhiều ảnh hơn.

### 1.2. Vấn đề trong mã

1. `image_downloader.py` ghi `data/raw/fakeddit/images`, trong khi
   `preprocess_fakeddit.py` chỉ tìm `data/processed/pro_fakeddit/images`.
   Chạy bộ tải rồi chạy tiền xử lý không tự chuyển ảnh giữa hai nơi.
2. Preprocessor chỉ yêu cầu tiêu đề không rỗng và dài ít nhất 3 ký tự;
   `src/run/common.py` lọc sau bỏ dấu cách, dùng phép so sánh `>20` trong cấu
   hình hiện tại. Chọn 3.000 trước bộ lọc khác này không bảo đảm chạy đủ 3.000.
3. Công thức lấy từng lớp là `min(max(200, 3000 // n_classes), count)`.
   Không phân bổ lại phần thiếu, không kiểm tra tổng cuối bằng 3.000. Với sáu
   lớp và pool local hiện tại, tổng quota chỉ là **1.697 trước lọc runner**.
   Tăng `MIN_PER_CLASS` không tạo thêm dữ liệu cho lớp chỉ có 54 ảnh.
4. Đổi task 2way/6way thay cách chọn mẫu nhưng mặc định ghi cùng một đường dẫn
   đầu ra. Chạy lại có thể thay cohort của các phép so sánh trước đó.
5. Kiểm tra sẵn sàng dùng tỷ số lớp `>3` như điều kiện thất bại. Đây là quy tắc
   của script, không phải bằng chứng khoa học bắt buộc mọi tập test phải cân bằng.

### 1.3. Cách chọn lại

Chọn **một split nguồn** trước. Validation dành cho khảo sát; muốn báo cáo test
thì dùng `multimodal_test_public.tsv` riêng và ghi rõ đây là public test.
Không ghép validation vào test để bù ảnh thiếu.

Đồng bộ điều kiện lọc với runner, kiểm tra ID/nhãn và xác thực ảnh trước khi
chốt cohort. Tải hoặc chuẩn bị ảnh theo danh sách ứng viên đã xáo trộn bằng
seed cố định; ảnh lỗi được ghi lý do và thay bằng ứng viên khác cùng tầng nếu
còn đủ. Việc tải thất bại theo lớp cũng phải được thống kê.

Cohort chính nên giữ gần tỷ lệ sáu nhãn của pool hợp lệ trong split được chọn.
Một cohort cố định có thể phục vụ cả 2way và 6way, nhưng không thể đồng thời
cân bằng hoàn toàn cả hai: chia sáu lớp đều 500 sẽ cho 2.500 Fake và 500 True.
Nếu cần một bộ chẩn đoán cân bằng sáu lớp, lưu thành cohort phụ và báo rõ phân
bố đã thay đổi. Không nhân bản mẫu để tăng support đánh giá.

## 2. CrisisMMD

### 2.1. Split và nguồn nhãn thực tế

Tệp chính thức trong `crisismmd_datasplit_all/` có:

| Task | Train | Dev | Test | Sự kiện trong mỗi split |
|---|---:|---:|---:|---:|
| Informativeness | 13.608 | 2.237 | 2.237 | 7 |
| Humanitarian | 13.608 | 2.237 | 2.237 | 7 |

Trong **từng task**, không có giao nhau tweet ID hoặc image ID giữa ba split.
Toàn bộ đường dẫn ảnh trong sáu bảng có tệp raw tương ứng, chưa kiểm tra giải
mã. Hai task không được giả định có cùng danh sách train/dev/test chỉ vì có
cùng kích thước và cùng tổng pool; cần giữ `source_split` riêng theo task.

Đối chiếu 3.000 dòng đầu processed bằng **image_id chính xác**:

| Task | Thuộc train nguồn | Thuộc dev nguồn | Thuộc test nguồn | Khớp `label_image` |
|---|---:|---:|---:|---:|
| Informativeness | 2.273 | 386 | 341 | 3.000/3.000 |
| Humanitarian | 2.260 | 346 | 394 | 3.000/3.000 |

Cohort hiện tại gồm Harvey 1.745, Irma 785 và California 470 cặp. Cả 3.000 nhãn
đang dùng khớp nhãn ảnh chính thức, nhưng chỉ có 16 cặp/task khác nhau giữa
nhãn text và ảnh. Để đối chiếu, ở **test toàn bộ** tỷ lệ bất đồng là 703/2.237
cho informativeness và 1.282/2.237 cho humanitarian. Như vậy, không thể chỉ
xáo trộn 3.000 dòng hiện tại rồi coi là đại diện của toàn bộ dữ liệu.
Chưa xác định được script đã tạo ra mức chọn lọc này; không quy kết nguyên
nhân chỉ từ phân bố quan sát được.

README đi kèm phân biệt `label`, `label_text`, `label_image`; cột `label` cho hai
task được chọn ngẫu nhiên từ nhãn text và ảnh tại bước tạo dữ liệu nguồn.
**Không tự đổi cột đích sang `label` mà vẫn coi là cùng bài toán với kết quả cũ.**
Để duy trì ý nghĩa tác vụ hiện tại, đề nghị dùng rõ `label_source=image` ở vòng
tiếp theo; lưu thêm nhãn text và cờ đồng thuận để phân tích. Đây là lựa chọn
giao thức đánh giá được đề nghị, không phải khẳng định nó là giao thức duy nhất.

### 2.2. ID và script tiền xử lý

- 470/3.000 `tweet_id` processed không phải chuỗi chữ số, không bằng phần
  tweet ID của `image_id`; đã quan sát dạng khoa học trong annotation raw.
  Cách đọc `dtype=str` ngăn làm hỏng thêm nhưng không khôi phục phần chữ số
  đã mất trong chính tệp nguồn đó.
- Số ID lưu hiện tại là 2.679; phần tweet ID chính xác của `image_id` cho
  2.730 ID. Dùng ID lưu sai để nhóm tweet có thể gộp nhầm tweet khác nhau.
- Bảng split chính thức có tweet ID nguyên vẹn, toàn bộ khớp prefix `image_id`.
  Cần dùng bảng này làm nguồn định danh hoặc khôi phục bằng join `image_id`
  có kiểm tra; không chuyển số khoa học qua float rồi coi là ID gốc.
- `preprocess_crisismmd.py` chỉ kiểm tra TSV processed và có tùy chọn copy ảnh;
  không tạo lại dữ liệu từ raw. Regex `^\S+$` chấp nhận cả ID sai dạng khoa học.
  Vì vậy, chạy lại script này không sửa split, ID hoặc cách chọn nhãn.

### 2.3. Cách chọn lại

Đề nghị đánh giá **toàn bộ test chính thức 2.237 cặp cho mỗi task**, sau kiểm
tra tính hợp lệ. Không ghép dev/test hoặc lặp mẫu để ghi là 3.000 test. Nếu
bài tập bắt buộc một phép thử 3.000, có thể tạo cohort phát triển riêng từ
train, ghi đúng nguồn và giữ test nguyên vẹn; kết quả đó vẫn là thăm dò.

Nguồn nhãn ảnh ở test humanitarian có support:

| Nhãn | Support |
|---|---:|
| Affected Individuals (Người bị ảnh hưởng) | 88 |
| Infrastructure and Utility Damage (Thiệt hại hạ tầng và tiện ích) | 458 |
| Injured or Dead People (Người bị thương hoặc tử vong) | 10 |
| Missing or Found People (Người mất tích hoặc được tìm thấy) | 3 |
| Not Humanitarian (Không thuộc nội dung nhân đạo) | 1.055 |
| Other Relevant Information (Thông tin liên quan khác) | 331 |
| Rescue, Volunteering or Donation Effort (Cứu hộ, tình nguyện hoặc quyên góp) | 256 |
| Vehicle Damage (Thiệt hại phương tiện) | 36 |

Mất cân bằng lớp hiếm vẫn tồn tại trong test nguồn. Giữ phân bố đó cho bảng
chính, báo Macro-F1, F1/Recall/support từng lớp và độ bất định; không kết luận
vững chắc về lớp chỉ có ba mẫu. Bộ chẩn đoán lớp hiếm có thể lấy riêng từ
phần phát triển còn lại, không trộn vào điểm test.

Không tự gộp tám lớp thành năm: bản benchmark đồng thuận của tác giả có bước
gộp riêng và là giao thức khác với bài toán tám lớp hiện tại.

## 3. MM-IMDb

### 3.1. Nguồn và lỗi chọn tập

`data/raw/mmimdb/split.json` quy định train 15.552, dev 2.608 và test 7.799 ID;
ba danh sách không giao nhau. Đối chiếu tên tệp có đủ 25.959 cặp JSON–ảnh,
bao gồm 7.799 cặp trong test. Đây là kiểm tra sự tồn tại, chưa phải kiểm tra
toàn bộ plot/genre/khả năng đọc ảnh.

`preprocess_mmimdb.py` quét toàn bộ pool, lọc `kind`, plot, genre và ảnh rồi
lấy ngẫu nhiên 3.000 phim. Script **không đọc `split.json`**. Do đó, cohort hiện
tại gồm 1.819 train + 284 dev + 897 test. Nhận định trước đây rằng MM-IMDb có
thể dùng 3.000 mẫu chỉ đúng về số dòng cho thăm dò; đánh giá theo split chính
thức cần chọn lại.

Hai điểm cần bảo toàn khi chỉnh script:

- Giữ 23 nhãn, gồm **Short**. Comment gợi ý loại Short không phù hợp nếu vẫn
  gọi đây là bài toán 23 nhãn hiện tại.
- Kiểm tra kiểu dữ liệu trước khi ép `str`: giá trị JSON `null` có thể thành
  chuỗi `"None"` và vượt ngưỡng ba ký tự. Chuẩn hoá danh sách genre bằng tập
  nhãn hợp lệ, loại nhãn lặp trong cùng phim nhưng không làm mất đồng xuất hiện.

### 3.2. Cách chọn lại

Giới hạn pool bằng ID trong **test** trước; xác nhận plot, 23 genre và ảnh hợp
lệ; sau đó lấy 3.000 phim không hoàn lại, seed cố định. Nếu số hợp lệ chưa đủ,
báo số thực có thay vì mượn phim từ train/dev.

Vì đây là nhiều nhãn, dùng phân tầng nhiều nhãn để giữ gần tỷ lệ genre; kiểm
tra thêm số nhãn/phim và các cặp genre đồng xuất hiện. Nếu dùng chọn ngẫu
nhiên đơn giản, vẫn phải công bố độ lệch tỷ lệ nhãn so với pool hợp lệ. Không
lấy quota độc lập cho từng genre rồi ghép, vì một phim có thể thuộc nhiều
genre và bị chọn lặp.

Không cần ép Drama và Film-Noir có số phim bằng nhau. Sự chênh lệch tự nhiên
của genre không phải lỗi chuẩn hoá ảnh hay lý do tự động huấn luyện lại model.

## 4. Quy trình chốt cohort cho zero-shot

1. Chọn split và nguồn nhãn từ trước; không chọn lại mẫu theo model đúng/sai.
2. Đọc ID dạng chuỗi, bảo toàn số 0 đầu của ID phim; kiểm tra khóa mẫu và khóa
   nhóm tweet. Nhiều ảnh khác nhau của cùng tweet là các cặp hợp lệ, không tự
   xóa tất cả vì tweet ID lặp.
3. Áp dụng một điều kiện text nhất quán với runner; giữ Unicode và nội dung
   ngữ nghĩa cần thiết. Không bắt buộc bỏ toàn bộ ký tự ngoài ASCII, dịch text
   sang tiếng Việt hoặc loại nội dung hashtag chỉ để làm dữ liệu trông sạch.
4. Kiểm tra ảnh trước chốt danh sách; ghi số lượng loại theo lý do và lớp/sự
   kiện. Dùng bộ tiền xử lý ảnh của từng backbone ở inference; không cần
   chuẩn hoá pixel thủ công giống nhau cho mọi backbone.
5. Chọn không hoàn lại: Fakeddit theo tỷ lệ nhãn nguồn, MM-IMDb theo nhiều
   nhãn, CrisisMMD dùng toàn bộ test/task. Chốt seed và thứ tự ID.
6. Lưu TSV mới cùng manifest: dataset, task, nguồn/split, `label_source`, seed,
   quy tắc lọc, số mẫu trước/sau lọc, ID, phiên bản và checksum. Không ghi đè
   các TSV đã dùng tạo kết quả cũ. Giữ nguyên tên nhãn máy đọc bằng tiếng Anh;
   phần hiển thị/báo cáo dùng English (Tiếng Việt).
7. Mọi backbone và baseline/DCLIP dùng cùng cohort trong từng phép so sánh.
   Bật cơ chế sample lock phù hợp sau khi chốt. Báo số mẫu **thực sự suy luận**,
   không chỉ `max_samples`. Nếu ảnh lỗi làm giảm N, xử lý và chốt lại chung
   trước khi so sánh, không bù khác nhau theo model.
8. Không dùng oversampling/SMOTE cho tập đánh giá. Giữ prompt, trọng số fusion
   và ngưỡng cố định cho test; điều chỉnh bằng nhãn trên dev phải được ghi rõ
   là có hiệu chỉnh có giám sát, dù không cập nhật trọng số backbone.

`configs/launcher_3000.yaml` hiện chỉ tăng giới hạn đọc trên cohort cũ; **không
thực hiện các bước chuẩn bị trên**. Chưa nên dùng nó để tạo bảng benchmark
test chính thức. Lần rà soát này chỉ tạo hồ sơ kiểm tra, chưa sửa script tiền
xử lý, chưa tạo lại sample hoặc chạy inference.

## Tài liệu đối chiếu

- Mã nguồn trong dự án: `src/preprocessing/data_fakeddit/preprocess_fakeddit.py`,
  `image_downloader.py`, `src/preprocessing/data_mmimdb/preprocess_mmimdb.py`,
  `src/preprocessing/data_crisismmd/preprocess_crisismmd.py`, `src/run/common.py`.
- README raw tại `data/raw/CrisisMMD_v2.0/crisismmd_datasplit_all/Readme.txt`:
  định nghĩa cột nhãn, v2.0 và nguồn ID.
- [Fakeddit, kho chính thức](https://github.com/entitize/Fakeddit): dữ liệu
  multimodal, clean_title, public test và các phương án lấy ảnh.
- [CrisisMMD, trang chính thức](https://crisisnlp.qcri.org/crisismmd): nhãn theo
  modality, bản v2.0 và sự khác nhau giữa split toàn bộ/đồng thuận.
- [Arevalo và cộng sự, 2017](https://arxiv.org/abs/1702.01992): MM-IMDb là bài
  toán dự đoán genre nhiều nhãn từ plot và poster.
- [Sechidis và cộng sự, 2011](https://doi.org/10.1007/978-3-642-23808-6_10):
  phương pháp phân tầng cho dữ liệu nhiều nhãn. Áp dụng ở đây là đề nghị thiết
  kế cohort, chưa phải kết quả của một lần chọn mẫu lại đã thực thi.
