# ZeroShot_Classification_Multimodal

# Data Processing Plan

Kế hoạch và quy trình xử lý dữ liệu cho dự án Zero-Shot Multi-Modal Content Classification. Mỗi dataset có một mục riêng, ghi lại các bước đã làm, các lỗi phát hiện được, và cách xử lý.

---

## 1. Fakeddit 

### 1.1. Nguồn dữ liệu

- Bản dùng: `multimodal_only_samples` từ Google Drive chính thức của tác giả Fakeddit (`entitize`/Kai Nakamura).
- Lý do chọn bản này thay vì `all_samples`: paper gốc chỉ tính baseline trên các mẫu có cả ảnh và text, nên dùng đúng bản `multimodal_only_samples` để kết quả so sánh được với literature.

### 1.2. Vị trí lưu trữ

```
data/raw/fakeddit/
├── multimodal_train.tsv
├── multimodal_validate.tsv
├── multimodal_test_public.tsv
└── images/            # để trống, dành cho ảnh tải sau nếu cần
```

### 1.3. Cột giữ lại và mô tả

Dataset gốc có 16 cột. Chỉ giữ 6 cột sau, đúng theo guideline chính thức của Fakeddit (dùng `clean_title` thay cho `title` thô, bỏ các cột `Unnamed...`):

| Cột | Mô tả | Kiểu | Giá trị |
|---|---|---|---|
| `id` | Mã bài đăng Reddit, dùng để tra cứu và loại trùng, không đưa vào model | string | chuỗi ký tự-số, ví dụ `7reuo3` |
| `clean_title` | Tiêu đề đã làm sạch, dùng làm input text cho model | string | văn bản tự do |
| `image_url` | URL ảnh đính kèm | string (URL) | bắt đầu bằng `http://` hoặc `https://` |
| `hasImage` | Bài có ảnh hay không | boolean | `True` / `False` |
| `2_way_label` | Nhãn nhị phân | int | `0` = fake, `1` = real |
| `6_way_label` | Nhãn chi tiết 6 loại | int | `0` True · `1` Satire/Parody · `2` False Connection · `3` Imposter Content · `4` Manipulated Content · `5` Misleading Content |

**Cột bỏ qua:** `title` (dùng `clean_title` thay thế), `author`, `created_utc`, `domain`, `subreddit` (metadata phụ, không liên quan nội dung), `num_comments`, `score`, `upvote_ratio` (tín hiệu tương tác xã hội, không phải đặc trưng nội dung), `linked_submission_id`, `3_way_label` (dư thừa khi đã có 2-way và 6-way).

### 1.4. Kiểm tra dữ liệu gốc (`check_fakeddit.py`)

Chạy trên file `multimodal_train.tsv` (564,000 dòng), kiểm tra:

- **Null/missing** theo từng cột bắt buộc.
- **Trùng lặp** theo `id` và trùng lặp toàn dòng.
- **Text quá ngắn** (`clean_title` dưới 3 ký tự) — phát hiện 1,049 dòng.
- **Tính hợp lệ của nhãn** — `2_way_label` phải thuộc {0,1}, `6_way_label` phải thuộc {0..5}.
- **Mismatch `hasImage` vs `image_url`** — phát hiện 1,534 dòng có `hasImage=True` nhưng `image_url` rỗng. Đây là lỗi cần loại bỏ trước khi dùng cho bài toán multimodal.
- **Định dạng URL** — kiểm tra URL có bắt đầu đúng `http(s)://`.

Kết quả: 562,466 / 564,000 dòng đạt chuẩn (99.73%) sau khi loại các trường hợp mismatch.

### 1.6. Sampling — subset cân bằng cho zero-shot

Vì zero-shot không cần dataset lớn để đánh giá có ý nghĩa, giảm từ 562,466 dòng xuống subset nhỏ hơn:

- **Tổng số mẫu mục tiêu:** 3,000
- **Số mẫu tối thiểu mỗi lớp:** 200 (đảm bảo lớp hiếm như Imposter Content vẫn đủ mẫu để đánh giá, thay vì random sample thuần túy sẽ khiến lớp hiếm chỉ còn vài chục dòng)
- **Phương pháp:** stratified sampling theo `6_way_label`, `random_state=42` để tái lập được kết quả.

### 1.7. Kiểm tra lại bản sample (`check_fakeddit_sample.py`)

Sau khi sample, chạy lại toàn bộ kiểm tra ở mục 1.4 trên subset mới, cộng thêm:

- **Phân bố nhãn sau sample** — xác nhận tỷ lệ lớp nhiều nhất/ít nhất không lệch quá 3 lần.
- **Xem mẫu thực tế theo từng lớp** — đọc bằng mắt vài dòng mỗi lớp để xác nhận nội dung hợp lý, không chỉ dựa vào số liệu.
- **Kiểm tra ảnh còn tải được không** — thử tải thật 10 URL ngẫu nhiên bằng `requests` + `PIL`, vì URL đúng định dạng không đồng nghĩa ảnh còn tồn tại (ảnh Reddit cũ hay bị xóa). Nếu tỷ lệ lỗi trên mẫu thử vượt 30%, cần tăng cỡ subset ban đầu để bù trừ phần ảnh chết.

### 1.8. File đầu ra

```
data/processed/pro_fakeddit
├── fakeddit_clean.tsv    # toàn bộ dữ liệu đã làm sạch (~562K dòng)
└── fakeddit_sample.tsv    # subset 3,000 dòng dùng cho zero-shot
```

### 1.9. Việc còn lại

- [ ] Tải ảnh thật cho 3,000 dòng trong `fakeddit_sample.tsv` (dùng `image_downloader.py` từ repo gốc hoặc script tương tự), lưu vào `data/raw/fakeddit/images/`.
- [ ] Loại bỏ các dòng có ảnh không tải được sau bước tải thật.
- [ ] Viết `fakeddit_loader.py` theo class `BaseZeroShotDataset` đã thiết kế.

---

# 2. CrisisMMD 

## 2.1. Mục tiêu

Xây dựng một bộ dữ liệu đánh giá (evaluation set) sạch, đáng tin cậy từ CrisisMMD v2.0, phục vụ cho việc đánh giá các mô hình **zero-shot multimodal classification** (CLIP, BLIP-2, LLM có vision như GPT-4V/Claude...) trên bài toán phân loại **humanitarian categories** (hoặc informative), sử dụng đồng thời **text (tweet)** và **image**.

> Lưu ý: Zero-shot không cần tập train — dữ liệu xử lý ở đây chỉ dùng để **đánh giá (test/eval)**, không dùng để fine-tune.

---

## 2.2. Nguồn dữ liệu

- **Bộ dữ liệu:** CrisisMMD v2.0
- **Nguồn gốc:** 7 thảm họa thiên nhiên năm 2017 (động đất, bão, cháy rừng, lũ lụt), thu thập từ Twitter
- **File sử dụng:** 6 file annotation (TSV), mỗi file tương ứng 1 sự kiện:
  - `california_wildfires_final_data.tsv`
  - `hurricane_harvey_final_data.tsv`
  - `hurricane_irma_final_data.tsv`
  - `iraq_iran_earthquake_final_data.tsv`
  - `mexico_earthquake_final_data.tsv`
  - `srilanka_floods_final_data.tsv`
- **Vị trí lưu trữ raw:** `data/raw/CrisisMMD_v2.0/annotations/`
- **Ảnh gốc:** `data/raw/CrisisMMD_v2.0/data_image/`

---

## 2.3. Nhiệm vụ phân loại (Task) được chọn

| Task | Nhãn | Lý do chọn |
|---|---|---|
| **Task 2 – Humanitarian categories** (mặc định) | 8 lớp: affected_individuals, infrastructure_and_utility_damage, injured_or_dead_people, missing_or_found_people, rescue_volunteering_or_donation_effort, vehicle_damage, other_relevant_information, not_humanitarian | Nhiều lớp hơn, phù hợp để đánh giá khả năng phân biệt chi tiết của model zero-shot |
| Task 1 – Informative | 2 lớp: informative / not_informative | Dự phòng, dùng nếu cần bài toán binary đơn giản hơn |

---

## 2.4. Quy trình xử lý dữ liệu (Data Processing Pipeline)

```
Raw TSV (6 events)
      │
      ▼
[Bước 1] Đọc & chọn cột cần thiết
      │
      ▼
[Bước 2] Lọc theo Confidence Score
      │
      ▼
[Bước 3] Lọc theo sự đồng thuận nhãn Text–Image (label agreement)
      │
      ▼
[Bước 4] Làm sạch văn bản (text cleaning)
      │
      ▼
[Bước 5] Loại bỏ trùng lặp
      │
      ▼
[Bước 6] Kiểm tra ảnh tồn tại trên đĩa
      │
      ▼
[Bước 7] Gộp 6 sự kiện thành 1 file dataset duy nhất
      │
      ▼
[Bước 8] Kiểm tra chất lượng cuối (QA check)
      │
      ▼
Output: crisismmd_multimodal.tsv
```

### Bước 1: Chọn cột cần thiết

Từ mỗi file TSV gốc (15 cột), chỉ giữ lại các cột phục vụ trực tiếp cho zero-shot multimodal:

| Cột giữ lại | Mục đích |
|---|---|
| `tweet_id`, `image_id` | Định danh, tra cứu |
| `tweet_text` | Input văn bản gốc (trước khi làm sạch) |
| `image_path` | Đường dẫn ảnh local để load vào model |
| `image_url` | Link ảnh gốc, dùng để kiểm tra thủ công/online |
| `text_human`, `image_human` (hoặc `text_info`, `image_info`) | Nhãn gốc theo từng modal |
| `text_human_conf`, `image_human_conf` | Độ tin cậy nhãn |
| `event` | Tên sự kiện thảm họa (lấy từ tên file) |

Loại bỏ: `image_damage`, `image_damage_conf` (Task 3, không dùng trong dự án này).

### Bước 2: Lọc theo Confidence Score

- Chỉ giữ dòng có **cả** confidence của nhãn text và nhãn ảnh **≥ ngưỡng (mặc định 0.7)**
- Mục đích: loại bỏ các nhãn có độ chắc chắn thấp từ người gán nhãn (crowdsourcing qua Figure Eight), tránh đánh giá model dựa trên ground-truth không đáng tin

### Bước 3: Lọc theo sự đồng thuận nhãn Text–Image

- CrisisMMD gán nhãn **riêng biệt** cho text và ảnh, có thể không trùng nhau
- Với bài toán multimodal (dùng cả 2 input để dự đoán 1 output), chỉ giữ lại các dòng mà `text_human == image_human`
- Nhãn đồng thuận này trở thành cột `label` duy nhất — vừa đảm bảo chất lượng, vừa tạo ground-truth thống nhất cho cả 2 modal

### Bước 4: Làm sạch văn bản (Text Cleaning)

Áp dụng cho cột `tweet_text` → sinh ra `clean_text`:

1. Xóa tiền tố `RT @user:`
2. Xóa toàn bộ `@mention`
3. Xóa URL (link rút gọn t.co, http/https)
4. Xóa ký tự `#` nhưng giữ từ hashtag
5. Sửa lỗi encoding (mojibake), kể cả lỗi encoding **lặp 2 lần**:
   - Áp bảng ánh xạ các mẫu lỗi đã biết
   - Dùng thư viện `ftfy` để tổng quát hóa việc sửa lỗi
   - Thử round-trip encode/decode (cp1252 / latin1 / mac_roman) nếu vẫn còn lỗi
6. **Loại bỏ toàn bộ ký tự non-ASCII còn sót lại** (catch-all cuối cùng) — vì dữ liệu là tweet tiếng Anh, đảm bảo văn bản đầu ra sạch hoàn toàn dù không nhận diện được nguồn gốc lỗi
7. Gộp khoảng trắng thừa

### Bước 5: Loại bỏ trùng lặp

- Drop các dòng trùng theo cặp (`tweet_id`, `image_id`)

### Bước 6: Kiểm tra ảnh tồn tại trên đĩa

- Với mỗi `image_path`, kiểm tra file ảnh có thực sự tồn tại trong `data_image/` hay không
- Loại bỏ các dòng thiếu ảnh (tránh lỗi khi load ảnh lúc chạy model)

### Bước 7: Gộp thành 1 dataset duy nhất

- Gộp dữ liệu đã xử lý của cả 6 sự kiện thành **1 file duy nhất**
- Cột `event` giữ lại tên sự kiện tương ứng để phân tích theo từng thảm họa nếu cần

### Bước 8: Kiểm tra chất lượng (QA check)

- Kiểm tra `clean_text` không còn ký tự non-ASCII (đảm bảo làm sạch thành công)
- In thống kê:
  - Số dòng còn lại sau mỗi bước lọc
  - Phân bố nhãn (`label`) — kiểm tra mất cân bằng lớp
  - Số dòng theo từng sự kiện (`event`)

---

## 2.5. Output cuối cùng

**File:** `data/process/pro_CrisisMMD/crisismmd_multimodal.tsv`

**Cấu trúc cột:**

| Cột | Mô tả |
|---|---|
| `tweet_id` | ID tweet gốc |
| `image_id` | ID ảnh (tweet_id + index) |
| `clean_text` | Văn bản đã làm sạch, sẵn sàng đưa vào text encoder |
| `image_path` | Đường dẫn ảnh local |
| `image_url` | Link ảnh gốc (kiểm tra thủ công) |
| `label` | Nhãn thống nhất (text–image đồng thuận) |
| `event` | Tên sự kiện thảm họa |

---

## 2.6. Bước tiếp theo sau khi có dữ liệu sạch (định hướng)

1. **Chọn model zero-shot**: CLIP (ViT-B/32, ViT-L/14), BLIP-2, hoặc LLM đa phương thức (GPT-4V, Claude, Gemini)
2. **Thiết kế prompt / candidate labels**: chuyển 8 nhãn humanitarian thành câu mô tả tự nhiên (ví dụ: `"a photo related to rescue, volunteering, or donation efforts"`)
3. **Chạy inference**: đưa `clean_text` + ảnh tại `image_path` vào model, lấy nhãn dự đoán
4. **Đánh giá**: so sánh dự đoán với `label`, tính Accuracy, F1-score (macro/weighted) theo từng lớp và từng sự kiện
5. **Phân tích lỗi**: xem model nhầm lẫn ở nhóm nhãn nào, sự kiện nào nhiều nhất
6. **(Tùy chọn) So sánh** hiệu năng zero-shot khi dùng: chỉ text / chỉ ảnh / kết hợp cả hai

---

## 2.7. Cấu hình có thể điều chỉnh (config trong script xử lý)

| Config | Giá trị mặc định | Ghi chú |
|---|---|---|
| `TASK` | `"humanitarian"` | Đổi thành `"informative"` nếu cần Task 1 |
| `CONF_THRESHOLD` | `0.7` | Ngưỡng lọc confidence, có thể tăng để lấy dữ liệu "sạch" hơn nhưng ít mẫu hơn |
| `CHECK_IMAGE_EXISTS` | `True` | Tắt nếu chưa tải đủ ảnh về máy |
| `STRIP_NON_ASCII` | `True` | Tắt nếu cần giữ lại ký tự có dấu hợp lệ (chấp nhận rủi ro còn sót rác) |
