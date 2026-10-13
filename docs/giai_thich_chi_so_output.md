# Giải thích các chỉ số và file output của dự án

Đối chiếu code ngày **13-10-2026**. Tài liệu mô tả output thực sự được các runner hiện tại tạo ra, gồm pipeline chính, demo, upstream/adapted và đối chứng ảnh–text. Mọi ví dụ số dưới đây là **minh họa**, không phải kết quả chạy trên dataset của bạn.

## 1. Đọc gì trước khi xem điểm?

Đọc `dataset`, `task`, `model` và `manifest.json` để biết kết quả thuộc bài toán nào. Sau đó đọc metric tổng, metric từng nhánh và cuối cùng các mẫu sai.

| Dataset/task | Một mẫu được mang bao nhiêu nhãn? | Nên đọc chỉ số nào trước? |
|---|---|---|
| Fakeddit `2way` | Một trong 2 nhãn | `macro_f1`, `balanced_accuracy`, từng lớp; thêm `auroc` nếu có |
| Fakeddit `6way` | Một trong 6 nhãn | `macro_f1`, `balanced_accuracy`, precision/recall từng lớp |
| CrisisMMD `informativeness` | Một trong 2 nhãn | Như binary ở trên |
| CrisisMMD `humanitarian` | Một trong 8 nhãn | Như multiclass ở trên |
| MM-IMDb `genres` | Nhiều thể loại trong 23 genre có thể cùng đúng | `macro_f1`, `micro_f1`, `samples_f1`, nhãn thiếu/thừa |

**Ba nhóm thông tin khác nhau:**

- **Metric đánh giá:** cần nhãn thật, cho biết model dự đoán tốt đến đâu trên tập mẫu được đánh giá.
- **Điểm suy luận/chẩn đoán:** similarity, confidence, margin, entropy, JS… cho biết model chấm điểm hoặc phân vân thế nào; tự chúng không chứng minh đúng/sai.
- **Metadata:** cấu hình, seed, số mẫu, hash… cho biết điều kiện sinh ra kết quả, giúp so sánh và tái lập.

Số như `0.80` ở accuracy/F1 tương ứng `80%` khi biểu diễn theo phần trăm. Không đổi mọi con số thành phần trăm: cosine, z-score, entropy, token count và hash có ý nghĩa khác.

## 2. Mỗi file output dùng để làm gì?

Đường dẫn có thể đổi theo YAML; cấu trúc thông thường:

```text
results/
├── summary_table_<dataset>_<task>_<số>.tsv
└── raw_predictions/<run_id>/
    ├── raw_predictions.tsv
    ├── error_analysis.tsv
    ├── manifest.json
    ├── diagnostics.tsv
    ├── text_lengths.tsv
    └── evaluation_snapshot.json
```

| File | Đơn vị mỗi dòng / nội dung | Dùng để làm gì? |
|---|---|---|
| `summary_table_*.tsv` | Một lượt model × task × lựa chọn prompt | So sánh kết quả tổng của các lượt chạy |
| `raw_predictions.tsv` | Một mẫu hợp lệ đã được suy luận | Tra nhãn thật/dự đoán, điểm lớp, nhánh và các cờ |
| `error_analysis.tsv` | Một mẫu sai hoặc không khớp đủ genre | Tìm lỗi để phân tích prompt, fusion và dữ liệu |
| `manifest.json` | Thông tin một lượt chạy | Xác định model, cấu hình hiệu lực và số lỗi |
| `diagnostics.tsv` | Một nhánh: `image`, `text`, `fusion` | Xem metric và phân bố điểm của từng nhánh |
| `text_lengths.tsv` | Một mẫu | Kiểm tra token, nguy cơ cắt text và chunking |
| `evaluation_snapshot.json` | Snapshot cấu hình, prompt, cohort, code | Đối chiếu điều kiện chạy và truy nguồn kết quả |
| Research `summary.tsv` | Một trial | So sánh phương pháp/seed/output variant |
| Research `seed_summary.tsv` | Một nhóm các lượt cùng điều kiện | Đọc trung bình và dao động qua seed |
| Relation ablation `comparison.tsv` | Một mẫu × một phương pháp | Xem điểm và nhãn trước/sau công thức relation |
| `frozen_plan.json` | Kế hoạch trước inference | Kiểm tra prompt, tham số và nguồn đã chốt |
| `failure.json`, nếu có | Thông tin lỗi chạy | Biết lượt chạy đã dừng và có thể chưa đủ trial/seed |
| Demo `demo_*.png` | Ảnh kèm text, nhãn thật và dự đoán | Quan sát trực quan một nhóm mẫu nhỏ |

Ba file audit `diagnostics.tsv`, `text_lengths.tsv`, `evaluation_snapshot.json` chỉ có khi audit được bật. Summary rút gọn không giữ mọi trường: cần mở manifest/snapshot nếu không thấy một cột. Main MM-IMDb dùng task `genres` trong summary nhưng logger đặt tên task `multi_label`.

## 3. Khái niệm nền: TP, FP, FN, TN và support

Với một lớp đang xét, coi lớp đó là positive và các lớp còn lại là negative:

| Ký hiệu | Ý nghĩa | Ví dụ đang xét nhãn fake |
|---|---|---|
| `TP` | Đúng là positive và dự đoán positive | Tin fake được đoán fake |
| `FP` | Không phải positive nhưng dự đoán positive | Tin real bị đoán fake |
| `FN` | Đúng là positive nhưng bỏ sót | Tin fake bị đoán real |
| `TN` | Không phải positive và dự đoán negative | Tin real được đoán real |
| `support` | Số mẫu thật thuộc lớp, bằng `TP + FN` | Có bao nhiêu tin fake trong tập đang đánh giá |

Trong multiclass, tính riêng cho mỗi lớp theo one-vs-rest. Trong multi-label, tính riêng cho mỗi genre theo có/không có genre đó. TP/FP/FN/TN là ký hiệu giải thích công thức, không phải các cột đếm riêng được mọi runner xuất sẵn.

## 4. Chỉ số chất lượng dự đoán

### 4.1 Accuracy — tỷ lệ mẫu dự đoán đúng

$$
\operatorname{Accuracy} = \frac{\text{số mẫu có }y_{pred} = y_{true}}{N}
$$

`accuracy` nằm trong `[0,1]`, càng cao càng tốt trên cùng bộ đánh giá. Có trong kết quả single-label. Nó trả lời: **Trong toàn bộ mẫu, model chọn đúng nhãn bao nhiêu lần?**

Điểm này có thể bị lớp đông chi phối. Ví dụ 99 mẫu A và 1 mẫu B, luôn đoán A sẽ đạt accuracy `0.99` nhưng bỏ sót toàn bộ B. Vì vậy dự án dùng thêm Macro-F1 và balanced accuracy.

### 4.2 Precision — các dự đoán thuộc lớp có đáng tin không?

$$
\operatorname{Precision}_c = \frac{TP_c}{TP_c + FP_c}
$$

Ví dụ dự đoán 10 mẫu fake, 8 mẫu thật sự fake: precision fake `0.8`. Dùng để quan sát **gán nhãn thừa/báo nhầm**. Precision cao vẫn có thể đi kèm bỏ sót nhiều nếu model chỉ chọn ít mẫu dễ.

### 4.3 Recall — model tìm được bao nhiêu mẫu thật của lớp?

$$
\operatorname{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

Có 10 mẫu fake nhưng chỉ tìm được 6: recall fake `0.6`. Dùng để quan sát **bỏ sót**. Recall cao vẫn có thể đi kèm nhiều báo nhầm nếu model gán lớp quá rộng.

### 4.4 F1 của một lớp — kết hợp precision và recall

$$
F1_c = \frac{2P_cR_c}{P_c + R_c}
 = \frac{2TP_c}{2TP_c + FP_c + FN_c}
$$

F1 nằm trong `[0,1]`, cao khi cả precision lẫn recall tốt. Code dùng `zero_division=0`: trường hợp mẫu số không xác định được gán 0, không để NaN. Lớp không có support trong một cohort nhỏ có thể góp F1 bằng 0; phải đọc support trước khi kết luận model yếu ở lớp đó.

Precision, recall, F1 và support từng lớp được in ở `classification_report` của single-label trên terminal. Code hiện tại không lưu báo cáo từng lớp thành một TSV riêng; có thể tính lại từ raw log khi cần.

### 4.5 Macro-F1 — mỗi lớp có trọng số ngang nhau

$$
\operatorname{MacroF1} = \frac{1}{C}\sum_{c = 1}^{C}F1_c
$$

`macro_f1` dùng để đánh giá khả năng nhận diện đều các lớp, kể cả lớp hiếm. Với single-label, code lấy đủ lớp trong `label_order`; với MM-IMDb là đủ 23 genre. Không phải trung bình F1 của riêng những lớp được đoán đúng.

Ví dụ lớp A F1 `0.9`, lớp B F1 `0.3`: Macro-F1 `0.6`. Macro-F1 nhạy với lớp ít mẫu, nên giá trị trên 5 mẫu chỉ dùng kiểm tra luồng, không đủ đánh giá đáng tin cậy toàn bộ task.

### 4.6 Micro-F1 — gộp TP/FP/FN trước khi tính

$$
\operatorname{MicroF1} =
\frac{2\sum_cTP_c}{2\sum_cTP_c + \sum_cFP_c + \sum_cFN_c}
$$

`micro_f1` phản ánh kết quả trên toàn bộ quyết định nhãn, thường chịu ảnh hưởng mạnh của lớp xuất hiện nhiều.

- Single-label, mỗi mẫu chọn đúng một lớp và xét đầy đủ taxonomy: Micro-F1 bằng accuracy.
- Multi-label: Micro-F1 thường khác accuracy; nó phản ánh tổng genre đúng, thiếu và thừa trên tất cả phim.

### 4.7 Balanced accuracy — trung bình recall giữa các lớp có nhãn thật

$$
\operatorname{BalancedAccuracy} =
\frac{1}{|C_{true}|}\sum_{c\in C_{true}}\operatorname{Recall}_c
$$

`balanced_accuracy` có trong audit single-label. Code hiện tại dùng các lớp xuất hiện ở nhãn thật của bộ mẫu được xét, không phải luôn đủ mọi lớp trong taxonomy.

Nó trả lời: **Nếu mỗi lớp thật được coi trọng ngang nhau, model tìm đúng bao nhiêu?** Trong ví dụ 99 A/1 B, luôn đoán A: recall A=1, recall B=0, balanced accuracy `0.5` dù accuracy `0.99`.

Khác Macro-F1: balanced accuracy trung bình recall, không trực tiếp đưa precision vào công thức. Trên cohort thiếu lớp, phạm vi lớp của hai chỉ số có thể khác.

### 4.8 Binary F1 và AUROC — chỉ xuất khi runner gọi đánh giá binary

| Trường | Ý nghĩa | Điều kiện/cách đọc |
|---|---|---|
| `f1` | F1 riêng lớp `positive_label` | Không phải Macro-F1 hai lớp |
| `auroc` | Diện tích dưới ROC khi quét ngưỡng điểm positive | Càng cao càng phân biệt thứ hạng positive/negative tốt |

AUROC dùng điểm của lớp positive **trước argmax**. Runner chính truyền điểm fusion softmax của lớp đó; tên docstring `similarity` không có nghĩa đây là cosine thô. TPR=recall; FPR=`FP/(FP+TN)`.

AUROC `1` là thứ hạng phân biệt hoàn hảo; khoảng `0.5` tương ứng mức không phân biệt theo thứ hạng; dưới `0.5` cần kiểm tra nhãn positive/chiều điểm. AUROC không đo xác suất đã calibration và không bảo đảm F1 tại decision hiện tại tốt.

Fakeddit 2way/Crisis informativeness chỉ có `f1`, `auroc` khi cấu hình `positive_label` được khai báo; YAML chính hiện không khai báo. Upstream hiện gọi đánh giá single-label nên không tự có AUROC. Nếu bộ mẫu chỉ có một lớp thật, AUROC là `None`/`N/A`, không phải 0.

### 4.9 Samples-F1 — trung bình mức khớp genre của từng phim

Với tập genre thật `Y_i` và tập genre dự đoán `P_i`:

$$
\operatorname{SamplesF1} = \frac{1}{N}\sum_i
\frac{2|Y_i\cap P_i|}{|Y_i| + |P_i|}
$$

Code dùng `zero_division=0` cho trường hợp mẫu số bằng 0. Chỉ số này có trong audit multi-label; được xuất dạng `*_samples_f1`, summary rút gọn có thể hiển thị `samples_f1`.

Nó trả lời: **Trung bình mỗi phim được gán thể loại gần đúng đến đâu?** Macro-F1 trung bình theo genre, Samples-F1 trung bình theo phim; hai chỉ số không thay thế nhau.

### 4.10 Hamming accuracy và exact match — hai cách “đúng” khác nhau

$$
\operatorname{HammingAccuracy} =
\frac{\text{số ô nhãn có/không được đoán đúng}}{N\times23}
$$

`hamming_accuracy` tính cả đúng âm tính: model đúng khi không gán genre thật sự không có. Do mỗi phim thường không mang phần lớn trong 23 genre, điểm này có thể cao dù thiếu/thừa genre quan trọng. Có trong kết quả đánh giá MM-IMDb, nhưng summary rút gọn hiện bỏ cột này.

`exact_match` trong raw log là boolean: **toàn bộ tập genre có khớp hoàn toàn không?** Tỷ lệ exact match có thể tính bằng trung bình cột này, nhưng runner hiện chưa xuất metric tổng `exact_match_accuracy` riêng.

Ví dụ một phim có thật `{Drama, Romance}`, model đoán `{Drama, Comedy}`:

- TP=1, FP=1, FN=1; F1 của phim là `0.5`.
- `missing_genres=Romance`, `extra_genres=Comedy`, `exact_match=False`.
- Có 2 ô sai trong 23 genre, nên Hamming accuracy của phim là `21/23 ≈ 0.9130`.

Không được diễn giải điểm Hamming 91.3% thành 91.3% phim đúng hoàn toàn.

### 4.11 Weighted average trên terminal

`weighted avg` trong classification report lấy trung bình chỉ số từng lớp theo support của lớp. Ví dụ weighted-F1 là `Σ support_c × F1_c / N`. Nó chịu ảnh hưởng của lớp đông; không phải Macro-F1 và hiện không phải metric riêng trong summary.

## 5. Điểm từng mẫu trong raw_predictions.tsv

### 5.1 Các trường chung

| Trường | Ý nghĩa | Dùng để làm gì? |
|---|---|---|
| `id` | ID mẫu từ dataset | Tra cùng mẫu giữa các lượt; kết hợp path/index khi ID lặp |
| `text` | Title, tweet hoặc plot dùng làm input | Xem nội dung khiến model chọn nhãn |
| `image_path` | Đường dẫn ảnh đã resolve | Xem ảnh đúng của mẫu |
| `y_true` | Nhãn thật | Mốc đánh giá, không phải điểm model sinh ra |
| `y_pred` | Nhãn model chọn | Kết quả suy luận cần đối chiếu |

Fakeddit 2way: `0=fake`, `1=real`. Fakeddit 6way: `0=True`, `1=Satire/Parody`, `2=False Connection`, `3=Imposter Content`, `4=Manipulated Content`, `5=Misleading Content`. **ID nhãn thật đổi nghĩa giữa hai task.**

### 5.2 Single-label: correct, confidence, margin và score

| Trường | Công thức/ý nghĩa | Cách dùng |
|---|---|---|
| `correct` | `y_pred == y_true` | Lọc mẫu đúng/sai |
| `confidence` | `max(score_<label>)` sau decision scoring/relation | Quan sát độ nhọn của điểm lớp được chọn |
| `margin` | Điểm lớn nhất trừ điểm lớn nhì | Tìm ca có hai ứng viên gần nhau |
| `score_<label>` | Điểm cuối từng lớp, trước argmax | Xem thứ hạng và các nhãn cạnh tranh |
| `score_image_<label>` | Điểm riêng nhánh ảnh sau biến đổi scoring | Kiểm nhánh ảnh thiên về lớp nào |
| `score_text_<label>` | Điểm riêng nhánh text sau biến đổi scoring | Kiểm nhánh text thiên về lớp nào |

Pipeline chính dùng softmax riêng nhánh rồi fusion. Khi bật class-aware relation, `score_<label>` và confidence/margin đã phản ánh hiệu chỉnh; điểm nhánh vẫn là điểm nền riêng từng nhánh.

Ví dụ Fakeddit 2way: `score_0=.52`, `score_1=.48` → `y_pred=0`, `confidence=.52`, `margin=.04`. Đây là quyết định gần biên; `.52` **chưa chứng minh xác suất đúng 52%**. Không có ngưỡng margin “an toàn” chung cho mọi model/task.

Điểm các nhánh không phải cosine thô dù logger dùng tên tham số `sim_image/sim_text`. Riêng upstream `original_image` single-label dùng softmax trên cosine với scale hiệu lực 1 để ghi log, giữ quyết định argmax ảnh gốc; không so confidence của nó trực tiếp với softmax scale lớn ở multimodal.

### 5.3 Multi-label: prob, zscore và lỗi genre

| Trường | Ý nghĩa | Cách dùng |
|---|---|---|
| `y_true`, `y_pred` | Các genre ghép bằng `\|` | So sánh theo tập, không theo thứ tự chuỗi |
| `missing_genres` | Genre có thật nhưng model bỏ sót | Phân tích FN/recall |
| `extra_genres` | Genre model gán nhưng không có thật | Phân tích FP/precision |
| `n_missing`, `n_extra` | Số genre thiếu/thừa | Đo số lỗi của phim |
| `exact_match` | Hai tập genre bằng nhau | Xem ca khớp hoàn toàn |
| `total_errors` | `n_missing+n_extra`, thêm trong error_analysis | Ưu tiên phim sai nhiều genre |
| `zscore_<genre>` | Điểm z cuối khi `relative_z` | So với `relative_z_threshold` |
| `prob_<genre>` | Điểm sigmoid/fusion khi `fixed_threshold` | So với `threshold`; chưa tự chứng minh calibration |
| `score_image_<genre>`, `score_text_<genre>` | Điểm z hoặc sigmoid từng nhánh, tùy mode | Xem nguồn điểm của genre |

Trong `relative_z`, từng nhánh tính `z_c=(S_c−mean(S))/std(S)` qua 23 genre **của chính mẫu đó**, rồi fusion. Dòng có std quá nhỏ được gán 0. Điểm z có thể âm hoặc >1, không phải tỷ lệ hay xác suất.

Rule hiện tại: `relative_z` chọn `z > relative_z_threshold` (mặc định 1.0); `fixed_threshold` chọn `p >= threshold`. Hai rule có cách xử lý bằng ngưỡng khác nhau. Nhiều genre vượt ngưỡng có thể cùng đúng; không lấy margin top-1/top-2 làm thước đo tranh chấp nhãn như single-label.

### 5.4 Các cờ nhánh và fallback

| Trường | Khi True nghĩa là gì? | Cần hiểu gì khi đọc? |
|---|---|---|
| `_text_was_empty` | Text được suy luận coi là rỗng | Mẫu dùng ảnh dự phòng, không có bằng chứng nhánh text |
| `_used_top1_fallback` | Không genre nào qua ngưỡng và đã dùng top-1 dự phòng | Nhãn được gán do fallback, không phải đã qua ngưỡng |
| `_image_encoded` | Nhánh ảnh được tính theo điều kiện phân loại của logger | Không dùng điểm nhánh bỏ qua như kết quả model |
| `_text_encoded` | Nhánh text được tính theo điều kiện phân loại của logger | Nhánh text rỗng/bị bỏ có điểm trống |

Cờ `*_encoded` phục vụ log nhánh phân loại; nếu relation bật, có thể có encode riêng dành cho quan hệ dù nhánh phân loại bị bỏ. Vì vậy đây không phải thống kê toàn bộ số lần encoder chạy.

`_text_was_empty` và `_used_top1_fallback` là hai fallback khác nhau: một cái thiếu input text, một cái không nhãn nào qua ngưỡng.

## 6. diagnostics.tsv: đọc ba nhánh image/text/fusion

### 6.1 Metric theo nhánh và số mẫu

`image` là kết quả riêng nhánh ảnh; `text` là riêng nhánh text; `fusion` là điểm cuối dùng để chọn nhãn. Audit chọn nhãn từ từng bảng điểm bằng rule tương ứng của task.

| Trường | Ý nghĩa | Mục đích |
|---|---|---|
| `branch` | `image`, `text`, `fusion` | Nhận diện nhánh |
| `n_samples` | Số mẫu được đánh giá ở nhánh đó | Kiểm tra mẫu số trước khi so metric |
| `score_type` | `probability` hoặc `relative_z` | Biết cách đọc điểm |
| `macro_f1`, `micro_f1`, `samples_f1`, `balanced_accuracy` | Metric của riêng nhánh | Xem nhánh nào đóng góp tốt/yếu |
| `mean_predicted_labels` | Trung bình số genre sau threshold/fallback | Phát hiện gán quá ít/quá nhiều genre |
| `fallback_rate` | Tỷ lệ dùng top-1 khi fallback bật; 0 nếu fallback tắt | Kiểm tra độ phù hợp của ngưỡng multi-label |

Không phải mọi metric có ở mọi task: audit single-label có Macro-F1/balanced accuracy; audit multi-label có Micro/Macro/Samples-F1. Không có accuracy per-branch riêng trong audit hiện tại.

Text rỗng bị loại khỏi metric nhánh text; nhánh bị bỏ có thể có `n_samples=0`. Do đó score ảnh/text/fusion có thể được tính trên các tập khác nhau. Không kết luận fusion tốt hơn chỉ từ ba con số mà bỏ qua `n_samples` và cohort.

`fallback_rate=0.4` khi bật fallback: 40% mẫu nhánh đó không có genre qua ngưỡng và phải lấy top-1. Khi fallback tắt, giá trị 0 không chứng minh mọi mẫu đều vượt ngưỡng.

### 6.2 Độ nhọn và độ trải của điểm

| Trường | Công thức/phạm vi | Cách dùng và giới hạn |
|---|---|---|
| `top2_margin` | Trung bình `(điểm lớn nhất−điểm nhì)` qua các mẫu | Quan sát độ tách điểm; multi-label không phải thước đo xung đột hai genre |
| `entropy` single-label | Trung bình `−Σ p_c ln(p_c)` | Điểm phẳng thì entropy lớn; không chứng minh sai |
| `entropy` multi-label sigmoid | Trung bình Bernoulli entropy `−p ln p−(1−p)ln(1−p)` qua genre rồi qua mẫu | Quan sát mức phân vân có/không của từng genre |
| `cosine_mean` | Trung bình similarity trước scale/z trên mọi ô mẫu × lớp của nhánh | Quan sát mức similarity nền |
| `cosine_std` | Std population của các ô similarity đó | Quan sát độ trải trước scale/z |

Entropy output dùng **log tự nhiên**, chưa chia cho `ln(C)`. Single-label có trần lý thuyết `ln(C)`; sigmoid multi-label có trần `ln(2)` cho trung bình genre. `relative_z` không xuất entropy vì z không phải xác suất.

Với binary `[.48,.52]`, entropy output khoảng `.69235`; entropy chuẩn hóa khoảng `.99885` chỉ là phép tính bổ sung, chưa phải trường log hiện tại. Không so entropy raw của task 2 lớp và 8 lớp như cùng một thang chuẩn hóa.

`cosine_mean` không phải mean similarity của riêng lớp đúng/được chọn. `mean_sim` là trung bình similarity nhiều prompt; `max_sim` là cực đại theo từng genre qua chunk. `cosine_std` không phải std qua seed và không phải độ tin cậy thống kê của F1.

### 6.3 Các cột nhánh trong summary và fusion gain

Snapshot/summary nội bộ có thể có `image_macro_f1`, `text_macro_f1`, `fusion_macro_f1`, `*_balanced_accuracy`, `*_micro_f1`, `*_samples_f1`, `image_n_samples`, `text_n_samples`, `fusion_n_samples`.

Summary rút gọn đưa metric `fusion_*` về tên chung như `macro_f1`, `samples_f1` và đưa `fusion_n_samples` về `n_samples`. Nhiều cột chi tiết bị bỏ; xem diagnostics/snapshot để tra đầy đủ.

$$
\operatorname{fusion\_macro\_f1\_gain}
 = F1_{fusion}-\max(F1_{image},F1_{text})
$$

- Gain > 0: fusion hơn nhánh đơn tốt nhất trên cùng bộ mẫu.
- Gain < 0: fusion làm Macro-F1 thấp hơn nhánh đơn tốt nhất.
- Gain = 0: không khác ở Macro-F1, dù có thể đổi những mẫu khác nhau.
- Trống: thiếu dữ liệu metric hoặc số mẫu các nhánh không đủ điều kiện so sánh; không diễn giải thành 0.

`fusion_improves_macro_f1=True` khi gain > `1e−12`. Đây là so sánh số học, **không phải kiểm định ý nghĩa thống kê**. `False` có thể là bằng nhau hoặc thấp hơn.

Ví dụ image `.60`, text `.55`, fusion `.63`: gain `.03`, tức tăng **3 điểm phần trăm**, không phải tăng tương đối 3%. Code chỉ tính khi cả hai nhánh có đủ cùng số mẫu như fusion và metric hợp lệ. Upstream `original_image` không báo fusion gain.

## 7. Chỉ số quan hệ ảnh–text khi bật flag

Các cột này chỉ có khi relation được tính. Multi-label hiện chỉ có cosine và cờ áp dụng; không tự tính JS/entropy categorical từ z-score.

| Trường | Phạm vi/ý nghĩa | Dùng để làm gì? |
|---|---|---|
| `image_text_similarity` | Cosine giữa vector ảnh và text ghép đúng mẫu, trong `[-1,1]` | Quan sát gần nhau trong embedding space; không xác minh tin thật |
| `image_text_js_divergence` | JS chuẩn hóa giữa hai phân phối nhãn, trong `[0,1]` | Đo bất đồng dự đoán ảnh/text |
| `image_text_reliability` | Hệ số từ entropy hai nhánh, trong `[0,1]` | Giảm ảnh hưởng bằng chứng relation khi nhánh phẳng |
| `image_text_match_evidence` | Heuristic match trong `[0,1]` | Bằng chứng hiệu chỉnh lớp có policy match |
| `image_text_mismatch_evidence` | Heuristic mismatch trong `[0,1]` | Bằng chứng hiệu chỉnh lớp có policy mismatch |
| `relation_applied` | Có evidence dương được dùng để hiệu chỉnh điểm mẫu | Xác định điểm đã được hiệu chỉnh; không khẳng định nhãn đã đổi |

Đặt `r=cosine`, `M=(P_I+P_T)/2`, số lớp `C`:

$$
J = \frac{KL(P_I \Vert M) + KL(P_T \Vert M)}{2\ln 2}
$$

$$
R = \operatorname{clip}\left(1 - \frac{\max(H(P_I), H(P_T))}{\ln C}, 0, 1\right)
$$

$$
A = \frac{1+r}{2}(1-J)R
$$

$$
D = \frac{1-r}{2}JR
$$

Các trường JS/reliability/match/mismatch lần lượt là `J/R/A/D`. Hai nhánh cùng phẳng có thể JS nhỏ nhưng R nhỏ, nên không coi đó là đồng thuận mạnh. `R` không phải xác suất model đúng; `A/D` không phải xác suất ảnh và caption thật sự khớp/lệch.

Với class-aware Fakeddit 6way: lớp 0 dùng `e_0=A`, lớp 2 dùng `e_2=D`, lớp khác neutral `e_c=0`:

$$
p'_c = \frac{B_c(1 + \lambda e_c)}{\sum_kB_k(1 + \lambda e_k)}
$$

`relation_applied=True` khi evidence được dùng, kể cả nhãn argmax sau đó vẫn giữ nguyên. `diagnostic` chỉ đo/ghi, nên cờ False. Text rỗng hoặc vector không hợp lệ để tính relation cho giá trị trống; trống không có nghĩa similarity bằng 0.

## 8. text_lengths.tsv và chẩn đoán chunking

| Trường | Ý nghĩa | Dùng để làm gì? |
|---|---|---|
| `sample_index` | Vị trí mẫu trong cohort lượt chạy | Ghép với raw log khi ID lặp |
| `id` | ID mẫu | Tra nội dung/path |
| `token_count` | Số token text gốc trước cắt, gồm token đặc biệt theo tokenizer được hỗ trợ | Xem input dài đến đâu |
| `truncated` | `token_count > context_length` | Cảnh báo text gốc vượt context; không tự chứng minh chunk đã mất nội dung |
| `n_chunks`, nếu có | Số đoạn đã encode | Kiểm tra việc chia plot |
| `encoded_chunk_truncated`, nếu có | Cờ các đoạn trong đường chunking đã kiểm context | Phân biệt input gốc dài với đoạn thực sự encode |

Trong manifest/snapshot có block `truncation`; diagnostics đặt tiền tố `input_` cho các trường sau ở hàng text:

| Trường | Ý nghĩa |
|---|---|
| `status` / `input_status` | `measured`: đo được; `unknown`: tokenizer không hỗ trợ đo; `disabled`: tắt đo |
| `context_length` / `input_context_length` | Giới hạn token của text encoder |
| `n_nonempty` / `input_n_nonempty` | Số text không rỗng |
| `n_measured` / `input_n_measured` | Số text đếm được token |
| `n_truncated` / `input_n_truncated` | Số text gốc dài hơn context |
| `truncated_rate` / `input_truncated_rate` | `n_truncated/n_measured`, không chia mọi mẫu |
| `chunking_enabled` | Có ghi nhận đường chunking |
| `total_chunks` | Tổng số đoạn được encode |
| `n_chunked_texts` | Số text được chia thành hơn một đoạn |

Ví dụ plot 200 token, context 77 và được chia thành 3 đoạn vừa context: `truncated=True` cho text gốc có thể đi cùng `n_chunks=3`, `encoded_chunk_truncated=False`. Không kết luận plot bị cắt chỉ từ cột `truncated`.

Giá trị unknown/trống cần được giữ là chưa đo; không coi bằng 0. Đây là token input text/title/plot, không phải quota token của Codex/ChatGPT hay token sinh output của model.

## 9. Metadata: đọc cấu hình để hiểu điểm

### 9.1 Danh tính, kích thước cohort và lỗi

| Trường | Ý nghĩa / mục đích |
|---|---|
| `model`, `backbone` | Model đăng ký và kiến trúc CLIP, tùy runner; dùng nhận diện model |
| `dataset`, `task` | Dataset và loại nhãn đang đánh giá |
| `device` | CPU/CUDA/MPS hiệu lực, nếu runner xuất |
| `n_samples` | Số mẫu fusion trong summary khi audit cung cấp |
| `n_total_samples` | Số dòng raw log hợp lệ trong manifest |
| `n_errors` | Single-label: số mẫu sai; multi-label: số mẫu không exact match |
| `error_rate` | `n_errors/n_total_samples`; không phải tỷ lệ genre sai |
| `max_samples` | Giới hạn cấu hình trước kiểm tra ảnh; không bảo đảm số mẫu output bằng giới hạn |
| `n_skipped_images` | Số ảnh lỗi bị bỏ, hiện upstream có trường này |
| `min_text_length` | Điều kiện lọc text trước lựa chọn mẫu |
| `batch_size` | Số mẫu/chunk xử lý theo batch; không phải số mẫu đánh giá tổng |
| `timestamp` | Thời gian ghi manifest |
| `prediction_log_dir` | Thư mục chứa raw log/manifest của lượt |

Fakeddit giới hạn sau lọc độ dài text; ảnh lỗi tiếp tục bị bỏ và không tự bù bằng dòng tiếp theo. Với single-label, `error_rate=1−accuracy` trên cùng raw cohort. Với multi-label, nó là tỷ lệ phim không khớp hoàn toàn.

### 9.2 Fusion, scale, ngưỡng và prompt

| Trường | Ý nghĩa / mục đích |
|---|---|
| `image_weight`, `text_weight` | Trọng số ảnh và text (`text_weight=1−image_weight`); text rỗng có override ảnh cho mẫu đó |
| `fusion_enabled` | `0<image_weight<1` theo cấu hình; không bảo đảm mỗi mẫu có đủ text |
| `logit_scale`, `logit_bias` | Scale/bias pretrained được wrapper lấy; tên calibrated logits không chứng minh calibration theo dataset |
| `text_logit_scale_requested` | Giá trị scale text yêu cầu; null là chưa override |
| `text_logit_scale_effective` | Scale text thực sự dùng theo mode; có thể null khi bỏ qua |
| `text_logit_scale_status` | `model_default`, `manual`, `ignored_relative_z`, `unused_image_only` |
| `text_scale_override_applied` | Scale text thủ công có được sử dụng không |
| `image_logit_scale_effective` | Scale ảnh hiệu lực trong upstream; đọc thay vì đoán từ tên backbone |
| `decision_mode`, `score_type` | `relative_z`/`fixed_threshold` và kiểu điểm tương ứng |
| `relative_z_threshold`, `threshold` | Ngưỡng hiệu lực của đúng mode; ngưỡng mode còn lại có thể null/không xuất |
| `fallback_top1` | Có bật lấy top-1 nếu không genre qua ngưỡng |
| `model_scale_applied`, `apply_bias_to_text` | Cờ mode MM-IMDb: relative_z bỏ scale/bias |
| `prompt_version`, `prompt_variant` | Phiên bản/biến thể prompt: không tự đồng nghĩa model tốt hơn |
| `prompts_per_label` | Số prompt/lớp nếu mọi lớp bằng nhau; khác nhau có thể null |
| `prompt_definition_version` | Phiên bản định nghĩa bộ prompt |
| `prompt_aggregation`, `text_prompt_aggregation`, `image_prompt_aggregation` | Cách gộp prompt của nhánh tương ứng |
| `rendered_prompts`, `research_image_prompts` | Câu prompt thực tế, nếu manifest xuất |
| `chunk_long_text`, `chunk_aggregation` | Bật chia text và gộp `mean_embed`/`max_sim` |

`mean_embed`: trung bình embedding prompt rồi normalize lại. `mean_sim`: trung bình similarity từng prompt. Khác nhau về thang điểm; khi so confidence cần giữ phương thức này nhất quán.

### 9.3 Cấu hình relation

| Trường | Ý nghĩa / mục đích |
|---|---|
| `relation_enabled` | Flag tính relation đã bật hay chưa |
| `relation_mode_requested` | Mode yêu cầu, có thể `auto` |
| `relation_mode` | Mode hiệu lực `diagnostic`/`class_aware` |
| `relation_weight` | Hệ số `λ` trong công thức relation; không phải image_weight |
| `relation_class_policies` | Lớp nào match/mismatch/neutral |
| `relation_algorithm` | Tên phiên bản công thức được ghi nhận |
| `relation_text_representation` | Vector text dùng tính pair: text đơn hoặc trung bình chunk chuẩn hóa |

Flag mặc định tắt. `auto` class-aware cho Fakeddit 6way, diagnostic cho task còn lại. Upstream `original_image` giữ diagnostic. `relation_enabled=True` không đồng nghĩa `relation_applied=True` cho mọi mẫu.

### 9.4 Hash, split và tái lập

| Trường | Ý nghĩa / mục đích |
|---|---|
| `sample_set_id` | Hash cohort: mẫu/thứ tự, nhãn, text hash, path/stat ảnh và thông tin cohort |
| `prompt_id` | Hash prompt của lượt; phát hiện prompt thay đổi |
| `config_id` | Hash cấu hình/prompt/model/inference hiệu lực được audit lưu |
| `split` | Tên tập do YAML khai báo: val/test/unverified… |
| `split_verified` | Hiện False: code không xác minh dữ liệu thật thuộc split đó |
| `sample_lock_path` | File khóa cohort nếu chế độ khóa mẫu bật |
| `git_hash` | Commit HEAD lúc lưu; không tự phản ánh thay đổi chưa commit |
| `source_sha256`, `source_hashes` | Hash file code; giúp đối chiếu code thực tế chạy |
| `evaluation_snapshot`, `frozen_plan` | Vị trí snapshot/kế hoạch để truy chi tiết |
| `descriptor_sha256`, `vocabulary_sha256` | Hash mô tả lớp/từ điển của phương pháp nghiên cứu |

Hash là định danh, không có ý nghĩa tốt/xấu hay cao/thấp. Đổi cấu hình cohort/path/stat có thể đổi `sample_set_id` dù ID trông giống nhau; khác hash phải đối chiếu snapshot. Hash cohort không phải hash toàn bộ nội dung byte của mọi ảnh.

`split=val` hoặc `test` không tự chứng minh nguồn split chính xác. Khi so sánh, giữ cùng task/taxonomy, mẫu, cách lọc, ngưỡng và nguồn dữ liệu; chỉ đổi yếu tố muốn nghiên cứu.

## 10. Các output riêng của nghiên cứu

### 10.1 Phương pháp và seed

| Trường | Ý nghĩa |
|---|---|
| `method`, `image_method` | Phương pháp tạo/chấm prompt ảnh: clip/baseline/class_name/waffle/dclip… tùy runner |
| `seed`, `research_seed` | Seed sinh prompt ngẫu nhiên; null có thể là phương pháp không chạy lặp seed |
| `output_variant` | Upstream `original_image` hoặc `multimodal` |
| `classification_scope` | `paper_image`, `multimodal_extension`, `multi_label_extension` |
| `protocol`, `method_protocol` | Mô tả protocol, như frozen/no-fit/exploratory; không phải metric |
| `implementation` | Phiên bản cách triển khai; phân biệt adapted với hàm tác giả |
| `waffle_pairs`, `waffle_alphabet` | Số cặp và bảng ký tự sinh prompt của Waffle |
| `descriptor_provenance` và các trường nguồn descriptor | Nguồn mô tả lớp; không tự khẳng định dùng descriptor nguyên paper |

`original_image` là decision riêng ảnh; không được đọc như kết quả fusion. Các bản multimodal/multi-label là mở rộng pipeline; không coi metric trên dataset của bạn là tái hiện nguyên benchmark tác giả.

### 10.2 Trung bình và độ lệch qua seed

| Trường | Công thức / cách đọc |
|---|---|
| `n_runs` | Số lượt trong nhóm được tổng hợp |
| `<metric>_mean` | Trung bình metric qua các lượt |
| `<metric>_std` | Std population (`ddof=0`) qua các lượt; null nếu chỉ một lượt |

$$
\bar m = \frac{1}{K} \sum_{k=1}^{K} m_k
$$

$$
\sigma = \sqrt{\frac{1}{K} \sum_k (m_k - \bar m)^2}
$$

Ví dụ Macro-F1 qua 3 seed `.40, .42, .44`: mean `.42`, std khoảng `.01633`. Std nhỏ cho thấy ít dao động giữa những seed này, không chứng minh model luôn đúng hoặc có ý nghĩa thống kê.

`_std` không phải standard error/khoảng tin cậy và không phải `cosine_std`. Với `n_runs=1`, cột trống không có nghĩa std đã đo bằng 0. Upstream nhóm theo model/dataset/task/method/output_variant; adapted nhóm model/image_method trong lượt cấu hình một dataset/task. Runner thường chỉ lặp seed cho Waffle, nên số lượt giữa phương pháp có thể khác nhau.

### 10.3 Đối chứng quan hệ ảnh–text

Trong `results/relation_ablation/<lượt>/summary.tsv`:

| Trường | Ý nghĩa |
|---|---|
| `method` | `baseline`, `legacy_false_connection_0.1`, `class_aware` |
| `n_samples`, `accuracy`, `macro_f1`, `balanced_accuracy` | Metric trên cùng nhóm tối đa 5 dòng nguồn sau điều kiện hợp lệ |
| `relation_weight` | Trọng số công thức của phương pháp; legacy và mới khác công thức |
| `changed_predictions` | Số mẫu đổi nhãn so với baseline |
| `model`, `device`, `sample_set_id` | Bối cảnh chạy/cohort |

`changed_predictions=2` chỉ có nghĩa hai nhãn đổi, không có nghĩa sửa đúng hai mẫu. Cần so `y_true`/`y_pred` trong `comparison.tsv` để phân biệt `sai→đúng`, `đúng→sai`, `sai→sai`. `js_divergence`, `reliability`, `match_evidence`, `mismatch_evidence` ở comparison là tên rút gọn của các trường relation đã giải thích.

## 11. Cách đọc kết quả để quyết định bước tiếp theo

| Điều bạn thấy | Kiểm tra tiếp | Dùng để quyết định gì? |
|---|---|---|
| Accuracy cao, Macro-F1 thấp | Support, recall/F1 lớp hiếm | Model đang ưu tiên lớp đông hay không |
| Macro-F1 fusion thấp hơn image | Gain, số mẫu nhánh, score ảnh/text | Text/fusion có kéo lệch kết quả hay không |
| Nhiều mẫu sai nhưng confidence cao | Error log, nhãn, prompt, scale | Tìm lỗi hiểu lớp có hệ thống |
| Margin nhỏ, JS nhỏ | Điểm/entropy riêng hai nhánh | Hai nhánh cùng ít phân biệt hay cùng đồng thuận |
| Margin nhỏ, JS lớn | Ảnh, text, nhãn quy định | Hai phương thức đang xung đột về lớp |
| Multi-label recall kém, nhiều missing | Điểm genre quanh ngưỡng, plot/chunking | Ngưỡng hoặc input có bỏ sót bằng chứng |
| Multi-label precision kém, nhiều extra | Ngưỡng, prompt genre, điểm nhánh | Có đang gán genre quá rộng |
| Hamming cao, exact match thấp | Missing/extra và F1 | Không nhầm đúng nhiều ô âm tính với đúng thể loại phim |
| Fallback rate cao | Threshold và phân bố điểm, mode | Cách chọn nhãn có đang phải dự phòng nhiều |
| Truncation rate cao | n_chunks, encoded_chunk_truncated | Phân biệt text dài với text thật sự bị cắt |
| Seed std lớn | Đủ seed/cohort, prompt ngẫu nhiên | Kết quả có nhạy với seed |

Chưa có output tự động cho ECE, Brier score, risk/coverage, prediction set conformal hoặc cơ chế giải quyết uncertainty mới. Những nội dung đó hiện là nghiên cứu đề xuất, xem [báo cáo uncertainty](nghien_cuu_du_doan_thieu_chac_chan_20261013.md).

## 12. Code làm căn cứ

- [evaluate.py](../src/evaluate.py): metric và classification report.
- [evaluation_audit.py](../src/evaluation_audit.py): metric từng nhánh, token/chunk và snapshot.
- [logging_utils.py](../src/logging_utils.py): cột raw/error log và manifest.
- [inference.py](../src/inference.py): biến đổi similarity, fusion, threshold và fallback.
- [pair_relation.py](../src/pair_relation.py): cosine/JS/reliability/evidence/hiệu chỉnh.
- [text_scale.py](../src/text_scale.py): metadata về scale và trọng số.
- [run/common.py](../src/run/common.py): summary rút gọn và fusion gain.
- [run_upstream.py](../src/research/run_upstream.py), [run_adapted_methods.py](../src/research/run_adapted_methods.py): metadata và tổng hợp seed.
- [run_relation_ablation.py](../src/research/run_relation_ablation.py): changed_predictions/comparison.
- [demo.py](../src/demo.py): hình minh họa dự đoán.

Tài liệu dựa trên code/cấu hình, không mở dataset, không suy luận hoặc huấn luyện. Một trường có thể không xuất trong lượt cũ hoặc khi tính năng tương ứng tắt; dùng manifest/code phiên bản lượt đó để đối chiếu.
