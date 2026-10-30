# Kiểm tra trước khi mở rộng ba bộ dữ liệu lên 3.000 mẫu

**Cập nhật sau khi đối chiếu raw:** xem
[rà soát nguồn và preprocessing](ra_soat_raw_preprocessing_3000.md).
MM-IMDb và CrisisMMD hiện trộn split; test CrisisMMD có 2.237 cặp/task.
Các nhận xét dưới đây về khả năng dùng 3.000 mẫu là cho thăm dò trên processed,
không phải xác nhận cohort test chính thức.

Ngày đối chiếu: 20/10/2026. Sau yêu cầu kiểm tra 3.000 mẫu của người dùng,
đã thống kê tối đa 3.000 dòng mỗi tệp processed, chỉ các cột ID, nhãn, event,
đường dẫn ảnh và trường metadata liên quan. Không đọc cột text/plot, không mở
ảnh, không truy cập dữ liệu thô, không chạy tiền xử lý hoặc GPU.

Số liệu máy đọc: [kiem_tra_3000_metadata.json](kiem_tra_3000_metadata.json).
Đọc bằng `read_csv(usecols=..., nrows=3000)` ngay từ đầu, không nạp toàn bộ tệp
rồi cắt. Các số đếm là **đầu vào trước bộ lọc text và kiểm tra nội dung ảnh**.

## Kết luận cho từng bộ dữ liệu

| Bộ dữ liệu/tác vụ | Số dòng đã đọc | Lệch lớp | Quyết định cho 3.000 mẫu |
|---|---:|---|---|
| Fakeddit 2way | 2.841, đã hết tệp | Fake 1.441; Real 1.400, tỷ số 1,03 | Chưa đủ 3.000; cần bổ sung mẫu riêng biệt từ nguồn hợp lệ hoặc báo đúng số thực có. |
| Fakeddit 6way | Cùng 2.841 dòng | True 1.400; Imposter Content 54, tỷ số 25,93 | Cần thiết kế lại cohort nếu đánh giá sáu lớp; không nhân bản 54 mẫu để ép đủ chỉ tiêu. |
| CrisisMMD informativeness | 3.000 dòng đầu | 1.622/1.378, tỷ số 1,18 | Đủ số dòng, nhãn tương đối cân bằng; còn thiên lệch sự kiện. |
| CrisisMMD humanitarian | Cùng 3.000 dòng đầu | Not Humanitarian 1.378; Missing or Found People 1 | Không phù hợp để kết luận ổn định cho cả tám lớp từ cohort đầu tệp này. |
| MM-IMDb genres | 3.000 dòng đầu | Drama 1.640; Film-Noir 47, tỷ số 34,89 | Có thể dùng cho đánh giá thăm dò; không cần ép 23 nhãn bằng nhau. |

Không có cặp ID–đường dẫn ảnh trùng hoàn toàn trong ba phần dữ liệu đã đọc.
Mọi dòng đều có tệp ảnh tồn tại; chưa kiểm tra nội dung ảnh có mở được hay không.
Không có ID trùng trong Fakeddit và MM-IMDb. CrisisMMD có 2.679 tweet ID riêng
biệt trên 3.000 cặp ảnh–tweet; không xóa các ảnh khác nhau chỉ vì tweet ID trùng.

### Fakeddit

Tệp đang dùng chỉ có 2.841 dòng. Cả 2.841 dòng có `hasImage=True`, ID riêng biệt
và ảnh tồn tại. Hai hệ nhãn không mâu thuẫn ở quan hệ `6_way_label=0` tương ứng
`2_way_label=1`. Vì runner còn loại tiêu đề có độ dài sau bỏ dấu cách không quá
20 ký tự, số mẫu suy luận thực tế **có thể thấp hơn 2.841**. Lần kiểm tra này
không đọc tiêu đề nên chưa đếm số dòng vượt bộ lọc đó.

| 6_way_label | Tên lớp | Số mẫu |
|---|---|---:|
| 0 | True | 1.400 |
| 1 | Satire/Parody | 134 |
| 2 | False Connection | 430 |
| 3 | Imposter Content | 54 |
| 4 | Manipulated Content | 744 |
| 5 | Misleading Content | 79 |

Không thể đồng thời làm đều hai và sáu lớp bằng cùng một phép chia đều: nếu
sáu lớp đều 500 mẫu, nhãn True có 500 nhưng nhóm fake của 2way có tổng 2.500.
Nếu cần một cohort chung, xác định phân bố mục tiêu từ split nguồn đã xác minh;
nếu dùng cohort chẩn đoán riêng cho từng task, báo rõ hai cohort khác nhau.
Không đổi nghĩa ID nhãn: 0 của 2way là fake, 0 của 6way là True.

### MM-IMDb

Cả 23 thể loại đều có mẫu. Có 7.349 lượt gán nhãn trên 3.000 phim, trung bình
2,4497 nhãn/phim; `n_genres` khớp số nhãn phân tách ở mọi dòng, không có tập nhãn
rỗng. Các thể loại hiếm nhất là Film-Noir 47, Short 51, Sport 62 và Western 67.
Đây là nhiều nhãn đồng xuất hiện, không phải 23 nhóm mẫu rời nhau.

| Thể loại | Support | Thể loại | Support |
|---|---:|---|---:|
| Drama | 1.640 | Sci-Fi | 204 |
| Comedy | 1.009 | Family | 183 |
| Romance | 657 | Biography | 160 |
| Thriller | 590 | War | 155 |
| Crime | 438 | History | 126 |
| Action | 395 | Music | 125 |
| Horror | 304 | Animation | 112 |
| Adventure | 296 | Musical | 90 |
| Mystery | 219 | Western | 67 |
| Fantasy | 210 | Sport | 62 |
| Documentary | 209 | Short | 51 |
| Film-Noir | 47 | | |

Micro-F1 phản ánh nhiều hơn các lượt gán nhãn phổ biến; Macro-F1 coi các thể loại
ngang nhau và nhạy hơn với lớp ít mẫu. Với Film-Noir 47 mẫu, một TP tương ứng
1/47 ≈ 2,13 điểm phần trăm Recall. Không có một ngưỡng tỷ số lớp chung tự động
buộc lấy lại dữ liệu. Nếu cần chọn lại từ một pool lớn, dùng phân tầng nhiều nhãn
để giữ gần đúng tần suất nhãn và theo dõi đồng xuất hiện, không lấy 500 phim mỗi
thể loại rồi ghép vì dễ tạo bản ghi trùng và làm méo quan hệ nhãn.

## Kết quả đã có và phân bố quan sát được

Kho dự án đã có hai lượt **MetaCLIP2 / informativeness / 3.000 mẫu**:

| Phương pháp | Run ID | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Baseline ảnh + tweet | `6acc864b7851` | 0,824333 | 0,824283 |
| DCLIP adapted ảnh + tweet | `03f1882357bb` | 0,865667 | 0,864545 |

Không cần chạy lại hai lượt này chỉ để có kết quả 3.000 mẫu. Cần đối chiếu lại
cohort/prompt nếu đầu vào hoặc cấu hình đã thay đổi trước khi ghép với lượt mới.

Các số dưới đây tính từ `y_true` và đường dẫn ảnh trong `raw_predictions.tsv`
của các archive, không dùng nhãn dự đoán để thống kê phân bố thật.

| Nhãn informativeness | 1.000 mẫu | 3.000 mẫu | Tỷ lệ ở 3.000 mẫu |
|---|---:|---:|---:|
| Informative | 537 | 1.622 | 54,07% |
| Not informative | 463 | 1.378 | 45,93% |

Tỷ số lớp lớn/nhỏ ở 3.000 mẫu là 1,177. Không có bằng chứng mất cân bằng nhãn
nghiêm trọng cho **cohort informativeness này**. Đây không phải thống kê nhãn
humanitarian và không phải thống kê toàn bộ CrisisMMD.

| Sự kiện suy ra từ đường dẫn ảnh | 1.000 mẫu | 3.000 mẫu | Tỷ lệ ở 3.000 mẫu |
|---|---:|---:|---:|
| California wildfires | 470 | 470 | 15,67% |
| Hurricane Harvey | 530 | 1.745 | 58,17% |
| Hurricane Irma | 0 | 785 | 26,17% |
| Bốn sự kiện còn lại của bộ phát hành | 0 | 0 | 0% |

**Vấn đề chính là độ bao phủ sự kiện.** Loader chọn đầu tệp, không chọn ngẫu nhiên
phân tầng. Tập 3.000 có 2.679 ID riêng biệt; 1.000 dòng đầu trùng hoàn toàn và
cùng thứ tự với cohort 1.000 đã dùng (ID, text, tên ảnh và nhãn). Hai nghìn dòng
bổ sung có 1.085 informative và 915 not informative. Vì vậy việc tăng cỡ mẫu chưa
loại bỏ thiên lệch thứ tự và không tạo một test độc lập. Phân bố sự kiện suy ra
từ đường dẫn trong log đã được đối chiếu với cột `event` ở 3.000 dòng TSV nguồn;
số đếm ba sự kiện trùng nhau.

Humanitarian đã được thống kê trực tiếp ở cùng 3.000 dòng nguồn:

| Nhãn | 1.000 mẫu lịch sử | 3.000 dòng nguồn |
|---|---:|---:|
| Not humanitarian | 463 | 1.378 |
| Other relevant information | 198 | 789 |
| Rescue, volunteering or donation effort | 178 | 465 |
| Infrastructure and utility damage | 145 | 337 |
| Affected individuals | 9 | 17 |
| Vehicle damage | 4 | 11 |
| Injured or dead people | 2 | 2 |
| Missing or found people | 1 | 1 |

Tỷ số lớn/nhỏ của 3.000 dòng là 1.378. Bốn lớp hiếm có tổng 31 mẫu (1,03%);
hai lớp ít nhất vẫn chỉ có 2 và 1 mẫu dù tăng từ 1.000 lên 3.000. Một quyết định
đúng/sai ở lớp có một mẫu đổi Recall 100 điểm phần trăm; ở lớp hai mẫu là 50 điểm.
Đây là giới hạn độ ổn định ước lượng, không thể khắc phục bằng chỉ chuẩn hóa điểm
hoặc tăng class weight. Chưa có output inference humanitarian 3.000 mẫu; bảng
mới là thống kê nhãn đầu vào, không phải kết quả mô hình.

## Cách xử lý phù hợp với zero-shot

1. **Giữ cấu hình suy luận đã chốt khi mở rộng cohort lịch sử.** Không thay
   prompt, trọng số fusion, ngưỡng hay gộp nhãn chỉ để tăng điểm trên nhãn đánh giá
   vừa xem. Baseline và DCLIP cần chạy cùng task, cùng cohort và thứ tự lớp.
2. **Không oversample/SMOTE/nhân bản lớp hiếm trong tập đánh giá chính.** Việc này
   thay đổi phân bố cần đo và không tăng lượng quan sát độc lập. Mất cân bằng tập
   đánh giá cũng không tự động đòi hỏi huấn luyện lại một backbone zero-shot.
3. **Báo Macro-F1, Recall/F1 và support từng lớp**; dùng Accuracy bổ sung. Với
   informativeness có thể báo thêm AUROC và balanced accuracy. Mức tăng ở lớp có
   1–2 mẫu chưa đủ để kết luận ổn định.
4. **Tách mở rộng lịch sử khỏi đánh giá đại diện.** Profile đi kèm giữ cách chọn
   3.000 dòng đầu để đối chiếu những lượt hiện có. Nếu mục tiêu là một cohort
   mới bao phủ dữ liệu phát hành, cần xác minh split trước, rồi cố định phép lấy
   mẫu theo sự kiện và nhóm tweet với seed/manifest chung. Chọn ngẫu nhiên chỉ
   trong 3.000 dòng đầu vẫn không lấy được bốn sự kiện chưa xuất hiện.
5. Nếu muốn một tập phụ cân bằng theo lớp để chẩn đoán, công bố rõ cách chọn,
   giữ các quan sát riêng biệt và báo tách khỏi tập phân bố tự nhiên. Không lấy
   điểm của tập phụ làm điểm benchmark chính thức.

"Chuẩn hóa dữ liệu" nên phân biệt với "làm đều lớp": giữ schema/ánh xạ nhãn,
loại cặp trùng thật, kiểm split, ảnh lỗi và độ dài text; ảnh vẫn dùng preprocess
của checkpoint, MM-IMDb vẫn dùng chia plot và quy tắc nhiều nhãn đã chốt. Không
cần sửa các chuẩn hóa embedding hoặc scale chỉ vì số mẫu giữa các lớp khác nhau.

**Đề xuất:** bổ sung hoặc tạo lại cohort Fakeddit từ pool hợp lệ vì thiếu số mẫu;
chọn lại cohort CrisisMMD bao phủ sự kiện và kiểm tra đủ support trước khi chốt
đánh giá tám lớp; có thể giữ MM-IMDb 3.000 mẫu cho thăm dò, kèm Micro/Macro-F1
và support. Việc chọn một cohort mới cần pool ngoài phạm vi 3.000 dòng đầu và
split được xác minh; lần kiểm tra này chưa đọc pool đó hoặc tạo lại dataset.

Nguồn: [CLIP](https://arxiv.org/abs/2103.00020) về chuyển tác vụ zero-shot;
[CrisisNLP](https://crisisnlp.qcri.org/crisismmd) về bảy sự kiện, nhãn riêng của
ảnh/text và các bản split có/không yêu cầu nhãn đồng thuận;
[scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)
về balanced accuracy. Việc giữ nguyên taxonomy tám lớp của dự án khác giao thức
multimodal năm lớp có gộp nhãn trong tài liệu phát hành; không tự động chuyển
giao thức giữa các lượt so sánh.

Phân tầng nhiều nhãn: [Sechidis, Tsoumakas và Vlahavas (2011), DOI
10.1007/978-3-642-23808-6_10](https://doi.org/10.1007/978-3-642-23808-6_10).
Lưu ý [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)
không xem phân tầng là giải pháp bảo đảm độ độc lập thống kê hoặc loại bỏ mọi
thiên lệch đánh giá.

## Hồ sơ chạy đã chuẩn bị

`configs/launcher_3000.yaml` dùng launcher hiện có, gồm bốn job:
`crisismmd_info`, `crisismmd_humanitarian`, `adapted_crisismmd_info` và
`adapted_crisismmd_humanitarian`. Mỗi job giới hạn 3.000; nhánh adapted chỉ có
`methods: [dclip]`. Mỗi lượt Kaggle phải chọn đúng một model. Các job đều tắt
mặc định, không thay các YAML thí nghiệm 1.000 mẫu.

Đã kiểm tra 16 kế hoạch cấu hình (4 job × 4 backbone) bằng `validate`,
`validate_single_cloud_trial` và `job_plan`, chỉ tạo cấu hình trong thư mục tạm.
Không chạy `run_one(..., use_kaggle=true, dry_run=true)` vì chế độ đó vẫn đóng
gói và đọc toàn bộ processed khi `data_mode=full_dataset`.

**Trạng thái sau kiểm tra:** đã đo metadata cho cả ba bộ dữ liệu, chưa chạy mô
hình 3.000 mẫu mới và chưa tạo lại dữ liệu. Profile CrisisMMD vẫn là cấu hình
mở rộng cohort lịch sử theo đầu tệp; không phải cấu hình sửa thiên lệch sự kiện.
Không đưa lệnh chạy cả ba bộ 3.000 mẫu như thể tất cả đã sẵn sàng, vì Fakeddit
hiện thiếu số dòng và cohort humanitarian vẫn quá ít mẫu ở một số lớp.

Phần kiểm tra dữ liệu đã hoàn tất trong phạm vi giới hạn. Đã cập nhật cách hiển
thị nhãn trong báo cáo thành tiếng Anh kèm nghĩa tiếng Việt; không đổi ID nhãn,
dữ liệu hoặc prompt mô hình. Thống kê đầu vào mới được tách riêng khỏi bảng
điểm thực nghiệm; chưa báo Fakeddit/MM-IMDb/humanitarian như đã inference 3.000.
