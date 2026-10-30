# Phát triển cổng kết hợp bằng chứng và đánh giá trên dev

## Giả thuyết và quy tắc đã cố định

Cổng cứng giữ nguyên `semantic informativeness argmax`. Đối chứng mới
`margin_pool` dùng hai margin cosine có dấu:

```
d_info = s(informative) - s(not_informative)
d_human = max(s(bảy nhóm cụ thể)) - s(not_humanitarian)
d_pool = (d_info + d_human) / 2
```

Nếu d_pool >= 0 thì lấy nhóm có điểm cao nhất trong bảy nhóm; nếu âm trả
not_humanitarian. Cổng là tổng hợp điểm liên tục trước quyết định, không có trọng
số học được hay ngưỡng tối ưu theo nhãn. Margin không phải confidence đã hiệu
chuẩn. Đây là giả thuyết inference/ablation, không tuyên bố ý tưởng mới toàn cầu.

Đã kiểm tra trên log train seed 0, trước khi chạy model trên dev:

| Phương pháp | Accuracy | Macro-F1 |
|---|---:|---:|
| Một tầng | 57,60 | 44,99 |
| Hai tầng cứng | 68,03 | 49,88 |
| Margin pool cố định | 63,93 | 47,74 |

Margin pool sửa đúng 118 nhưng làm sai 241 mẫu so với cổng cứng. Không thay
cổng cứng mặc định hoặc giấu kết quả âm. Dùng dev để kiểm tra sự tổng quát và
lựa chọn phiên bản; không tiếp tục vặn ngưỡng trên test để buộc tăng điểm.

## Chuẩn bị dev

Theo yêu cầu mới cho phép dùng raw CrisisMMD, đã chạy `prepare_dev.py`. Phiên bản
hiện tại chỉ đọc `task_humanitarian_text_img_dev.tsv`, không mở train hoặc test.
Annotation dev có 2.237 ảnh; loại 18 ảnh có SHA256 trùng với bộ phát triển đã dùng,
giữ 2.219 ảnh. Mọi ảnh còn lại đã verify và decode, sao chép nguyên byte sang
`data/processed/pro_crisismmd_dev_humanitarian/`. Không ghi đè cohort train.

| Lớp | Support dev sau lọc |
|---|---:|
| not_humanitarian | 1078 |
| infrastructure_and_utility_damage | 400 |
| other_relevant_information | 336 |
| rescue_volunteering_or_donation_effort | 280 |
| affected_individuals | 77 |
| vehicle_damage | 38 |
| injured_or_dead_people | 8 |
| missing_or_found_people | 2 |

Nhãn đích là label_image. Không yêu cầu nhãn informative thật để dự đoán gate;
không gán giả label_informative từ humanitarian. Cột gate để rỗng, cohort gate
lưu y_true=null, không đánh giá metric informativeness trên bộ này. Chỉ dùng
annotation humanitarian dev chính thức, giữ mọi dòng hợp lệ sau lọc trùng.

Kiểm tra ban đầu cho thấy hai dev task có 2.237 dòng riêng nhưng chỉ giao nhau
344 ảnh; bộ pilot giao hai task 341 ảnh ở pro_crisismmd_dev không được dùng cho
đánh giá chính vì làm mất phần lớn dev. Runner đã sửa để không cần lấy giao.

Manifest ghi nguồn/hash, các mẫu loại và lý do, support, SHA256 ảnh và TSV.
Chưa loại ảnh gần trùng; các event vẫn là những event có trong train. Dev là
validation phục vụ phát triển, không gọi là test cuối. Lớp chỉ hai mẫu vẫn rất
không ổn định.

## Chạy

Đã smoke OpenAI CLIP thật trên CPU với năm ảnh đầu dev, kiểm tra artifact và
argmax. Đây chỉ là kiểm tra kỹ thuật. Chưa chạy MetaCLIP2 trên toàn dev, chưa
gửi job Kaggle. Khi chạy một job, model tải một lần, ảnh encode một lần và xuất
cả ba dòng đối chứng; không cần gửi ba job riêng.

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/clip_two_stage/launcher_dev_humanitarian.yaml --job two_stage_dev --seed 0 --use-kaggle true
```

Không thêm --max-samples 3000 vì dev thực có 2.219 ảnh. Muốn smoke T4 trước dùng
--max-samples 5. Muốn local dùng --use-kaggle false; model mặc định vẫn MetaCLIP2.

Đầu ra: `results/clip_two_stage_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/`.
Tên thư mục heldout nghĩa là tách khỏi cohort train đã chọn phương pháp; manifest
và cohort vẫn ghi source_split=dev, không được gọi là test cuối.

Sau khi tải kết quả:

```sh
venv/bin/python -B experiments/clip_two_stage/analyze.py --run-dir PATH_KET_QUA_DEV
```

Kiểm tra ba metric, gate false positives/negatives, từng lớp, fixed/broken và
theo sự kiện. Nếu margin pool không hơn cổng cứng, loại nó. Test chỉ chuẩn bị và
chạy sau khi chốt phiên bản; phải kiểm tra không trùng cả cohort train và dev.
Tham chiếu loại trùng hiện tại chỉ gồm cohort train đã phát triển, cần bổ sung
dev vào tham chiếu trước vòng test cuối. Không hứa đạt điểm cao hơn trên dev.
