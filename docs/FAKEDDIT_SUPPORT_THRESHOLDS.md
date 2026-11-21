# So support policy ở cùng mục tiêu recall Fake

Phân tích CPU offline từ run `fakeddit_fewshot_support` full dev. Không login Kaggle, không gửi job, không nạp backbone/feature cache/trọng số hoặc đọc dataset. Giữ nguyên run nguồn.

## Chạy

Từ thư mục gốc dự án:

```sh
venv/bin/python -B scripts/analyze_fakeddit_support_thresholds.py \
  --run results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run \
  --target-recall 0.87
```

Mặc định chỉ phân tích ảnh–tiêu đề, selection fraction 0,5 và split seed 42. Target là tỷ lệ, `0.87` nghĩa là 87%, không nhập `87`. Muốn phân tích ba modality, thêm `--modalities image title image_title`. Có thể truyền nhiều target đã định trước, ví dụ `--target-recall 0.85 0.87 0.90`; không chọn target tốt nhất bằng kết quả audit rồi gọi đó là xác nhận độc lập.

Mỗi lần chạy tạo thư mục mới dưới `results/final/fewshot_support_thresholds/recall_tradeoff/`, không ghi đè nguồn hoặc kết quả trước. Dùng `--output-root` để đổi nơi lưu; không được đặt bên trong run nguồn.

## Quy tắc

- Chỉ nhận full dev 3.000 mẫu, hai policy, năm seed 0–4, nhãn 0=Real/1=Fake và `test_opened=false`.
- Kiểm receipt/hash run và checksum hai artifact dùng trực tiếp: `cohort.json`, `replay_scores.npz`. Không xác minh lại mọi artifact của source và không đọc `features.npz` hay `linear_models.npz`; phạm vi xác minh ghi trong run mới.
- Kiểm score hữu hạn trong [0,1], tổng hai cột gần 1, đúng shape, ID/thứ tự và mapping nhãn. Chia dev theo sáu subtype: sắp ID rồi shuffle bằng RNG seed cố định, lấy floor(fraction × số mẫu), giới hạn để mỗi subtype có cả selection lẫn audit. Cả hai policy, mọi modality/seed/target dùng cùng partition.
- Với từng policy/modality/seed/target, chỉ dùng selection để chọn **ngưỡng cao nhất đạt recall Fake ít nhất bằng target**. Quyết định tuned là `p_fake >= threshold`; tied scores cùng được nhận Fake. Quy tắc này ưu tiên ngưỡng cao nhất khả thi, không tối ưu macro-F1 bằng audit.
- Baseline mặc định giữ `argmax([p_real,p_fake])`: đồng hạng chọn Real. Không thay baseline thành quy tắc `>=0,5`.
- Áp dụng nguyên ngưỡng lên audit; không tìm lại ngưỡng ở đó. Vì phân bố và hữu hạn mẫu, recall audit có thể thấp hơn target. Target selection bằng nhau không có nghĩa recall audit bằng nhau chính xác.
- Audit là một phần của dev đã được xem trước; đây là phân tích phát triển, không phải test độc lập. Std theo seed không phải khoảng tin cậy; không chọn seed tốt nhất.

## Output

| File | Nội dung |
|---|---|
| `split.json` | ID, subtype, partition; hash nằm trong frozen plan |
| `thresholds.tsv` | Ngưỡng và metric selection theo policy/modality/seed/target |
| `precision_recall_selection.tsv` | Đường PR theo các score khác nhau; chỉ selection |
| `summary.tsv`, `metrics_by_seed.tsv` | Metric **audit** mặc định và tuned; mean/sample std và từng seed |
| `class_summary.tsv` | Recall Real/Fake, precision Fake, số lỗi hai chiều trên audit |
| `comparisons.tsv`, `comparisons_summary.tsv` | Balanced − legacy ghép cặp cùng modality/seed/quy tắc; delta điểm phần trăm và số sửa sai/làm sai |
| `subtype_detection.tsv`, `subtype_summary.tsv` | Recall phát hiện binary trong từng subtype; không phải accuracy tầng hai |
| `predictions.tsv`, `per_class.json` | Dự đoán audit, score, ngưỡng và metric lớp để tái kiểm tra |
| `frozen_plan.json`, `run.json`, `completed.json` | Nguồn, hash, tham số, giới hạn diễn giải và checksum đầu ra |

## Kết quả chạy kiểm tra đầu tiên

Run `results/final/fewshot_support_thresholds/recall_tradeoff/dev_1b4a7accc90c`, target selection 87%, ảnh–tiêu đề, 1.500 selection và 1.500 audit, năm seed:

| Policy tuned | Accuracy audit (%) | Macro-F1 audit (%) | Recall Real (%) | Recall Fake (%) |
|---|---:|---:|---:|---:|
| Legacy 64/320 | 81,65 | 80,80 | 75,64 | 85,67 |
| Balanced 192/192 | 82,28 | 81,46 | 76,57 | 86,10 |

Chênh macro-F1 trung bình +0,66 điểm phần trăm, sample std của chênh lệch 0,90 điểm; bốn seed dương, một seed âm. Có dấu hiệu lợi ích nhỏ sau điều chỉnh ngưỡng, chưa đủ kết luận thắng ổn định hoặc đạt recall 87% trên dữ liệu mới. Không so trực tiếp số audit này với bảng full dev 3.000 mẫu như cùng một phép đánh giá.

## Kiểm tra triển khai

```sh
venv/bin/python -B -m unittest discover -s tests -p test_fewshot_support_thresholds.py -q
```

Test dùng dữ liệu tổng hợp: threshold/ties, split không trùng và không phụ thuộc thứ tự ID, thay score audit không làm đổi ngưỡng, không sửa nguồn, checksum/split/mapping/score sai, CLI nhiều target/modality. Kiểm thêm regression của `test_fewshot_task1_analysis.py`.

Kết quả xác minh triển khai ngày 16/11/2026: 8 test mới, 2 test phân tích Task 1 cũ và 3 test support đều passed. Đo line coverage module mới bằng stdlib trace/dis đạt 197/202 dòng (97,5%); không phải branch coverage, không tính coverage của subprocess CLI. Syntax/whitespace passed; không có cấu hình type checker/linter riêng được chạy. Audit độc lập bằng hàm `audit` của `scripts/audit_report_results.py` trên riêng output mới passed: 13 artifact, 20 tổ hợp method–seed, 30.000 prediction; tính lại accuracy/macro-F1/BA và mean/sample std. Không ghi đè audit tổng ngày 16/11.

Tự rà soát theo skill `agent-self-evaluation`: chính xác 4/5 (test và audit khớp, chưa đo branch coverage); đầy đủ 4/5 (đủ CLI/bảng/checksum, PR xuất TSV chưa có hình); rõ ràng 4/5 (phân biệt selection/audit, cần đọc bảng khi so nhiều target); khả thi 4/5 (lệnh CPU đã chạy thành công, yêu cầu đủ posterior source); súc tích 4/5 (logic tách khỏi runner, tài liệu cần chi tiết để ngăn diễn giải nhầm). Trung bình 4,0/5. Ưu tiên cải thiện tiếp nếu cần: biểu đồ PR từ TSV và kiểm định ghép cặp có đơn vị resampling rõ; không tự dùng audit để chọn thêm tham số. Người dùng có thể đối chiếu các nhận định qua test và artifact, đây không phải chứng nhận chất lượng ngoài dev.
