# Tầng 1 min5: phân tích lỗi và đối chứng regularization

Nguồn baseline: `results/final/kaggle/fakeddit_fewshot_task1_min5_4051d18f8c31/run/`.
Artifact phân tích: `results/final/fewshot_task1_analysis/train_cv/dev_4700bc4c6d15/`.

Chỉ dùng cache đã xác minh checksum, không đọc ảnh/text gốc, không nạp backbone, không gửi GPU hoặc mở test. Giữ nguyên run/config gốc. Hai unit tests kiểm chứng phân tích binary theo subtype, support train-only trong CV, tie-break và source không bị thay đổi. Run phân tích mới cũng có checksum và đã tính lại 20 bộ seed metrics, mean/std.

## Lỗi hiện tại

Các tỷ lệ dưới đây là **nhận đúng Real/Fake trong từng nhóm 6way**, không phải phân loại đúng loại fake ở tầng hai.

| Nhóm thật | Số mẫu dev | Ảnh (%) | Tiêu đề (%) | Ảnh–tiêu đề (%) |
|---|---:|---:|---:|---:|
| Real | 1202 | 72,25 | 63,34 | 72,71 |
| Satire/Parody | 182 | 77,14 | 83,63 | 80,77 |
| False Connection | 572 | 57,41 | 73,71 | 71,22 |
| Imposter | 64 | 60,31 | 69,69 | 71,25 |
| Manipulated | 868 | 99,47 | 96,57 | 99,54 |
| Misleading | 112 | 85,54 | 86,61 | 88,04 |

Text bổ trợ ảnh rõ nhất ở False Connection (+13,81 điểm phần trăm) và Imposter (+10,94); nhưng nhánh kết hợp không hơn nhánh tốt nhất ở mọi nhóm. False Connection vẫn kém title-only 2,48 điểm phần trăm, Satire cũng kém title-only 2,86.

Ảnh–tiêu đề sửa trung bình 142,2 mẫu Fake mà ảnh nhận sai nhưng làm sai thêm 46,2 mẫu Fake; lợi ích ròng 96/seed. Với Real, sửa 120,6 và làm sai 115,0; lợi ích ròng chỉ 5,6/seed. Đây là mô tả thay đổi dự đoán, không phải khẳng định nguyên nhân học máy.

Có 264 mẫu bị nhánh ảnh–tiêu đề nhận sai ở cả năm seed: 151 Real, 81 False Connection, 15 Satire, 10 Imposter, 5 Misleading, 2 Manipulated. Danh sách ID/tần suất lỗi ở `persistent_errors.tsv`; không xuất ảnh hoặc tiêu đề. `subtype_detection.tsv` và `modality_changes.tsv` lưu chi tiết.

## Train-only cross-validation chọn C

Giữ nguyên features và 384 support từng seed. Chỉ thay C của nhánh ảnh–tiêu đề: grid 0,01/0,1/1/10/100, ba fold phân tầng theo nhãn 6way của support train. Chọn mean balanced accuracy binary trên các validation fold; hòa thì ưu tiên C gần 1 trong log10, sau đó C nhỏ hơn. Refit trên đúng 384 support rồi mới dùng dev đánh giá. Không fit hoặc chọn C/ngưỡng bằng nhãn dev. Vẫn C=1 balanced làm baseline; solver/tolerance/class_weight không đổi.

| Phương pháp | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---:|---:|---:|
| Baseline ảnh–tiêu đề C=1 | **81,22 ± 0,75** | **80,16 ± 0,88** | **79,81 ± 1,02** |
| C chọn bằng train-CV | 80,23 ± 0,95 | 78,98 ± 1,57 | 78,64 ± 1,98 |

C được chọn theo seed 0..4: 1 / 0,1 / 1 / 10 / 0,1. Hai seed dùng C=1 giữ đúng dự đoán baseline; ba seed còn lại giảm accuracy lần lượt 0,43 / 3,27 / 1,27 điểm phần trăm. Không có cải tiến từ đối chứng này; không thay runner gốc. Không suy ra mọi phương án regularization đều thất bại.

Design thí nghiệm đã được gợi ý sau khi nhìn lỗi dev. Dù lựa chọn C chỉ dùng train, đây vẫn là nghiên cứu phát triển có quan sát dev trước, không phải test độc lập. Không báo std là khoảng tin cậy hoặc chọn seed tốt nhất.

## Quyết định và hướng kiểm chứng tiếp

Giữ baseline ảnh–tiêu đề C=1, accuracy 81,22%, macro-F1 80,16%. Chưa chuyển tầng hai. Trước khi thêm module mới, nên kiểm chứng cách chia ngân sách support cho bài toán binary: **384 mẫu giữ nguyên tổng**, so cấu hình cũ 64 Real + 320 Fake với ứng viên 192 Real + 192 Fake, đồng thời quy định rõ cách chọn năm loại Fake. Đây là giả thuyết về phân bổ support/độ đa dạng, chưa được xác nhận là nguyên nhân lỗi hoặc cải thiện.

Không lấy tùy ý từ cache hiện tại rồi gọi là thí nghiệm trên toàn bộ train: cache là hợp các support cũ, chỉ có 289 Real trong 1.172 mẫu và phân bố đã bị điều kiện hóa bởi cách lấy support trước. Muốn kiểm chứng sạch trên pool 3.000 train min5 phải chọn support mới từ pool đầy đủ, mã hóa thêm các ID chưa có trong cache, giữ nguyên dev và seed. Cần lưu riêng kết quả và không dùng test để chọn cấu hình.

Lệnh tái hiện phân tích này (CPU, fit các linear head nhỏ, không model encoder):

```sh
venv/bin/python -B scripts/analyze_fakeddit_task1_fewshot.py \
  --run results/final/kaggle/fakeddit_fewshot_task1_min5_4051d18f8c31/run
```
