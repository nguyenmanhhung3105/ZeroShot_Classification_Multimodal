# Chẩn đoán nhánh text Fakeddit: trạng thái thăm dò

Thứ tự đã khóa: (1) kiểm tra nhánh text trên dev độc lập, (2) M1 với pool tham chiếu không nhãn, (3) M2 cho Fakeddit 6way, (4) M3 cho MM-IMDb. Không chọn dấu, prompt, trọng số hoặc ngưỡng bằng nhãn test. Các kết quả dưới đây chỉ giúp tìm nguyên nhân; chúng **không** hoàn thành bước 1 vì split của log cũ chưa được xác minh.

## Những gì code và log hiện tại xác nhận

- Fakeddit 2way giữ `0=fake, 1=real`; 6way giữ `0=True` và `1..5` là các loại fake. `src/run/run_fakeddit.py` dựng cùng một `label_order` cho nhánh ảnh, nhánh text và cột điểm AUROC. Nhánh 2way đưa `p[:, label_order.index(positive_label)]` vào AUROC; với nhãn dương 1, đó là điểm real.
- `src/inference.py` tính cosine ảnh–prompt và text–prompt, softmax riêng từng nhánh rồi trộn. `score_text_1` trong log là điểm softmax của real, không phải cosine thô hay xác suất đúng đã hiệu chuẩn. AUROC nhánh text không phụ thuộc ngưỡng dự đoán.
- Ở đường chạy **single-label hiện tại**, SigLIP2 cũng đi qua softmax của `src/inference.py`; `logit_bias` giống nhau cho mọi lớp nên triệt tiêu trong softmax. Nhận xét rằng ensemble Fakeddit này đang trộn đầu ra sigmoid của SigLIP2 với softmax của CLIP không đúng với code hiện tại. Sự khác nhau về độ sắc/phẳng của điểm nhánh và model vẫn cần đo, nhưng không nên chẩn đoán sai cơ chế tính điểm.
- Bộ log 2way ba model và log 6way dùng cùng 1.000 ID, tiêu đề và ảnh. Manifest khai báo `split=val` nhưng `split_verified=false`. Không có bằng chứng đây là dev độc lập với tập test từng dùng để lựa chọn pipeline.
- Log chỉ lưu điểm đã gộp theo lớp. Không có điểm từng prompt cho từng mẫu. Preprocessor Fakeddit chỉ xuất `id, clean_title, image_path, hasImage, 2_way_label, 6_way_label`; không xuất subreddit. Vì vậy chưa thể tính AUROC từng prompt hoặc xem ảnh hưởng subreddit từ các đầu ra sẵn có.

Chạy lại chẩn đoán đã lưu, không nạp model hoặc dữ liệu gốc:

```bash
venv/bin/python -B experiments/uncertainty/diagnose_fakeddit_text.py \
  --run-dir results/baseline/fakeddit/2way/samples_1000/multi_model/2way_d85f0353d7d3 \
  --six-way-log results/baseline/fakeddit/6way/samples_1000/metaclip2_vith14/6way_metaclip2_vith14__fakeddit__6way__v1__20261002_151925_304282/raw_predictions.tsv \
  --output results/analysis/text_diagnosis/fakeddit/2way/samples_1000/multi_model/2way_fakeddit_text_exploratory/summary.json
```

| Model | AUROC text, real dương | So với Manipulated Content | So với Misleading Content | Spearman: độ dài ký tự / điểm real |
|---|---:|---:|---:|---:|
| MetaCLIP2 | 0,355 | 0,157 | 0,639 | -0,341 |
| DFN5B | 0,404 | 0,227 | 0,641 | -0,174 |
| SigLIP2 | 0,323 | 0,141 | 0,565 | -0,321 |

Mỗi AUROC theo lớp con so sánh **True** với đúng lớp fake đó. `Manipulated Content` có 117 mẫu trong số 385 fake và là nhóm có tín hiệu ngược mạnh nhất trong bộ log này. Tương quan độ dài là mô tả gộp; chưa kiểm soát lớp con, nguồn bài hoặc kiểu viết, nên không thể diễn giải nhân quả. Ba model dùng **cùng tập mẫu**; đây không phải ba lần xác nhận độc lập.

Trong cùng log, tương quan độ dài ký tự / điểm real riêng trong lớp fake là MetaCLIP2 `-0,415`, DFN5B `-0,304`, SigLIP2 `-0,417`; riêng trong lớp real là `-0,218`, `-0,019`, `-0,176`. Các đặc trưng kiểu viết đơn giản (chữ hoa, chữ số, dấu hỏi, dấu cảm) đều bằng 0 ở mọi tiêu đề đã làm sạch trong log, nên **không thể** đánh giá kiểu viết từ log này. Cần metadata/tiêu đề nguồn phù hợp, không suy diễn từ các số 0.

## Điều kiện để hoàn thành bước 1

1. Có manifest dev và test từ split nguồn, khóa danh sách ID và xác nhận không giao nhau với nhau cũng như với tập 1.000 mẫu cũ nếu tập đó đã dùng chọn phương pháp. Nếu dev mới cũng là `fakeddit_sample.tsv` lấy từ validate gốc, cần chứng minh cụ thể không trùng theo ID, caption và ảnh.
2. Chạy từng model trên đúng dev đã khóa, cùng prompt/config. Ghi cosine text–từng prompt và chỉ số prompt trong log chẩn đoán (không thay phép tổng hợp điểm dự đoán); tính AUROC của mỗi đóng góp prompt vào margin real–fake. Ghi rõ nhãn prompt và chiều đóng góp đã định nghĩa trước từ nhãn prompt, không lật theo AUROC.
3. Có bảng metadata an toàn chứa `id, subreddit` hoặc xác nhận trường đó không có. Đo tương quan với độ dài/kiểu viết trong từng lớp con và trên toàn bộ dev; báo cỡ mẫu. Không suy nguồn từ tiêu đề.
4. Lặp lại trên split thứ hai. Dùng nhãn ở bước này chỉ để chẩn đoán; mọi quyết định có dùng nhãn để sửa điểm phải được tách thành phương pháp có giám sát.

Chưa chạy M1–M3 hoặc thay cấu hình pipeline trong khi điều kiện split ở trên chưa đạt.
