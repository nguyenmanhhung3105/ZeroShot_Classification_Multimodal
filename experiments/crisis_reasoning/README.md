# Hướng hoàn thiện trong hai tuần: hiểu ảnh và kiểm chứng chọn lọc

## Kết luận nghiên cứu và dữ liệu

Đã kiểm tra schema và đúng năm dòng đầu của mỗi TSV chính CrisisMMD, Fakeddit,
MM-IMDb bằng `csv.DictReader` + `islice`, không nạp toàn file rồi head. Không đọc
raw. Có xem ảnh processed `910691545577345026_0.jpg`: cảnh phố thông thường,
trong khi tweet nói về bão phá nhà; nhãn ảnh là not_informative/not_humanitarian.
Một ảnh không đại diện cho phân bố toàn bộ dữ liệu.

- CrisisMMD dùng nhãn ảnh, có nhãn ảnh/text riêng và bất đồng. Đưa tweet vào
  không điều kiện có thể đưa bằng chứng không thuộc target. Giữ image-only khi
  đánh giá ứng viên mới; không thay taxonomy tám lớp.
- Fakeddit cần phân biệt châm biếm, giả kết nối và sai sự thật; cosine với từ
  "True" không phải kiểm chứng thông tin. VLM cũng không có khả năng xác minh
  nguồn chỉ từ một ảnh/title. Chưa mở rộng runner này sang Fakeddit.
- MM-IMDb là 23 nhãn đồng thời, plot dài; không ép argmax một nhãn. Đây là test
  nguồn theo tài liệu cohort, không dùng để tìm ngưỡng/phương pháp mới.

Cohort CrisisMMD 3.000 hiện tại lấy từ TRAIN nguồn, đã được xem kết quả nhiều
lần. Chỉ gọi là phát triển/thăm dò. Cần tập processed đánh giá độc lập, kiểm tra
trùng tweet/ảnh, trước khi tuyên bố cải thiện cuối. Không chia lại cohort đã dùng
chọn phương pháp rồi coi một phần đó là test chưa từng được sử dụng.

Nguồn dataset: https://crisisnlp.qcri.org/crisismmd ; nguồn cohort local:
`docs/experiments/du_lieu_chinh_3000.md`.

## Những hướng đã rà và quyết định

| Nghiên cứu/repo | Khả năng áp dụng | Quyết định |
|---|---|---|
| WaffleCLIP, https://github.com/ExplainableML/WaffleCLIP | Tạo nhiều biểu diễn tên lớp | Đã có kết quả; không tiếp tục mở rộng seed trước khi có hướng tốt hơn |
| Tip-Adapter, https://github.com/gaopengcuhk/Tip-Adapter | Cache ảnh có nhãn, few-shot | Có thể mạnh nhưng thay setting zero-shot; chỉ dùng nếu chốt cho phép support có nhãn |
| Transductive CLIP, https://github.com/SegoleneMartin/transductive-CLIP | Suy luận chung minibatch không nhãn | Khác giao thức độc lập từng mẫu, cần kiểm tra batch và mất cân bằng; không ưu tiên trong thời hạn này |
| Zero-Shot Social Media Crisis Classification, https://www.mdpi.com/2076-3417/16/5/2192 ; repo https://github.com/nored/multimodal-disaster-classification | VLM sinh phân loại CrisisMMD có JSON | Phù hợp vấn đề; repo Mistral 24B báo cần A100, không sao chép thiết lập nặng lên T4 |
| Qwen2.5-VL, https://arxiv.org/abs/2502.13923 ; https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | Hiểu ảnh, đọc chữ trong ảnh, sinh nhãn/bằng chứng | Triển khai ứng viên 3B, FP16, batch 1, giới hạn 512 visual tokens; cần smoke GPU thật |

Model card và API: https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
Repo Qwen2.5-VL hiện redirect sang Qwen3-VL; runner này khóa tên Qwen2.5-VL và
lưu commit checkpoint thực tế. Không dùng mã Qwen3 thay thế tùy tiện.

Không so trực tiếp điểm trong paper khác với điểm local: khác cohort, taxonomy,
modalities, F1 weighted/binary/macro và setting huấn luyện. Không hứa Qwen3B sẽ
đạt điểm của Mistral24B.

## Kết quả MetaCLIP2 đã xác thực

Đã tải và kiểm tra manifest, cohort, điểm cosine/argmax và đủ 3.000 dự đoán:

| Task | CLIP Accuracy / Macro-F1 | Waffle seed 0 Accuracy / Macro-F1 |
|---|---|---|
| Informativeness | 42,07 / 37,58 | 47,37 / 47,30 |
| Humanitarian | 41,17 / 29,79 | 37,67 / 27,48 |

Waffle info sửa đúng 485, làm sai 326; humanitarian sửa đúng 176, làm sai 281.
Phân tích ghép cặp lưu `results/analysis/metaclip2_waffle_research_20261028/analysis.json`.
CI bootstrap là thăm dò, theo từng mẫu, chưa xét phụ thuộc sự kiện/tweet.

Đã thử đối chứng phân cấp bằng cách lấy tổng xác suất informative/not_informative
để phân lại hai nhóm humanitarian. Trên MetaCLIP2 không nhiễu, Macro-F1 giảm
29,79 xuống 10,45, dù nhãn thật giữa hai task không vi phạm mapping trên cohort.
Không đưa công thức này vào pipeline chính. Softmax chưa hiệu chuẩn và task
informativeness yếu có thể phá classifier tám lớp. Lưu kết quả âm:
`results/analysis/crisis_hierarchy/db2048ad2d58/analysis.json`.

## Phương án đã triển khai

1. **Qwen direct:** ảnh → VLM sinh → JSON nhãn + bằng chứng. Đây là cách suy luận
   khác cosine retrieval, nhưng tự nó là baseline đã có tiền lệ, không phải
   phương pháp mới. Prompt không chứa tweet, event, tên file hay nhãn annotation.
   Sinh greedy; không tin confidence tự khai. Output không parse được tính sai,
   không bỏ mẫu, có `invalid_count`. Lưu revision model, prompt, cohort, latency.
2. **Cascade đối chứng offline:** primary CLIP và contrast Waffle bất đồng thì
   nhận nhãn Qwen; đồng ý thì giữ primary. Qwen invalid thì giữ primary và log.
   Quyết định routing không nhận y_true, không chọn ngưỡng bằng nhãn test.
   `cascade.py` so primary, Qwen toàn bộ và cascade trên cùng ảnh; báo fixed,
   broken, Macro-F1 và coverage. Cho phép reviewer chỉ có prefix 5/100 ảnh, nhưng
   mọi metric được tính lại trên prefix đó, không so 100 với 3.000 ảnh.

Hiện cascade là phân tích offline từ các dự đoán thật đã lưu, **chưa phải runner
end-to-end chỉ gọi Qwen trên mẫu cần chuyển**. Coverage chỉ là số lần gọi dự kiến;
không nhận là đã giảm runtime GPU. Chỉ triển khai online nếu kết quả đối chứng
cho thấy có lợi. Khi chạy online phải giữ nguyên quy tắc, đo cả chi phí hai nhánh
CLIP và Qwen; bỏ qua chi phí contrast sẽ làm báo cáo sai.

Cascade/routing đã có trong nghiên cứu, ví dụ Gatekeeper:
https://openreview.net/pdf?id=qYI4fw3g4v . Không tuyên bố ý tưởng cascade là mới.
Đóng góp khả thi cho môn học: thiết kế đánh giá theo nhãn ảnh, kiểm tra mâu thuẫn
modality, phân tích tính ổn định prompt và hiệu quả/chí phí của kiểm chứng chọn
lọc trên dữ liệu khủng hoảng. Tính mới toàn cầu và hiệu quả chưa được chứng minh.

## Chạy ngay: một ứng viên, 5 ảnh

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/crisis_reasoning/launcher.yaml --job qwen_info --max-samples 5 --use-kaggle true
```

Không thêm `--model`, `--method`, `--backbone` hoặc `--seed`: model và phương
pháp đã cố định trong job. Cùng kernel Kaggle nên chạy lần lượt. Nếu lỗi mạng sau
khi gửi job, dùng resume-stage đã in, không gửi lại.

Nếu inference/JSON thành công, chạy 100 ảnh mỗi task để đo tốc độ và lỗi:

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/crisis_reasoning/launcher.yaml --job qwen_info --max-samples 100 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/crisis_reasoning/launcher.yaml --job qwen_humanitarian --max-samples 100 --use-kaggle true
```

Chưa gọi 100 ảnh đầu là tập đại diện hoặc test độc lập, chưa chọn phương pháp
cuối chỉ bằng chúng. Dùng thời gian đo để quyết định có đủ quota chạy 3.000 không;
thay 100 bằng 3000 sau khi kiểm tra, không tự chạy lớn. Qwen3B FP16 có thể vẫn
gặp OOM hoặc vấn đề môi trường trên T4; chỉ smoke GPU thật mới xác nhận được.

Sau khi có Qwen, đối chiếu offline (thay PATH_REVIEW bằng folder tải từ Kaggle):

```sh
venv/bin/python -B experiments/crisis_reasoning/cascade.py --primary results/waffle_lab_clip/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_0e25ae00b8d1 --contrast results/waffle_lab_waffle/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_b233ab362e32 --reviewer PATH_REVIEW
```

Kết quả Qwen: `results/qwen_direct_image/crisismmd/<task>/samples_N/qwen2_5_vl_3b/`.

## Mốc công việc để kịp nộp

- Ngày 1–2: smoke Qwen, thử 100 ảnh, kiểm tra invalid và runtime; chốt target,
  cohort và bảng metric. Nếu Qwen không phù hợp thì dừng sớm, giữ hệ thống tốt nhất.
- Ngày 3–5: đối chứng direct và cascade trên cùng mẫu; chọn một phương án trên
  tập phát triển, không sửa nhiều yếu tố cùng lúc. So cả hệ thống ảnh–tweet đang
  mạnh nhất, không chỉ thắng prompt tên lớp yếu rồi gọi là cải tiến toàn dự án.
- Ngày 6–9: khóa cấu hình, đánh giá processed độc lập chưa dùng chọn phương pháp;
  giữ nhãn ảnh/tám lớp. Nếu chưa có tập đó, ghi rõ giới hạn và chuẩn bị trước.
- Ngày 10–14: hoàn thiện demo và báo cáo, bảng ablation, lỗi từng lớp, runtime,
  kết quả âm; đóng băng code, không tiếp tục mở thêm model không có đối chứng.

Không bắt buộc thêm thuật toán chỉ để gọi là mới. Nếu cascade kém direct hoặc
baseline tốt nhất, loại nó và trình bày bằng chứng; hệ thống cuối chọn theo chất
lượng/chi phí có kiểm chứng. Ba dataset có thể giữ trong benchmark và demo,
trong khi phần cải tiến sâu tập trung CrisisMMD để kịp thời hạn.
