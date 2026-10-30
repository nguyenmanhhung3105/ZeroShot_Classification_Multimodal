# StatA cho cổng informativeness của CLIP hai tầng

## Trạng thái 29-10-2026

Đã chạy MetaCLIP2 thật trên đủ **2.219 ảnh dev**, run `ca2d96c17e89`.
Đã replay embedding/prototype, kiểm tra cùng cohort và tính lại metric.
StatA tăng Accuracy nhưng giảm Macro-F1; chưa chọn thay phương pháp hiện tại.

| Phương pháp, cùng 2.219 ảnh | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Hai tầng gốc | 66,20 | 48,06 |
| EM-Dirichlet cổng binary, trung bình ba seed | 69,15 | 48,71 |
| Chỉ chuẩn hóa prototype cổng, argmax độc lập | 68,32 | 48,85 |
| StatA cổng binary, trung bình ba seed | 71,17 | 47,21 |

StatA Macro-F1 giảm 0,85 điểm phần trăm so với hai tầng gốc và 1,64 điểm
so với đối chứng chuẩn hóa. Recall rescue giảm từ 76,07% xuống trung bình
46,79%; recall not_humanitarian tăng từ 66,42% lên 86,70%. Không dùng việc
Accuracy cao nhất để kết luận mọi lớp đã tốt hơn. Đối chứng chuẩn hóa là ứng
viên đơn giản để kiểm chứng tiếp, không phải kết quả test độc lập.

Kết quả/replay: `results/analysis/clip_two_stage/1989843be998/analysis.json`.
Đối chiếu từng lớp, metric và solver tác giả trên đủ 90 nhóm thật:
`results/analysis/clip_two_stage/1989843be998/stata_comparison.json`.
24 kiểm thử liên quan phân tích/replay đạt sau kiểm tra lượt full.

101 kiểm thử liên quan đạt. Đối chiếu với solver chính thức trên embedding
tổng hợp có phân bố lệch 85/15: ca hai lớp và ca tám lớp chỉ có hai lớp xuất
hiện đều có nhãn khớp, sai số posterior tối đa dưới 3e-26. Xem
`reference_verification.json`; đây là đối chiếu số, không phải benchmark ảnh thật.

Replay lượt thật giữa Kaggle và local có sai số posterior FP32 tối đa khoảng
4,7e-4 nhưng mọi nhãn cổng và nhãn cuối đều giống nhau. Solver chính thức
cũng xác nhận đúng 6.657 nhãn cổng của ba seed. Analyzer kiểm tra xác suất
hợp lệ, yêu cầu argmax khớp tuyệt đối và sai số tuyệt đối <=1e-3, ghi sai số
vào báo cáo. Không thay dự đoán hoặc metric để qua kiểm tra.

Nguồn: [Realistic Test-Time Adaptation of Vision-Language Models, CVPR 2025](https://arxiv.org/abs/2501.03729),
[repo chính thức](https://github.com/MaxZanella/StatA), commit đối chiếu
`2635f9355f9389756064becb5d9ed54c2756809c`.

## Vì sao ưu tiên hướng này

Paper đánh giá suy luận theo nhóm và thích nghi online khi số lớp thực sự
xuất hiện trong nhóm thay đổi hoặc chuỗi dữ liệu có tương quan. StatA dùng
prototype văn bản làm điểm neo cho các tham số phân phối Gaussian ảnh. Đây
là cơ sở để thử giảm sự trôi khỏi kiến thức CLIP ban đầu khi nhóm thiếu lớp.

Trong dự án, EM-Dirichlet tầng 2 làm dự đoán quá nhiều vào các lớp hiếm.
Chuyển EM-Dirichlet lên cổng binary cải thiện nhẹ metric tổng thể nhưng giảm
recall informative/cứu trợ. StatA được chọn làm giả thuyết nghiên cứu tiếp,
không suy kết quả của paper thành mức cải thiện kỳ vọng trên CrisisMMD.

## Giao thức đã khóa

- MetaCLIP2, cùng 2.219 ảnh dev và bank prompt semantic/semantic_waffle hiện có.
- Giữ đủ tám nhãn humanitarian; chỉ thay cách ra quyết định cổng binary.
- StatA dùng **embedding ảnh thật và prototype text**, không dùng vector
  xác suất hai chiều thay thế embedding gốc. Không thể tính lại đúng StatA
  chỉ từ các điểm cosine trước đây.
- Prototype StatA là trung bình các text embedding unit rồi chuẩn hóa unit.
  Có đối chứng `normalized_gate_control` để tách tác động chuẩn hóa khỏi
  tác động thích nghi StatA. So cả hai với hai tầng gốc dùng mean cosine.
- Tầng humanitarian giữ nguyên mean cosine và argmax bảy lớp dương.
- Cả ảnh bị cổng cũ chặn cũng được xử lý bởi cổng mới.
- Grouping seeds cố định [0,1,2], query size 75, không dùng nhãn để chia nhóm.
- Giữ mặc định code tác giả: alpha=1, hard beta, lambda_y_hat=1,
  lambda_laplacian=1, tau=100, kNN=3, 10 vòng cập nhật và 5 vòng assignment.
- Một lần encode ảnh. Mã hóa lại prompt informative để lấy prototype và
  kiểm tra `image_features @ mean_prototypes.T` khớp cosine gốc trước khi chạy.
- Nhãn thật chỉ dùng kiểm tra cohort và tính metric; không có support có nhãn.

Có ba điều chỉnh kỹ thuật được ghi trong diagnostics: loại láng giềng là
chính ảnh đó một cách tường minh khi embedding trùng/tie; floor covariance
và mass để tránh suy biến số; nhóm <=3 ảnh giữ prior vì không đủ ba láng
giềng. Cohort 2.219 có 29 nhóm 75 và một nhóm 44, không cần fallback cuối.
Không đổi công thức cập nhật chính; code viết từ phương trình paper và đối
chiếu solver gốc. Đây là bản thích nghi cho checkpoint/prompt/data của dự án.

## Chạy Kaggle

Lượt full ở trên đã hoàn tất; các lệnh dưới chỉ để tái lập khi cần.

Chạy smoke năm ảnh trước:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_two_stage/launcher_dev_stata.yaml \
  --job two_stage_stata_dev \
  --max-samples 5 \
  --use-kaggle true
```

Sau khi lượt smoke hoàn tất và được kiểm tra, chạy đủ dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_two_stage/launcher_dev_stata.yaml \
  --job two_stage_stata_dev \
  --use-kaggle true
```

Job dùng seed prompt 0, đồng thời tính cả ba grouping seed; không gửi ba job
cho ba cách chia nhóm. Không thêm `--method` hay `--model`. Giữ một job cùng
kernel tại một thời điểm. Nếu đã gửi mà mất kết nối, dùng resume-stage theo
thông báo launcher thay vì gửi lại. Chưa gửi job trong lượt triển khai này.

## Artifact và kiểm tra

Kết quả tách dưới `results/clip_two_stage_stata_heldout/`. Giữ predictions.tsv
baseline và thêm stata_predictions.tsv cho normalized control/ba seed.
summary.tsv có từng seed và `stata_gate_mean`, không chọn seed tốt nhất.
metrics.json có metric/fixed/broken từng phương án so với hai tầng gốc.

`stata_features.npz` lưu embedding ảnh, prototype trung bình chưa chuẩn hóa
và ID theo thứ tự (không dùng pickle), có SHA256 trong manifest. Nhờ vậy
có thể replay kiểm tra StatA hoặc thử TDA sau mà không encode lại ảnh.
`stata_diagnostics.json` lưu nguồn, tham số, grouping, nhãn cổng và posterior.

Phân tích sau khi có artifact:

```sh
venv/bin/python -B experiments/clip_two_stage/analyze.py --run-dir PATH_TO_RUN
```

Analyzer kiểm tra NPZ hash/ID, tái tính StatA, đối chiếu nhãn, grouping và
posterior, từ chối bảng dự đoán bị sửa. Các mục `ablations` có metric StatA.
CI bootstrap mặc định của analyzer vẫn chỉ so **hai tầng gốc với một tầng**;
không diễn giải CI đó thành bằng chứng StatA thắng baseline.

Ngoài Accuracy/Macro-F1, cần xem recall informative (nếu có nhãn binary thật)
và recall rescue trên humanitarian. Dev hiện không có nhãn informativeness
riêng; không tự suy nhãn binary từ nhãn humanitarian để báo kết quả task đó.

## Kiểm thử và giới hạn

Đã kiểm tra unit đầu vào, posterior, baseline chuẩn hóa, giữ argmax tầng 2,
flag/path launcher, full runner bằng backend tổng hợp, lưu features và replay
từ chối artifact sửa sai. Giữ các lượt thật và tổng hợp phân biệt trong báo cáo.

Đối chiếu số lại (cần checkout đúng commit tham chiếu):

```sh
venv/bin/python -B experiments/clip_stata/verify_reference.py \
  --reference-dir /private/tmp/vlm-stata-reference \
  --output /private/tmp/stata_reference_check.json
```

Dev đã được xem nhiều lần; cần khóa phương án trước đánh giá test cuối.
Điểm neo không bảo đảm mọi lớp đều cải thiện và không thay thế kiểm chứng
trên dữ liệu dự án. Chưa tự chuyển phương pháp chính sang StatA.
