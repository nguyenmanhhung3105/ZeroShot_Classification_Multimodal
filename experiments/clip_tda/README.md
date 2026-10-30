# TDA cho task hai nhãn và CLIP hai tầng

## Mục tiêu

Đánh giá riêng informativeness bằng nhãn ảnh thật, rồi kiểm tra tác động của
cổng binary lên tám lớp humanitarian. Giữ ensemble 2/3 làm đối chứng hiện
tại. TDA dùng cache từ embedding và pseudo-label của CLIP, không lấy support
ground truth; hướng few-shot Tip-Adapter vẫn được lưu riêng.

Nguồn: [Efficient Test-Time Adaptation of Vision-Language Models, CVPR 2024](https://arxiv.org/abs/2403.18293),
[repo chính thức](https://github.com/kdiAAA/TDA), commit đối chiếu
`e697fb0c8078cdeff93daa56bcf8860702542069`.

## Phần đã triển khai

- TDA single-view, một ảnh mỗi bước, update cache trước khi dự đoán đúng
  thứ tự code gốc. Không coi các ảnh khác nhau là các view của cùng ảnh.
- Giữ cấu hình ImageNet tác giả: positive capacity=3, alpha=2, beta=5;
  negative capacity=2, alpha=0.117, beta=1, entropy trong (0.2,0.5), mask
  xác suất trong (0.03,1). Không tune trên CrisisMMD.
- Giữ cách tính entropy `H/log2(K)` của code gốc, dù H được tính bằng log tự
  nhiên; ghi rõ để tránh ngầm đổi ngưỡng.
- Một lượt theo thứ tự nguồn và ba lượt shuffle seed 0,1,2. Reset cả hai
  cache ở mỗi lượt và mỗi cohort. Mean chỉ tính ba shuffle, không chọn thứ
  tự tốt nhất hoặc gộp source_order vào mean.
- Chỉ thay cổng informative. Humanitarian giữ score/argmax bảy lớp dương và
  đầy đủ tám lớp đầu ra. CLIP/MetaCLIP2 frozen, không backpropagation.
- Tách binary metric và humanitarian metric. Nếu cohort không có nhãn
  informative thật, binary metric là null, không suy ra từ humanitarian.

Đối chiếu cache functions của tác giả bằng AST, bỏ import model/dataset và
chuyển CUDA/half sang CPU/float32 để phù hợp embedding đã lưu. Đây là bản
thích nghi cho MetaCLIP2/prompt/data của dự án, không tái lập toàn bộ benchmark
hoặc TDA nhiều augmented view trong một số thiết lập OOD của tác giả.

## Phát hiện riêng cho bài toán binary

Với hai lớp và cấu hình gốc, những ảnh được đưa vào negative cache có cả
hai xác suất vượt 0.03. Mask trở thành [1,1], nên negative cache trừ cùng
một lượng cho hai lớp. Khi đó nó không đổi thứ tự logit; tác động phân biệt
chủ yếu đến từ positive cache.

Đây là suy luận từ code/threshold gốc và đã được kiểm tra bằng unit test,
đối chiếu hàm cache, cùng `negative_logit_spread_max=0` trên cả bốn lượt dev
thật. Không tự tăng ngưỡng mask để làm negative cache có hiệu ứng rồi gọi
đó là TDA gốc. Nếu phát triển mask riêng cho binary, phải báo là ablation mới.

## Kết quả hiện có trên 2.219 ảnh dev

Dùng lại feature từ lượt StatA `humanitarian_ca2d96c17e89`, không chạy GPU mới.

| Phương án | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Hai tầng gốc | 66,20 | 48,06 |
| Cổng normalized CLIP | 68,32 | 48,85 |
| Ensemble 2/3 hiện tại | 69,84 | 49,24 |
| TDA, trung bình ba shuffle | 68,38 | 48,59 |

TDA theo source_order có Macro-F1 48,97%, nhưng không dùng lượt cao hơn đó
thay cho mean. Recall cứu trợ trung bình ba shuffle khoảng 69,76%, so với
67,50% của normalized control và 60,48% của majority. Đây là đánh đổi về
recall/precision, chưa có cơ sở thay ensemble theo tiêu chí Macro-F1.

Artifact: `results/analysis/clip_tda/c566420922e0/analysis.json` và `audit.json`.
Cohort này thiếu ground truth binary nên **chưa có metric informativeness
dev cho TDA**. Đầu ra tám lớp đã được tính lại trên cùng ID/thứ tự/nhãn.

## Đã kiểm tra thêm task hai nhãn bằng score có sẵn

Không cần image embedding để chuẩn hóa mean prototype: chia mean cosine
cho norm của mean unit text embeddings, đúng đại số của cosine với prototype
unit. Norm được lấy từ manifest nguồn và đối chiếu với prototype của lượt dev.

| Phương pháp trên 3.000 mẫu train có nhãn informative thật | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Mean cosine baseline | 79,93 | 79,58 |
| Normalized CLIP, argmax độc lập | 80,33 | **80,22** |
| EM-Dirichlet cũ, mean ba seed | 80,03 | 80,02 |
| Chuẩn hóa trước EM, mean ba seed | 79,64 | 79,63 |

Recall informative: baseline 89,94%; normalized CLIP 85,17%; EM cũ 80,01%.
Chọn hướng không chỉ nhìn Macro-F1: cổng có thể cân bằng hai lớp hơn nhưng
giảm khả năng phát hiện informative.

Chuẩn hóa trước EM cũng không cải thiện humanitarian dev: Macro-F1 48,19%
so với EM cũ 48,71%. Giữ làm đối chứng, chưa đưa vào ensemble.

Code: `experiments/clip_transductive/normalized_em.py`.
Artifact: `results/analysis/normalized_em_gate/a477768cac25/`.
Những số binary này là trên train đã dùng phát triển, không phải test độc lập.

## Kết quả full 3.000 mẫu có nhãn thật cho cả hai task

Lượt `humanitarian_471dc4a07814` đã hoàn tất cả export feature và hậu xử lý
TDA/ensemble. Cùng đúng 3.000 ảnh **train** như các baseline cũ; đối chiếu ID,
thứ tự, nhãn và hash ảnh đều khớp. Có 1.551 informative, 1.449 not_informative.
Hai task dùng annotation nhãn ảnh tương ứng, không suy binary từ dự đoán
humanitarian. Trong cohort này, hai annotation nhất quán về quan hệ binary.

| Cổng, cùng 3.000 train | Accuracy 2 nhãn (%) | Macro-F1 2 nhãn (%) | Accuracy 8 nhãn (%) | Macro-F1 8 nhãn (%) |
|---|---:|---:|---:|---:|
| CLIP gốc / hai tầng gốc | 79,93 | 79,58 | 68,03 | **49,88** |
| Normalized CLIP | 80,33 | 80,22 | 69,40 | 49,36 |
| TDA, mean ba shuffle | 80,67 | 80,55 | 69,51 | 49,38 |
| Ensemble 2/3, mean ba seed | **80,86** | **80,83** | **70,34** | 49,34 |

Ensemble tăng binary Macro-F1 1,24 điểm phần trăm nhưng giảm humanitarian
Macro-F1 0,54 điểm trên cùng cohort. TDA tăng binary 0,96 điểm nhưng giảm
humanitarian 0,49 điểm. Không kết luận cổng tốt hơn ở binary đồng nghĩa
tốt hơn ở tám lớp.

Recall informative: baseline 89,94%, TDA 85,58%, ensemble 81,93%. Recall
cứu trợ: baseline 73,04%, TDA 66,32%, ensemble 58,81%. Nhánh not_informative
tốt hơn đóng góp lớn vào binary Macro-F1, trong khi nhiều ảnh positive bị
chặn. Macro-F1 binary cho lớp negative một nửa trọng số, còn Macro-F1 tám
lớp chỉ cho not_humanitarian một phần tám; mục tiêu đánh giá khác nhau.

Tất cả bốn stream TDA và ba seed ensemble được replay tại local: mọi nhãn,
stream position và metric khớp. Kiểm tra hash toàn bộ output/code trong
manifest TDA, ID không lặp, nhãn hai task đúng source, cả hai phép chiếu
feature/prototype tái tạo score đều đạt. Audit và bảng đối chiếu lưu tại:

- `results/analysis/clip_tda/3f926d13558f/kaggle_audit.json`
- `results/analysis/clip_tda/3f926d13558f/comparison.json`

Các số 49,88/49,34% ở đây thuộc 3.000 train; các số 48,06/49,24% trước đó
thuộc 2.219 dev. Không so chéo hai cohort để chọn phương án. Chưa có test
cuối độc lập. Hướng tiếp cần bảo vệ recall informative/các lớp dương và đo
đồng thời đầu ra tám lớp, thay vì tối ưu duy nhất binary Macro-F1.

## Job xuất feature và đánh giá hai task

Bộ 3.000 train có nhãn informative thật trước đây chỉ lưu score. TDA cần
embedding ảnh nên đã chuẩn bị job xuất feature một lần và tự chạy TDA sau
khi model được unload. Job đồng thời giữ cả nhãn informative/humanitarian
để đánh giá hai task trên cùng ảnh; tên task trong runner vẫn là humanitarian
do dùng chung runner hai tầng, không có nghĩa bỏ đánh giá binary.

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_two_stage/launcher_binary_features.yaml \
  --job binary_features_train \
  --max-samples 3000 \
  --use-kaggle true
```

Lượt full trên đã xong; lệnh này chỉ để tái lập khi cần. Cả bốn thứ tự stream
được tính trong một lượt. Nếu đã gửi rồi mất kết nối, dùng resume-stage
được launcher in ra, tránh gửi trùng.

Đầu ra tách dưới `results/clip_two_stage_features/`:

- `metrics.json` có baseline informativeness thật và baseline humanitarian.
- `informativeness_summary.tsv` ghi rõ task hai nhãn, tách summary tám lớp.
- `clip_features.npz` có unit image features, mean prototype informative và
  humanitarian cùng ID, có SHA256. Prototype được kiểm tra tái tạo đúng score.
- `tda_analysis/analysis.json` có cả binary và humanitarian TDA; source train
  được ghi rõ, không trộn với dev. `tda_summary.tsv` riêng để không xung đột
  với summary của runner chính.
- Cũng tự tái tính **ensemble majority 2/3 hiện tại** từ cùng feature rồi
  đánh giá cả binary thật và humanitarian: `reference_ensemble_mean` trong
  analysis và `reference_ensemble.json` có từng seed/prediction. Không chỉ
  đo TDA hoặc baseline khi xét task hai nhãn.
- `tda_analysis_status` trong manifest phải complete; nếu bước hậu xử lý
  lỗi, ghi failed và launcher báo lỗi, không coi TDA đã chạy thành công.

Feature có thể tái sử dụng cho các phương pháp tiếp theo, thay vì mã hóa
lại ảnh mỗi lần. Cache TDA không được mang từ tập train này sang dev trong
protocol hiện tại; mỗi cohort có cache ban đầu rỗng.

Helper đối chứng ensemble đã được kiểm tra lại trên feature dev hiện có:
tái tạo đúng toàn bộ nhãn của cả ba seed, mean Macro-F1 49,24%. Nhãn binary
dev vẫn null vì không có annotation thật. Lượt train mới đã cung cấp phép
so sánh binary cho chính ensemble hiện tại, thay vì suy từ kết quả tám lớp.

## Kiểm tra và tái lập local

87 kiểm thử liên quan đạt sau bổ sung export/postprocess: cache capacity,
causal prefix (thêm tương lai không đổi dự đoán quá khứ), reset, thứ tự đủ
ảnh, binary negative degeneracy, nhãn thật chỉ ảnh hưởng metric, null khi
không có ground truth, launcher/path, prototype/score và TDA chỉ chạy sau
khi export hoàn tất/model đã unload.

`reference_verification.json` đối chiếu source functions cho hai/tám lớp.
Trên feature dev thật, đối chiếu lại source_order đủ 2.219 ảnh với hàm cache
tác giả cho mọi nhãn giống nhau, sai số logit tối đa khoảng 1,9e-6. Audit
tính lại metric cả bốn thứ tự, kiểm tra hash, ID, nhãn và stream positions.

Chạy lại trên artifact có feature:

```sh
venv/bin/python -B experiments/clip_tda/run.py --source-run PATH_TO_FEATURE_RUN
```

Thêm `--compare-ensemble` nếu muốn tái tính ensemble 2/3 để so trên cùng
feature (job Kaggle ở trên tự bật bước này).

Đối chiếu source:

```sh
venv/bin/python -B experiments/clip_tda/verify_reference.py \
  --reference-dir /private/tmp/vlm-tda-reference \
  --output /private/tmp/tda_reference_check.json
```

Dev đã xem nhiều lần; các kết quả là thăm dò. Cần khóa phương án và kiểm tra
train+dev overlap trước đánh giá test cuối. Tăng binary Macro-F1 không tự
bảo đảm đầu ra tám lớp tăng: còn phụ thuộc loại lỗi cổng và lỗi tầng 2.
