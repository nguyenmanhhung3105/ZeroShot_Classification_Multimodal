# Các phép thử hậu xử lý từ kết quả đã lưu

## Phân tích dự đoán phân vân của ba model

[`compare_consensus_gate.py`](compare_consensus_gate.py) thử quy tắc chỉ đổi nhãn MetaCLIP2 khi margin `<=0.10` và cả hai model phụ chọn cùng nhãn khác. [`diagnose_single_model.py`](diagnose_single_model.py) đo khả năng xếp mẫu sai bằng margin, entropy và bất đồng giữa ảnh/tweet. [`compare_prompt_stability.py`](compare_prompt_stability.py) đo sự đổi nhãn khi dùng prompt baseline/DCLIP adapted trên cùng model và mẫu. Ba script chỉ đọc log đã lưu, không gửi job Kaggle; kết quả và giới hạn được ghi trong [nghiên cứu dự đoán phân vân](../../docs/experiments/nghien_cuu_du_doan_phan_van.md).

[`compare_crisismmd_prompt_gate.py`](compare_crisismmd_prompt_gate.py) thử riêng trên CrisisMMD: nếu margin MetaCLIP2 `<=0.10` thì dùng nhãn từ lượt MetaCLIP2 + prompt DCLIP adapted, còn lại giữ baseline. Script yêu cầu hai archive Kaggle cùng thứ tự mẫu, ghi kết quả vào `results/analysis/prompt_margin_gate_v1/crisismmd/<task>/samples_N/<model>/`. [`run_crisismmd_prompt_gate.py`](run_crisismmd_prompt_gate.py) là runner Kaggle riêng cho cùng quy tắc, dùng một MetaCLIP2 để chạy hai prompt trên một cohort. Phương pháp, số liệu thăm dò và lệnh Kaggle **chưa chạy** nằm ở [ghi chú CrisisMMD](../../docs/experiments/crisismmd_uncertainty_methods.md).

## Fusion trọng số động không dùng nhãn thật trên CrisisMMD

`compare_crisismmd_entropy_fusion.py` dùng entropy chuẩn hóa của điểm ảnh và tweet để đặt trọng số riêng cho mỗi mẫu, rồi so với fusion cố định `0,5/0,5`. Script chỉ đọc `run_artifacts.zip`, không tải model và không dùng nhãn thật để đặt trọng số. Đây là phép kiểm tra công thức offline, **không phải lượt Kaggle**. Runner CrisisMMD có flag `fusion_mode: entropy_weighted` riêng và các job trong `configs/launcher.yaml`. Có thể kiểm tra trên log một model; với archive nhiều model, chỉ rõ `--model`:

```bash
venv/bin/python -B experiments/uncertainty/compare_crisismmd_entropy_fusion.py \
  --run-dir results/baseline/crisismmd/informativeness/samples_1000/multi_model/informativeness_e595fbb7f79a \
  --model metaclip2_vith14
```

Kết quả kiểm tra offline ở `results/analysis/entropy_weighted_fusion_v1/crisismmd/<task>/samples_N/<model>/`: `summary.json` và `predictions.tsv`. Không đưa các chỉ số offline này vào [báo cáo CrisisMMD](../../docs/experiments/tong_hop_phuong_phap_crisismmd.md) như số liệu Kaggle. Bốn job Kaggle `crisismmd_info_entropy`, `crisismmd_humanitarian_entropy`, `adapted_crisismmd_info_entropy`, `adapted_crisismmd_humanitarian_entropy` chạy riêng từng task và prompt nhánh ảnh.

## Cờ `needs_review`

`analyze_review.py` đọc `raw_predictions.tsv` bên trong `run_artifacts.zip` của một lượt single-label đã tải về. Nó không nạp model, không gọi Kaggle và không sửa nhãn dự đoán. Quy tắc thử nghiệm `branch_disagreement_v1` đánh dấu `needs_review=true` khi nhánh ảnh và nhánh text chọn khác nhãn; nếu một nhánh thiếu điểm, mẫu cũng được đánh dấu. Quy tắc **không dùng nhãn thật**.

```bash
python -B experiments/uncertainty/analyze_review.py \
  --run-dir results/adapted_dclip/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_03f1882357bb
```

Kết quả nằm tại `results/analysis/branch_disagreement_v1/<dataset>/<task>/samples_N/<model>/<task>_<run_id>_branch_disagreement_v1/`:

- `review_flags.tsv`: ID, nhãn gốc, dự đoán gốc, hai dự đoán nhánh, margin và cờ. `y_true`/`correct` chỉ dùng để phân tích hậu nghiệm.
- `summary.json`: độ phủ tự động, số lỗi và tỷ lệ lỗi của hai nhóm, tỷ lệ lỗi được cờ bắt lại.
- `by_class.tsv`: số mẫu và lỗi theo nhãn thật và nhóm.
- `analysis_manifest.json`: ID lượt nguồn, mã băm code/data, tập mẫu và phiên bản quy tắc.

Đây là phân tích thăm dò trên tập đã dùng để chọn MetaCLIP2 + DCLIP. Không diễn giải tỷ lệ lỗi của nhóm tự động thành bảo đảm trên dữ liệu mới. Trước khi bật cờ cho pipeline chính, giữ quy tắc cố định và đánh giá trên tập độc lập; nếu thêm ngưỡng margin, chọn ngưỡng chỉ trên tập hiệu chỉnh.

## Thử model phụ cho mẫu được gắn cờ

`compare_fallback.py` ghép từng mẫu theo ID, nhãn thật, caption, tên ảnh và thứ tự. Nó lấy dự đoán từ model phụ **chỉ khi** `needs_review=true`, rồi đếm số mẫu được sửa đúng (`n_fixed`) và số mẫu bị làm sai (`n_broken`). Script chỉ đọc kết quả đã lưu, không tải model hay gọi Kaggle.

```bash
python3 -B experiments/uncertainty/compare_fallback.py \
  --primary-dir results/adapted_dclip/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_2ff98e4f0871 \
  --flags-dir results/analysis/branch_disagreement_v1/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_2ff98e4f0871_branch_disagreement_v1 \
  --candidate-dir results/baseline/crisismmd/informativeness/samples_1000/multi_model/informativeness_e595fbb7f79a \
  --candidate-model dfn5b_vith14
```

Đổi `--candidate-model` thành `siglip2_so400m` để thử model còn lại trên cùng 1.000 mẫu. Kết quả ghi dưới `results/analysis/fallback/<dataset>/<task>/samples_N/<model>/`: `summary.json` gồm Accuracy/Macro-F1 trước và sau chuyển, cùng số sửa đúng/làm sai; `switches.tsv` liệt kê các mẫu đã đổi dự đoán. Các model phụ ở lượt chạy này dùng prompt gốc, nên phép thử đánh giá **cả model và cách đặt prompt**. Đây là so sánh hậu nghiệm trên tập hiện có, chưa phải kiểm chứng trên test độc lập.

## Thử tỷ trọng ảnh/text trên MetaCLIP2 + DCLIP

```bash
venv/bin/python -B experiments/uncertainty/sweep_fusion_weight.py \
  --run-dir results/adapted_dclip/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_03f1882357bb
```

Script kiểm tra rằng điểm hai nhánh tái tạo đúng dự đoán gốc, rồi thử `image_weight` từ `0` đến `1` theo bước `0.05`. Năm fold được ghép theo ID hoặc caption trùng nhau; mỗi fold chọn tỷ trọng bằng Macro-F1 trên bốn fold còn lại rồi chỉ đánh giá trên fold giữ lại. `full_cohort_sweep.tsv` là bảng **thăm dò**, không dùng để chọn tỷ trọng. `folds.tsv` và `summary.json` ghi lựa chọn và kết quả cross-fit. Không chạy inference, không đổi `experiment_config.yaml`.

Lượt 3.000 mẫu chọn `0.55` ở cả năm fold: Accuracy `0.8657 → 0.8703`, Macro-F1 `0.8645 → 0.8691`, sửa đúng `47` và làm sai `33` mẫu. Chênh lệch nhỏ (McNemar exact hai phía thăm dò `p≈0.146`). Cả 80 mẫu đổi nhãn đều có nhánh ảnh/text bất đồng và margin gốc `≤0.10`. Dataset này đã tham gia chọn model/prompt trước đó và manifest chưa xác minh split, nên đây **không phải** bằng chứng cải thiện trên test độc lập. Cần khóa `0.55` rồi chạy trên tập mới trước khi thay cấu hình chính.

## Thử conformal prediction trên Fakeddit 2way

Conformal dùng nhãn của phần **calibration** để chọn ngưỡng. Với mỗi mẫu calibration, tính `s = 1 - score_nhãn_thật`; ngưỡng là phần tử thứ `ceil((n+1)(1-alpha))` sau khi sắp tăng. Khi dự đoán, trả tập các nhãn có `1 - score_nhãn ≤ ngưỡng`. `pooled` dùng một ngưỡng chung; `classwise` dùng ngưỡng riêng từng nhãn. Không đổi trọng số model, không tăng scale và không tự gán lại nhãn cho mẫu có hai nhãn ứng viên.

```bash
venv/bin/python -B experiments/uncertainty/evaluate_conformal.py \
  --run-dir results/baseline/fakeddit/2way/samples_1000/metaclip2_vith14/2way_9ed97f55cf78 --alpha 0.10 --mode pooled
venv/bin/python -B experiments/uncertainty/evaluate_conformal.py \
  --run-dir results/baseline/fakeddit/2way/samples_1000/metaclip2_vith14/2way_9ed97f55cf78 --alpha 0.10 --mode classwise
```

Phép thử dùng 5 fold cố định, nhóm mẫu có ID/caption trùng vào cùng fold. Mỗi fold chỉ lấy nhãn thật từ 4 fold calibration, sau đó đo tập nhãn trên fold còn lại. Kết quả dưới `results/analysis/split_conformal_group5_v2/<dataset>/<task>/samples_N/<model>/<task>_<run_id>_split_conformal_group5_v2_<mode>_alpha100/` gồm `summary.json`, `folds.tsv`, `prediction_sets.tsv`. Cột `y_true` và `contains_true_label` trong output chỉ phục vụ báo cáo hậu nghiệm.

| Mode | Độ phủ chung | Độ phủ nhãn 0 / 1 | Tập một nhãn | Cần xem lại | Đúng trong tập một nhãn |
|---|---:|---:|---:|---:|---:|
| `pooled` | 90,1% | 78,7% / 97,2% | 280/1.000 | 720/1.000 | 64,6% |
| `classwise` | 90,0% | 89,9% / 90,1% | 195/1.000 | 805/1.000 | 48,7% |

Baseline dự đoán một nhãn đúng 59,1% trên cả 1.000 mẫu. `classwise` cân bằng độ phủ giữa hai nhãn nhưng giảm khả năng đưa ra một nhãn, và nhóm một nhãn còn sai nhiều. Đây là **phân tích thăm dò** trên log đã dùng để chọn pipeline; `split_verified=false`, không có test độc lập, nên không tuyên bố bảo đảm 90% cho dữ liệu mới. Không đưa conformal vào runner chính từ kết quả này. Xem [giới thiệu split conformal của Angelopoulos & Bates](https://arxiv.org/abs/2107.07511) và [class-conditional conformal prediction](https://papers.neurips.cc/paper_files/paper/2023/file/cb931eddd563f8d473c355518ce8601c-Paper-Conference.pdf).

## Hướng lưu lại, chưa triển khai: hiệu chỉnh quyết định bằng nhãn thật

Trên log MetaCLIP2 Fakeddit 2way `results/baseline/fakeddit/2way/samples_1000/metaclip2_vith14/2way_9ed97f55cf78/`, thử ngoại tuyến một bộ phân loại tuyến tính từ chênh lệch điểm của hai nhánh:

$$
z = b + \beta_{\mathrm{anh}}(p_{\mathrm{anh},1}-p_{\mathrm{anh},0})
      + \beta_{\mathrm{text}}(p_{\mathrm{text},1}-p_{\mathrm{text},0})
$$

Dự đoán nhãn 1 nếu `z >= 0`, ngược lại là nhãn 0. Các điểm `p` là đầu ra softmax của nhánh ảnh/text, không mặc nhiên là xác suất đã hiệu chuẩn. `b` và hai hệ số `β` được học từ **nhãn thật**; model VLM vẫn giữ nguyên nhưng toàn bộ pipeline vì vậy **không còn zero-shot thuần túy**.

Thử nghiệm thăm dò dùng 5 fold, mỗi fold học trên 800 mẫu và dự đoán 200 mẫu giữ lại; logistic regression với `C=1`, `class_weight=balanced`. So với baseline Accuracy `0,591`, Macro-F1 `0,519`, dự đoán ghép từ 5 fold đạt Accuracy `0,613`, Macro-F1 `0,605`, sửa đúng 204 và làm sai 182 mẫu. Nếu chỉ dùng bộ hiệu chỉnh khi margin gốc `<= 0,10`, Accuracy là `0,611`, Macro-F1 `0,559`, sửa đúng 70 và làm sai 50 mẫu. Đây là kết quả hậu nghiệm trên bộ 1.000 mẫu đã dùng khảo sát phương pháp; một fold còn giảm điểm và `split_verified=false`. Chưa có test độc lập để xác nhận lợi ích.

**Trạng thái: hoãn, không triển khai vào runner hoặc cấu hình hiện tại.** Hướng nghiên cứu của dự án ưu tiên zero-shot; chỉ cân nhắc bộ hiệu chỉnh có giám sát nếu sau này chấp nhận một nhánh thí nghiệm riêng. Khi đó cần tập hiệu chỉnh có nhãn tách khỏi tập test, khóa công thức/hệ số trước khi đánh giá và ghi rõ phương pháp không còn zero-shot thuần túy.
