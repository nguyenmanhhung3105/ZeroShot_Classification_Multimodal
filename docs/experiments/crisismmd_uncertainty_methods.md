# Xử lý dự đoán phân vân trên CrisisMMD

Ghi chú này chỉ xét **CrisisMMD**, tách `informativeness` (2 nhãn) và `humanitarian` (8 nhãn). Các phần 1–4 trình bày theo tiến trình thử phương pháp; phần 5 đối chiếu các job Kaggle chạy cổng margin, phần 6 lưu các đối chứng bổ sung, phần 7 ghi kết quả OpenAI CLIP và phần 8 lưu hướng khác chưa thử. MetaCLIP2 ViT-H/14 với prompt hiện tại là mốc đối chứng. Ngưỡng cố định của các phép thử là margin top-1 trừ top-2 của điểm cuối MetaCLIP2 `<= 0,10`; khi dùng Meta+DCLIP làm nhánh chính, margin lấy từ điểm cuối của nhánh đó.

**Trình tự cải tiến được khảo sát:** Meta với prompt hiện tại → Meta+DCLIP → DFN5B/SigLIP2 cũng dùng DCLIP → chỉ bỏ phiếu ba model ở mẫu Meta+DCLIP có margin thấp. Bỏ phiếu từ prompt hiện tại và cổng đổi prompt là các đối chứng; phép bỏ phiếu ba model+DCLIP mới được tính offline từ các lượt Kaggle riêng, chưa có job Kaggle chạy trực tiếp quy tắc đó.

## 1. Đối chứng: bỏ phiếu 2/3 với prompt hiện tại

Với mẫu có margin `<= 0,10`, MetaCLIP2, DFN5B và SigLIP2 mỗi model có **một phiếu**; nhãn được ít nhất hai phiếu sẽ là kết quả cuối. Nếu không có đa số (có thể xảy ra với 8 nhãn), giữ Meta. Với mẫu có margin `> 0,10`, luôn giữ Meta. Đây là `aux_consensus_low_margin` trong [script đối chiếu](../../experiments/uncertainty/compare_consensus_gate.py), **không phải** trung bình điểm softmax và **không phải** bỏ phiếu trên toàn bộ mẫu.

| Task, Kaggle 1.000 mẫu | Meta Accuracy / Macro-F1 | Bỏ phiếu có cổng Accuracy / Macro-F1 | Mẫu sai → đúng / đúng → sai / sai → sai khác |
|---|---:|---:|---:|
| [`informativeness`](../../results/margin_majority/crisismmd/informativeness/samples_1000/multi_model/informativeness_19271b05a511/summary.tsv) | 0,806 / 0,806 | 0,800 / 0,800 | 21 / 27 / 0 |
| [`humanitarian`](../../results/margin_majority/crisismmd/humanitarian/samples_1000/multi_model/humanitarian_093e75429270/summary.tsv) | 0,731 / 0,456 | 0,731 / 0,482 | 18 / 18 / 6 |

Ở `informativeness`, cách này đổi 48/1.000 nhãn và giảm nhẹ cả Accuracy lẫn Macro-F1. Hai model phụ có thể cùng sai; vì vậy đồng thuận của chúng không đủ để tự động thay Meta. Ở `humanitarian`, Accuracy giữ nguyên và Macro-F1 tăng, nhưng 6 nhãn đổi vẫn sai và nhiều lớp rất hiếm. Đây là đối chứng dùng **prompt hiện tại ở cả ba model**; kết quả Kaggle chi tiết được đối chiếu cùng cổng prompt ở phần 5. [Phân tích offline ban đầu trên informativeness](../../results/analysis/consensus_gate_v1/crisismmd/informativeness/samples_1000/multi_model/informativeness_e595fbb7f79a_consensus_gate_v1/summary.json) và [log local cũ 100 mẫu humanitarian](../../results/analysis/consensus_gate_v1/crisismmd/humanitarian/samples_100/multi_model/dataset_crisismmd_vkrvk61x_consensus_gate_v1/summary.json) được giữ làm lịch sử, không dùng thay cho lượt Kaggle 1.000 mẫu.

## 2. Meta+DCLIP và cổng đổi prompt khi margin thấp

Trước tiên, dùng DCLIP adapted ở **nhánh ảnh của MetaCLIP2 trên toàn bộ mẫu**: Accuracy/Macro-F1 là **0,834 / 0,833** ở `informativeness` và **0,733 / 0,465** ở `humanitarian`, so với baseline **0,806 / 0,806** và **0,731 / 0,456**. Hai lượt này là job Kaggle [informativeness](../../results/adapted_dclip/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_2ff98e4f0871/summary.tsv) và [humanitarian](../../results/adapted_dclip/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_8688b1c0c9cf/summary.tsv).

Đối chứng tiếp theo giữ **cùng model MetaCLIP2**. Nếu margin từ prompt hiện tại `<= 0,10`, lấy dự đoán của MetaCLIP2 khi **nhánh ảnh dùng descriptor DCLIP adapted**; nếu margin cao hơn, giữ dự đoán baseline. Nhánh tweet vẫn dùng prompt hiện tại và fusion `0,5/0,5`. Descriptor là bộ cố định của dự án, **không phải** descriptor GPT-3 nguyên mẫu trong paper DCLIP. Quy tắc không đọc nhãn thật để quyết định từng mẫu.

$$
\hat y(x)=\begin{cases}
\hat y_{\mathrm{Meta+DCLIP}}(x), & \Delta_{\mathrm{Meta,current}}(x)\le 0,10,\\
\hat y_{\mathrm{Meta,current}}(x), & \Delta_{\mathrm{Meta,current}}(x)>0,10.
\end{cases}
$$

Đã ghép **toàn bộ 1.000 dòng** theo `(id, text, image_path, y_true)` và xác nhận cùng task, model, thứ tự nhãn và điểm lớp; hai lượt `informativeness` có `sample_set_id` khác nhau nhưng dãy dòng khớp. Log gốc là lượt Kaggle có `run_receipt.json` và `run_artifacts.zip`. Bảng sau là metric **hậu nghiệm của phép chọn prompt**, chưa phải một Kaggle job chạy quy tắc này:

| Task, 1.000 mẫu | Baseline Accuracy / Macro-F1 | DCLIP toàn bộ Accuracy / Macro-F1 | Chỉ chuyển khi margin thấp Accuracy / Macro-F1 | Kích hoạt; đổi nhãn; sai → đúng / đúng → sai |
|---|---:|---:|---:|---:|
| `informativeness` | 0,806 / 0,806 | **0,834 / 0,833** | **0,818 / 0,818** | 124; 42; 27 / 15 |
| `humanitarian` | 0,731 / 0,456 | 0,733 / 0,465 | **0,742 / 0,471** | 144; 53; 28 / 17; thêm 8 sai → sai khác |

Quy tắc này có **tín hiệu thăm dò tốt hơn bỏ phiếu model dùng prompt hiện tại** trên hai task, nhưng với `informativeness`, chạy DCLIP adapted trên toàn bộ mẫu còn tốt hơn cổng margin. Ở `humanitarian`, cổng có điểm cao hơn cả hai nhánh riêng trên cohort hiện có. Không xem đây là mức tăng được xác nhận trên dữ liệu mới: cùng 1.000 mẫu đã được dùng khảo sát prompt/phương pháp, và manifest ghi `split_verified=false`. Hai job Kaggle áp dụng cổng đã hoàn tất; kết quả nằm ở phần 5.

Nguồn tính toán và dự đoán từng mẫu: [informativeness](../../results/analysis/prompt_margin_gate_v1/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_e595fbb7f79a_informativeness_2ff98e4f0871_prompt_margin_gate_v1/summary.json), [humanitarian](../../results/analysis/prompt_margin_gate_v1/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_e20dbf10ac1f_humanitarian_8688b1c0c9cf_prompt_margin_gate_v1/summary.json). Script tái tính: [compare_crisismmd_prompt_gate.py](../../experiments/uncertainty/compare_crisismmd_prompt_gate.py).

## 3. Mở rộng DCLIP adapted sang DFN5B và SigLIP2

Để xem tác dụng của descriptor DCLIP có lặp lại ngoài MetaCLIP2 hay không, đã chạy **DFN5B + DCLIP** và **SigLIP2 + DCLIP** riêng từng lượt Kaggle, mỗi lượt 1.000 mẫu/task. Cả bốn giữ ảnh + tweet, fusion `0,5/0,5`, nhánh tweet và thứ tự nhãn; chỉ thay prompt nhánh ảnh. Đây là **DCLIP adapted**, không phải một backbone DCLIP mới. Baseline cùng model nằm trong archive [margin-majority informativeness](../../results/margin_majority/crisismmd/informativeness/samples_1000/multi_model/informativeness_19271b05a511/summary.tsv) và [margin-majority humanitarian](../../results/margin_majority/crisismmd/humanitarian/samples_1000/multi_model/humanitarian_093e75429270/summary.tsv).

| Model; task | Baseline Accuracy / Macro-F1 | DCLIP adapted Accuracy / Macro-F1 | Nhãn đổi; sai → đúng / đúng → sai / sai → sai khác |
|---|---:|---:|---:|
| [DFN5B; `informativeness`](../../results/adapted_dclip/crisismmd/informativeness/samples_1000/dfn5b_vith14/informativeness_b2fdb7cd9028/summary.tsv) | 0,631 / 0,602 | **0,651 / 0,636** | 166; 93 / 73 / 0 |
| [SigLIP2; `informativeness`](../../results/adapted_dclip/crisismmd/informativeness/samples_1000/siglip2_so400m/informativeness_db8dd4cf5e59/summary.tsv) | **0,704 / 0,695** | 0,679 / 0,670 | 153; 64 / 89 / 0 |
| [DFN5B; `humanitarian`](../../results/adapted_dclip/crisismmd/humanitarian/samples_1000/dfn5b_vith14/humanitarian_6be9bfd1eace/summary.tsv) | 0,707 / **0,501** | **0,716** / 0,475 | 117; 59 / 50 / 8 |
| [SigLIP2; `humanitarian`](../../results/adapted_dclip/crisismmd/humanitarian/samples_1000/siglip2_so400m/humanitarian_2e7bd2f5d771/summary.tsv) | 0,686 / 0,453 | **0,704 / 0,509** | 167; 83 / 65 / 19 |

Đã kiểm receipt, archive, 1.000 dự đoán mỗi lượt và đối chiếu toàn bộ `(id, text, image_path, y_true)` với baseline từng model; điểm nhánh tweet trùng tuyệt đối trong từng cặp. Một số `sample_set_id` khác nhau dù dãy mẫu khớp. Descriptor hash giống nhau giữa ba backbone trong cùng task. DCLIP giúp DFN5B về Accuracy ở cả hai task và giúp SigLIP2 ở `humanitarian`, nhưng làm SigLIP2 giảm trên `informativeness`; với DFN5B `humanitarian`, Accuracy tăng còn Macro-F1 giảm.

## 4. Chỉ bỏ phiếu khi Meta+DCLIP phân vân

Tái tính **offline** từ raw prediction của ba lượt DCLIP riêng cho mỗi task. MetaCLIP2 + DCLIP là nhánh chính. Khi margin top-1 trừ top-2 từ **điểm fusion của Meta+DCLIP** `<= 0,10`, mỗi model có một phiếu; nhãn có ít nhất hai phiếu được chọn. Nếu cả ba chọn ba nhãn khác nhau, giữ Meta. Ngoài cổng, giữ Meta. Không trung bình xác suất của SigLIP2 với hai CLIP.

| Task, 1.000 mẫu | Meta+DCLIP Accuracy / Macro-F1 | Bỏ phiếu có cổng Accuracy / Macro-F1 | Margin thấp; nhãn đổi; sai → đúng / đúng → sai / sai → sai khác |
|---|---:|---:|---:|
| `informativeness` | **0,834 / 0,833** | 0,829 / 0,828 | 115; 41; 18 / 23 / 0 |
| `humanitarian` | 0,733 / 0,465 | **0,748 / 0,509** | 150; 46; 27 / 12 / 7 |

Ở `humanitarian`, 10 mẫu margin thấp có ba phiếu khác nhau nên giữ Meta; F1 lớp `infrastructure_and_utility_damage` tăng từ 0,492 lên 0,547. Tuy nhiên đây là **phép ghép offline, chưa có job Kaggle riêng cho quy tắc bỏ phiếu DCLIP** và chưa phải cải thiện đã xác nhận. Cùng cohort 1.000 mẫu đã được dùng để xem các biến thể trước đó, `split_verified=false`; các lớp rất hiếm có 1–9 mẫu. Trước khi chọn phương pháp cho báo cáo kết quả cuối, cần đóng băng quy tắc và đánh giá trên tập độc lập.

## 5. Kết quả Kaggle của hai đối chứng dùng margin

Bốn job đã có `summary.tsv`, `run_receipt.json`, `run_artifacts.zip`; mọi nhánh raw prediction chứa đúng 1.000 mẫu, khớp toàn bộ `(id, text, image_path, y_true)` với baseline. `comparison.tsv` trong archive xác nhận số mẫu được cổng kích hoạt và bảng lật nhãn.

| Job; task | Baseline Accuracy / Macro-F1 | Quy tắc Accuracy / Macro-F1 | Margin thấp; nhãn đổi; sai → đúng / đúng → sai / sai → sai khác |
|---|---:|---:|---:|
| [Bỏ phiếu 2/3; `informativeness`](../../results/margin_majority/crisismmd/informativeness/samples_1000/multi_model/informativeness_19271b05a511/summary.tsv) | 0,806 / 0,806 | 0,800 / 0,800 | 124; 48; 21 / 27 / 0 |
| [Bỏ phiếu 2/3; `humanitarian`](../../results/margin_majority/crisismmd/humanitarian/samples_1000/multi_model/humanitarian_093e75429270/summary.tsv) | 0,731 / 0,456 | 0,731 / 0,482 | 144; 42; 18 / 18 / 6 |
| [Prompt DCLIP; `informativeness`](../../results/prompt_margin_gate/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_980d956edb4a/summary.tsv) | 0,806 / 0,806 | 0,818 / 0,818 | 124; 42; 27 / 15 / 0 |
| [Prompt DCLIP; `humanitarian`](../../results/prompt_margin_gate/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_2beb8e4f6241/summary.tsv) | 0,731 / 0,456 | **0,742 / 0,471** | 144; 53; 28 / 17 / 8 |

Lượt bỏ phiếu dùng prompt hiện tại ở `informativeness` thấp hơn baseline; lượt `humanitarian` giữ Accuracy và tăng Macro-F1 chủ yếu ở lớp ít mẫu. Prompt gate `informativeness` tăng so baseline nhưng thấp hơn MetaCLIP2 + DCLIP trên **toàn bộ mẫu** (0,834 / 0,833). Prompt gate `humanitarian` tăng cả Accuracy và Macro-F1 trên cohort này. Các phép khảo sát đều dùng cohort đã xem ở phần 1–2 (`split_verified=false`), chưa phải xác nhận trên test độc lập.

Các job bỏ phiếu nạp ba model lần lượt; job prompt gate nạp MetaCLIP2 rồi chạy hai bộ prompt. Bốn job DFN5B/SigLIP2 + DCLIP ở phần 3 và bốn job OpenAI CLIP ở phần 7 cũng đã hoàn tất. Các hướng ở phần 8 chưa có kết quả Kaggle.

## 6. Đối chứng bổ sung: đổi về prompt cũ và bỏ phiếu lai prompt

Thử đảo chiều cổng trên các log đã có: tính margin từ **điểm cuối DCLIP adapted**; chỉ khi margin `<=0,10` mới thay bằng dự đoán MetaCLIP2 prompt hiện tại. Đây là phép tính offline bổ sung, **khác** job `prompt_margin_gate` ở phần 5 (job đó dùng margin của prompt hiện tại để chuyển *sang* DCLIP).

| Task và số mẫu | Meta+DCLIP toàn bộ: Accuracy / Macro-F1 | Đổi về prompt hiện tại khi DCLIP margin thấp: Accuracy / Macro-F1 | Sửa đúng / làm sai so với DCLIP |
|---|---:|---:|---:|
| `informativeness`, 1.000 | 0,834 / 0,833 | 0,836 / 0,835 | 22 / 20 |
| `informativeness`, 3.000 | **0,8657 / 0,8645** | 0,8573 / 0,8567 | 49 / 74 |
| `humanitarian`, 1.000 | 0,733 / 0,465 | 0,731 / 0,469 | 22 / 24 |

Trong 3.000 mẫu `informativeness`, **1.000 mẫu đầu trùng toàn bộ và cùng thứ tự** với lượt 1.000; riêng 2.000 mẫu còn lại, cổng sửa 27 nhưng làm sai 54 mẫu, khiến Accuracy DCLIP **0,8815 → 0,8680**. Vì vậy mức tăng 0,002 trên 1.000 mẫu đầu **không giữ được** khi mở rộng; không coi cổng đảo chiều là cải thiện đáng tin. Phép này không dùng nhãn để quyết định từng mẫu, nhưng việc khảo sát nó sau khi đã xem các metric trước đó vẫn là thăm dò hậu nghiệm.

Thử thêm bỏ phiếu 2/3 khi **DCLIP** có margin thấp ở `informativeness` 1.000 mẫu: Meta+DCLIP là phiếu chính, DFN5B và SigLIP2 lấy từ lượt **prompt baseline** trên đúng cùng mẫu. Accuracy/Macro-F1 **0,834 / 0,833 → 0,831 / 0,831**; sửa 22 nhưng làm sai 25 mẫu. Đây là đối chứng **lai prompt**, tách khỏi phép bỏ phiếu ba model đều dùng DCLIP ở phần 4.

Nguồn log ghép cặp: [baseline 3.000](../../results/baseline/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_6acc864b7851/summary.tsv), [DCLIP 3.000](../../results/adapted_dclip/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_03f1882357bb/summary.tsv), [baseline 1.000 ba model](../../results/baseline/crisismmd/informativeness/samples_1000/multi_model/informativeness_e595fbb7f79a/summary.tsv) và hai cặp 1.000 ở phần trên. Các con số cổng được tái tính từ `raw_predictions.tsv` trong archive của các lượt đó; **chưa có Kaggle job nào áp dụng cổng đảo chiều hoặc bỏ phiếu lai prompt**.

## 7. Kết quả OpenAI CLIP ViT-B/32 trong cùng pipeline fusion

OpenAI CLIP ViT-B/32 đã có kết quả **chỉ ảnh** ở nhóm upstream. Bốn lượt mới chạy model này trong **cùng pipeline ảnh + tweet của dự án**: cùng 1.000 mẫu/task, nhãn và prompt hiện tại của dự án, fusion ảnh–tweet `0,5/0,5`, quan hệ ảnh–tweet tắt. `openai_clip_vit_b_32` dùng checkpoint `ViT-B-32`/`openai` qua wrapper OpenCLIP; [danh mục pretrained chính thức](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py) xác nhận tag này. Lượt upstream trước dùng API OpenAI CLIP và prompt của tác giả, nên **không ghép điểm upstream image-only vào phiếu fusion**.

Bốn lượt Kaggle đã hoàn tất, mỗi lệnh một phương pháp/task. Dòng baseline kiểm tác dụng của backbone với prompt hiện tại; dòng DCLIP chỉ thay prompt nhánh ảnh của **cùng OpenAI CLIP**. Lệnh đã dùng:

```bash
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info --model openai_clip_vit_b_32 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job adapted_crisismmd_info --model openai_clip_vit_b_32 --method dclip --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job crisismmd_humanitarian --model openai_clip_vit_b_32 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job adapted_crisismmd_humanitarian --model openai_clip_vit_b_32 --method dclip --max-samples 1000 --use-kaggle true
```

**Kiểm chứng:** cả bốn lượt có receipt, archive, summary và 1.000 raw prediction; dãy `(id, text, image_path, y_true)` khớp từng dòng với nhau và với Meta+DCLIP. Các `sample_set_id` khác nhau ở một số lượt dù dãy mẫu trùng. Trong cặp OpenAI baseline/DCLIP, điểm nhánh tweet trùng từng mẫu; descriptor hash trùng các model DCLIP khác ở cùng task. Manifest ghi `split_verified=false`.

| Task | OpenAI baseline: Accuracy / Macro-F1 | OpenAI+DCLIP adapted: Accuracy / Macro-F1 | Nhãn đổi; sai → đúng / đúng → sai / sai → sai khác |
|---|---:|---:|---:|
| [`informativeness`](../../results/adapted_dclip/crisismmd/informativeness/samples_1000/openai_clip_vit_b_32/informativeness_edf63ebf9005/summary.tsv) | 0,788 / 0,775 | **0,806 / 0,797** | 92; 55 / 37 / 0 |
| [`humanitarian`](../../results/adapted_dclip/crisismmd/humanitarian/samples_1000/openai_clip_vit_b_32/humanitarian_8af695404a55/summary.tsv) | 0,693 / 0,452 | **0,740 / 0,529** | 173; 96 / 49 / 28 |

OpenAI+DCLIP thấp hơn Meta+DCLIP ở `informativeness` (0,834 / 0,833), nhưng có Macro-F1 cao hơn ở `humanitarian` (0,529 so với 0,465) trên cùng cohort. Chi tiết pipeline, các nguồn baseline và F1 theo lớp nằm tại phần I của [báo cáo tổng hợp](tong_hop_phuong_phap_crisismmd.md).

Đã tái tính **offline** hai bộ ba có OpenAI+DCLIP thay lần lượt DFN5B+DCLIP hoặc SigLIP2+DCLIP. Meta+DCLIP vẫn là phiếu chính; chỉ bỏ phiếu khi margin Meta+DCLIP `<=0,10`, ba phiếu khác nhau thì giữ Meta. Kết quả: `informativeness` **0,838 / 0,837** với OpenAI + SigLIP2 và **0,837 / 0,836** với DFN5B + OpenAI; `humanitarian` lần lượt **0,746 / 0,512** và **0,746 / 0,494**. Đây là phép phân tích từ log, **không phải** job Kaggle chạy quy tắc. Job `*_margin_majority` hiện khóa bộ ba cũ, không nhận OpenAI CLIP làm phiếu thay thế. Bảng lật nhãn và so với bộ ba cũ nằm ở phần I. Không chọn bộ ba tốt nhất trên chính 1.000 nhãn đã xem rồi gọi là kết quả xác nhận.

Nếu cần so **prompt dự án với prompt upstream trên cùng backbone chỉ ảnh**, chạy thêm hai lượt riêng dùng `--job crisismmd_info_image_only` và `--job crisismmd_humanitarian_image_only` với `--model openai_clip_vit_b_32`, rồi ghép với image-only upstream đã có. Hai lượt này là đối chứng khác, không bắt buộc trước phép thử phân vân ảnh + tweet.

## 8. Hướng tiếp theo chưa có kết quả CrisisMMD

1. **Kiểm tra nhất quán qua nhiều prompt và biến đổi ảnh nhẹ trên nhóm margin thấp.** Giữ MetaCLIP2, cố định trước các prompt tương đương theo nhãn và một số phép biến đổi ảnh vẫn giữ dấu hiệu thiên tai; chỉ xử lý lại mẫu có margin thấp. Khi nhiều góc nhìn cùng nghiêng về một nhãn, thử dùng bằng chứng này để phân biệt top-1/top-2. [Ge và cộng sự, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ge_Improving_Zero-Shot_Generalization_and_Robustness_of_Multi-Modal_Models_CVPR_2023_paper.html) dùng tính nhất quán prompt/ảnh để phát hiện dự đoán CLIP dễ sai và mô tả thêm lớp theo phân cấp để sửa mẫu khó **trong bài toán chỉ ảnh**. Chuyển sang ảnh + tweet và nhãn CrisisMMD là đề xuất thích nghi, **chưa có số liệu**; không áp dụng máy móc WordNet nếu các nhãn humanitarian không tạo thành cây phân cấp phù hợp.
2. **Làm rõ cặp nhãn top-2 ở `humanitarian`.** Ở baseline 1.000 mẫu, 169/269 lỗi top-1 có nhãn đúng đứng thứ hai. Có thể tạo mô tả đối chiếu cố định, riêng cho những cặp thường bị nhầm, rồi tính lại với MetaCLIP2 trên mẫu margin thấp. Số `169` là giới hạn cơ hội khi *biết* nhãn thật sau đánh giá, **không phải** số mẫu một quy tắc mới sẽ sửa được. Chưa có cơ chế chọn đúng lớp top-2 mà không dùng nhãn thật.
3. **Chuyển mẫu chưa đủ bằng chứng sang `needs_review`.** Margin/top1 đã xếp mẫu dễ sai tương đối tốt trên CrisisMMD, nhưng cờ này chỉ giảm rủi ro của phần máy tự quyết; nó không tự tạo ra nhãn đúng cho mọi mẫu. Báo đồng thời tỷ lệ mẫu được tự động xử lý và Accuracy/Macro-F1 trên phần đó, tránh gọi đây là tăng Accuracy toàn bộ.

Ưu tiên thử bước 1 trước vì nó thêm bằng chứng từ **cùng model và nhiều góc nhìn**, thay vì giao quyền cho hai backbone phụ đã yếu hơn trên `informativeness`. Mỗi biến thể cần một phép đối chứng cùng mẫu, cùng nhãn, và bảng sai → đúng / đúng → sai; không quét prompt hoặc ngưỡng theo nhãn của 1.000 mẫu đã xem.
