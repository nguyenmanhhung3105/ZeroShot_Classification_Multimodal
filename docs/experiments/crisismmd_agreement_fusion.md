# Thử fusion ba tín hiệu cho CrisisMMD

Đây là **heuristic zero-shot thử nghiệm của dự án**, không phải phương pháp trong paper CrisisMMD. Paper/dataset gốc gán nhãn cho ảnh và tweet riêng; hai modality có thể cung cấp thông tin khác nhau. Vì vậy cosine ảnh–text **không** được gán trực tiếp cho `not_informative`, `not_humanitarian` hoặc bất kỳ lớp nhân đạo nào. Xem [nguồn CrisisNLP](https://crisisnlp.qcri.org/crisismmd.html) và [paper CrisisMMD](https://arxiv.org/abs/1805.00713).

## Quy tắc đã cố định trước khi chạy

Với mỗi mẫu, `P_img(c)` là điểm ảnh↔prompt lớp `c`, `P_text(c)` là điểm text↔prompt lớp `c`. Cả hai đã qua softmax riêng. Điểm nền giữ trọng số cấu hình `w=0,5`:

$$
B_c = w P_{img}(c) + (1-w) P_{text}(c)
$$

Nhánh thứ ba tính `r = cosine(ảnh, text)` trên embedding của cùng mẫu. `J` là Jensen–Shannon divergence đã chuẩn hóa giữa hai phân phối điểm lớp; `R` là độ tin cậy từ entropy hai nhánh, đều trong `[0,1]`. Tạo phân phối đồng thuận và hệ số điều tiết:

$$
Q_c = \frac{P_{img}(c)P_{text}(c)}{\sum_k P_{img}(k)P_{text}(k)}
$$

$$
\gamma = 0.25\,\max(0,r)(1-J)R, \qquad
P_{3}(c) = (1-\gamma)B_c + \gamma Q_c
$$

`Q` tăng tương đối các nhãn được cả hai nhánh cùng ủng hộ. Cosine ảnh–text chỉ quyết định **mức sử dụng** đồng thuận, không cộng điểm vào một nhãn cố định. Khi text rỗng, ảnh/text không có embedding hợp lệ, trọng số quan hệ bằng 0 hoặc chỉ chạy một modality, giữ nguyên dự đoán nền. Nếu tích điểm nhánh không chuẩn hóa được, cũng giữ nguyên nền. Không dùng nhãn thật để đặt `0,25` hay chọn đường đi cho từng mẫu.

Hệ số tối đa 0,25 nên đây là hiệu chỉnh có giới hạn, nhưng không bảo đảm cải thiện. `R` thấp làm thay đổi rất nhỏ trên các ca phân vân; nếu kết quả không đổi, cần báo đúng là phép thử không tạo đủ tác động, không diễn giải thành thành công. Chế độ mới chỉ cho `informativeness` và `humanitarian` của CrisisMMD; các dataset khác giữ quy tắc cũ.

## Chạy trên Kaggle, từng model/job một

Hai job standard dùng prompt hiện tại. Hai job adapted dùng descriptor DCLIP ở nhánh ảnh, cùng nhánh text của dự án. Các job đều mặc định **tắt**, chỉ chạy khi chỉ định `--job`:

```bash
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info_agreement --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job crisismmd_humanitarian_agreement --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job adapted_crisismmd_info_agreement --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job adapted_crisismmd_humanitarian_agreement --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
```

Chọn **một** lệnh để bắt đầu; không gửi bốn job cùng lúc. Có thể thêm `--dry-run` để xem kế hoạch nhưng ở chế độ `full_dataset`, launcher vẫn đóng gói dataset local trước khi dừng. Kết quả theo `results/<phương_pháp>/<dataset>/<task>/samples_N/<model>/<task>_<run_id>/`.

Sau khi launcher tải `run_artifacts.zip`, thay đường dẫn dưới đây bằng thư mục lượt vừa chạy:

```bash
venv/bin/python -B experiments/uncertainty/compare_crisismmd_agreement.py \
  --run-dir results/agreement_fusion/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_<run_id>
```

Script tái dựng `B` từ `score_image_*` và `score_text_*` **trong chính lượt chạy ba tín hiệu**, rồi xác minh `score_*` đúng công thức `P_3`. Báo Accuracy, Macro-F1, AUROC cho informativeness, số đổi nhãn, sai→đúng và đúng→sai; JSON lưu dưới `results/analysis/agreement_fusion/crisismmd/informativeness/samples_N/<model>/`. Với adapted bật quan hệ, kết quả nằm dưới `results/adapted_<method>_agreement_fusion/crisismmd/informativeness/samples_N/<model>/`. Cách đối chiếu này giữ cùng model, prompt, mẫu, thứ tự nhãn và phiên bản code; không phải chạy lại baseline trên Kaggle.

Trong log, `image_text_fusion_gate` là giá trị `γ`; `relation_applied=true` nghĩa là gate dương, **không** đồng nghĩa nhãn dự đoán đã đổi. Trường `n_changed` trong báo cáo mới đếm nhãn thực sự đổi.

Đây vẫn là phép thử trên split do cấu hình khai báo (`val`), chưa tự xác minh nguồn split/test độc lập. Chỉ kết luận hiệu quả tổng quát khi kiểm chứng trên tập độc lập đã khóa trước.

## Lượt MetaCLIP2 informativeness đầu tiên: 1.000 mẫu

Lượt `results/agreement_fusion/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_8d71e3c908af/` đã hoàn tất. Script so sánh dùng hai công thức trên **cùng log**; 1.000 ID, nhãn và text cũng khớp đúng thứ tự với baseline cũ `results/baseline/crisismmd/informativeness/samples_1000/multi_model/informativeness_e595fbb7f79a/`. Điểm baseline tái dựng từ log mới trùng chính xác điểm của lượt cũ (sai khác lớn nhất `0`).

| Chỉ số | Fusion hai tín hiệu | Fusion ba tín hiệu |
|---|---:|---:|
| Accuracy | 0,8060 | 0,8060 |
| Macro-F1 | 0,8059 | 0,8059 |
| AUROC, informative dương | 0,905551 | 0,905490 |
| Số nhãn dự đoán thay đổi | — | 0 |
| Sai→đúng / đúng→sai | — | 0 / 0 |

Gate dương ở 1.000/1.000 mẫu, nhưng trung bình chỉ `0,0117`, lớn nhất `0,0647`. Mỗi điểm lớp thay đổi trung bình khoảng `0,0010`, lớn nhất khoảng `0,00381`; **không có cải thiện phân loại** trên cohort này. Lượt chạy có `split_verified=false`, nên không dùng kết quả này để quét chọn trọng số mới trên cùng tập. [Báo cáo máy đọc](../../results/analysis/agreement_fusion/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_8d71e3c908af_metaclip2_vith14/summary.json).

## Lượt agreement humanitarian: 1.000 mẫu

Lượt Kaggle [agreement humanitarian](../../results/agreement_fusion/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_947b3970c150/summary.tsv) có receipt, archive và đủ 1.000 dự đoán. Toàn bộ `(id, text, image_path, y_true)` khớp với [baseline cùng task](../../results/baseline/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_e20dbf10ac1f/summary.tsv); `sample_set_id` khác nhưng từng dòng khớp. Accuracy **0,731 → 0,733**, Macro-F1 **0,456 → 0,462**; hai nhãn đổi đều sai → đúng, không có đúng → sai. Đây là mức tăng nhỏ trên cùng cohort đã khảo sát, chưa xác nhận trên tập độc lập.

## Thử nhánh ảnh–text tham gia trực tiếp

Chế độ `direct_pair_fusion` là phép thử riêng, không thay chế độ trên. Một cosine `r` là điểm chung của cả mẫu; cộng `r` như nhau vào mọi lớp sẽ không đổi `argmax`. Vì vậy nhánh thứ ba phải tạo **phân phối theo lớp**. Giữ `B` và `Q` ở trên, đặt `C` là số lớp, `m = (1+r)/2` (giới hạn vào `[0,1]`), rồi dùng trọng số cố định `λ=0,25`:

$$
P_{pair}(c) = (1-m)\frac{1}{C} + mQ_c
$$

$$
P_{direct}(c) = (1-\lambda)B_c + \lambda P_{pair}(c)
$$

Với `w=0,5`, ảnh↔prompt và text↔prompt mỗi nhánh đóng góp 37,5% điểm nền; nhánh ảnh↔text chiếm 25% trong công thức. Tuy nhiên phần `(1-m)/C` là điểm đều, nên phần **phân biệt lớp** của nhánh thứ ba chỉ là `λm`, không phải 25% trọn vẹn. `r` chưa được hiệu chuẩn thành xác suất khớp; đây chỉ là phép biến đổi tuyến tính cố định. `Q` cũng được suy ra từ hai nhánh đầu, nên nhánh thứ ba không cung cấp bằng chứng lớp độc lập. Nếu text thiếu, chỉ chạy một modality hoặc tích điểm không chuẩn hóa được, giữ nguyên điểm nền. Log `image_text_fusion_gate` ghi `λ` khi nhánh hoạt động, `image_text_pair_match_weight` ghi `m`; `relation_applied` chỉ báo công thức đã dùng.

Đã tái tính offline từ **cùng log** 1.000 mẫu MetaCLIP2 informativeness ở trên, không tải model hoặc dùng GPU. Nhánh thứ ba tham gia 1.000/1.000 mẫu, `m` trung bình `0,6315` (nên `λm` trung bình khoảng `0,1579`); Accuracy `0,8060 → 0,8060`, Macro-F1 `0,8059 → 0,8059`, AUROC `0,905551 → 0,905205`, và **0 nhãn đổi**. Kết quả này không cho thấy lợi ích so với baseline. [Báo cáo JSON](../../results/analysis/direct_pair_fusion/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_8d71e3c908af_metaclip2_vith14/summary.json).

Lượt Kaggle [direct-pair informativeness](../../results/direct_pair_fusion/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_14c6482fe446/summary.tsv) đã xác nhận trên đúng 1.000 dòng so với baseline: Accuracy **0,806**, Macro-F1 **0,806**, **0 nhãn đổi**. Lượt [direct-pair humanitarian](../../results/direct_pair_fusion/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_45948c7b7aa6/summary.tsv) cũng đã chạy: Accuracy **0,736**, Macro-F1 **0,444**, đổi 18 nhãn so với baseline, gồm 10 sai → đúng, 5 đúng → sai và 3 sai → sai khác. Ở task nhiều lớp, Accuracy tăng nhưng Macro-F1 giảm; không chọn phương pháp chỉ theo một chỉ số.

Với **đúng hai nhãn** và `w=0,5`, việc không đổi nhãn còn là tính chất của công thức: đặt `a=P_img(c_1)`, `b=P_text(c_1)` thì hiệu điểm giữa `c_1` và `c_2` của `B` bằng `a+b-1`, còn tử số hiệu điểm của `Q` cũng bằng `ab-(1-a)(1-b)=a+b-1`. Vì `m` và `λ` không âm, cả ba phân phối `B`, `Q`, `P_direct` luôn cùng phía của ranh giới dự đoán (trừ trường hợp hòa). Do đó cách này **không thể sửa nhãn sai** cho `informativeness` khi ảnh/text đều 50%; chỉ có thể đổi thang điểm. Task humanitarian nhiều nhãn không có bất biến đó, nhưng vẫn cần kiểm chứng trên dữ liệu riêng.

Tái tạo phép thử offline:

```bash
venv/bin/python -B experiments/uncertainty/compare_crisismmd_direct_pair.py --run-dir results/agreement_fusion/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_8d71e3c908af
```

Hai job dưới đây **đã chạy trên 1.000 mẫu**; chỉ dùng các lệnh này nếu chủ ý tái chạy trên cohort/model khác. Cả hai mặc định tắt và không chạy khi chọn job baseline:

```bash
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info_direct_pair --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job crisismmd_humanitarian_direct_pair --model metaclip2_vith14 --max-samples 1000 --use-kaggle true
```

Không cần chạy lại đúng 1.000 mẫu này chỉ để xác nhận chỉ số: đã có cả phép tính offline lẫn lượt Kaggle riêng. Với `split_verified=false`, kết quả chỉ là chẩn đoán trên cohort hiện có, không phải bằng chứng cải thiện trên test độc lập.
