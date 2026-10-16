# Thử ensemble khi dự đoán thiếu chắc chắn trên CrisisMMD (13/10/2026)

## Cách chạy và phạm vi

Lượt gốc dùng script độc lập `src/research/run_crisismmd_ensemble.py` (đã chuyển
sang `experiments/ensemble/run_ensemble.py`). Cách chạy lại bằng launcher hiện tại:

```bash
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info --ensemble true --max-samples 100
venv/bin/python -B scripts/launch_runs.py --job crisismmd_humanitarian --ensemble true --max-samples 100
```

Script đọc giới hạn ngay từ đầu bằng `max_samples`, chạy cả `informativeness` (2 lớp) và `humanitarian` (8 lớp) trên cùng các dòng đầu của file CrisisMMD. Ba backbone là MetaCLIP2, DFN5B và SigLIP2. Prompt, tỷ trọng ảnh/text (`0.5`), cách tính score hiện tại của mỗi backbone và cờ quan hệ ảnh–text tắt được giữ cố định. Không train, fit trọng số hay chọn ngưỡng theo nhãn thật. Trước khi ghép, script kiểm tra `sample_set_id`, `prompt_id`, thứ tự dòng, nhãn thật, ảnh, text và thứ tự lớp đều khớp.

Đặt $p_{m,c}(x)$ là **score cuối cùng** của model $m$ cho lớp $c$ sau khi model đó đã kết hợp ảnh và text. Ensemble dùng trung bình đều:

$$
\bar p_c(x) = \frac{1}{3}\sum_{m=1}^{3} p_{m,c}(x), \qquad
\hat y_{\mathrm{ens}}(x) = \arg\max_c \bar p_c(x)
$$

Cổng margin dùng MetaCLIP2 làm baseline, với ngưỡng cố định $0.10$:

$$
\Delta(x) = p_{\mathrm{Meta},(1)}(x) - p_{\mathrm{Meta},(2)}(x)
$$

$$
\hat y_{\mathrm{gate}}(x) =
\begin{cases}
\hat y_{\mathrm{ens}}(x), & \Delta(x) \le 0.10 \\
\hat y_{\mathrm{Meta}}(x), & \Delta(x) > 0.10
\end{cases}
$$

Score softmax ở đây **chưa được hiệu chuẩn** thành xác suất đúng thực nghiệm. Vì vậy `0.10` chỉ là ngưỡng minh họa đã chọn trước, không phải mức rủi ro bảo đảm. Macro-F1 tính theo toàn bộ 2 hoặc 8 lớp đã khai báo.

## Kết quả chính: 100 dòng đầu

| Tác vụ | Phương pháp | Accuracy | Macro-F1 | Sai → đúng so Meta | Đúng → sai so Meta |
|---|---|---:|---:|---:|---:|
| Informativeness | MetaCLIP2 | 0.710 | 0.708 | — | — |
| Informativeness | DFN5B | 0.620 | 0.601 | 12 | 21 |
| Informativeness | SigLIP2 | 0.690 | 0.689 | 7 | 9 |
| Informativeness | Trung bình đều | 0.710 | 0.710 | 4 | 4 |
| Informativeness | Cổng margin 0.10 | 0.700 | 0.699 | 1 | 2 |
| Humanitarian | MetaCLIP2 | 0.570 | 0.284 | — | — |
| Humanitarian | DFN5B | 0.550 | 0.370 | 10 | 12 |
| Humanitarian | SigLIP2 | 0.540 | 0.253 | 4 | 7 |
| Humanitarian | Trung bình đều | 0.540 | 0.252 | 2 | 5 |
| Humanitarian | Cổng margin 0.10 | 0.560 | 0.267 | 2 | 3 |

Với MetaCLIP2, 10/100 mẫu informativeness và 17/100 mẫu humanitarian có margin $\le 0.10$. Trung bình đều chỉ tăng Macro-F1 informativeness khoảng `0.002`, trong khi số mẫu được sửa bằng số mẫu bị làm sai. Trên humanitarian, cả trung bình đều lẫn cổng margin đều kém MetaCLIP2. DFN5B có Macro-F1 humanitarian cao hơn MetaCLIP2 trong nhóm này, nhưng accuracy thấp hơn; không thể từ đó suy ra backbone tốt hơn trên toàn bộ dữ liệu.

## Đối chiếu 30 dòng đầu

| Tác vụ | MetaCLIP2 Macro-F1 | Trung bình đều Macro-F1 | Cổng margin Macro-F1 | Trung bình đều: sai → đúng / đúng → sai |
|---|---:|---:|---:|---:|
| Informativeness | 0.603 | 0.641 | 0.576 | 2 / 2 |
| Humanitarian | 0.296 | 0.130 | 0.155 | 0 / 3 |

30 dòng đầu nằm trong 100 dòng đầu, nên đây **không phải** hai tập đánh giá độc lập. Mức tăng Macro-F1 informativeness ở 30 dòng giảm gần hết khi mở đến 100 dòng; xu hướng giảm ở humanitarian vẫn còn.

## Giới hạn và hướng kiểm tra tiếp

- 100 dòng đầu informativeness có 57 mẫu `informative`, 43 mẫu `not_informative`. Humanitarian chỉ có 5/8 lớp: 43 `not_humanitarian`, 28 `other_relevant_information`, 21 `infrastructure_and_utility_damage`, 7 `rescue_volunteering_or_donation_effort`, 1 `injured_or_dead_people`. Ba lớp vắng mặt có F1 bằng 0 trong Macro-F1 cố định 8 lớp. Không dùng kết quả này để kết luận về toàn bộ CrisisMMD.
- Log đánh giá từng nhánh của chính lượt chạy cho thấy MetaCLIP2 trên informativeness có Macro-F1 nhánh ảnh `0.788`, cao hơn score ảnh+text hiện tại `0.708`; DFN5B humanitarian có nhánh ảnh `0.454`, cao hơn ảnh+text `0.370`. Đây là quan sát hậu nghiệm, cần kiểm tra bằng tập hiệu chỉnh và tập đánh giá tách biệt trước khi đổi tỷ trọng hoặc quy tắc chọn nhánh.
- Muốn kiểm tra các phương pháp khắc phục phân vân tiếp theo, cần một bộ calibration/validation có đủ lớp hiếm, cố định ngưỡng hoặc trọng số trên đó, rồi đánh giá trên bộ khác. Với kết quả hiện tại, **không bật ensemble hay cổng margin trong pipeline chính**.

Log đầy đủ và kế hoạch đóng băng nằm tại `results/ensemble/mixed/dataset/samples_100/multi_model/dataset_crisismmd_vkrvk61x/` (100 mẫu) và `results/ensemble/mixed/dataset/samples_30/multi_model/dataset_crisismmd_vmn8yql3/` (30 mẫu). Mỗi thư mục có `frozen_plan.json`, `summary.tsv`, `details.json`, `comparison_<task>.tsv` và log từng backbone. Các file dưới `results/` là kết quả chạy cục bộ, có thể bị `.gitignore` bỏ qua.
