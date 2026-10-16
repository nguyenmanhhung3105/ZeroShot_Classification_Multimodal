# Nghiên cứu xử lý dự đoán phân vân trong phân loại zero-shot

Tài liệu này tách hai mục tiêu: **phát hiện mẫu dễ sai** và **đổi nhãn để sửa sai**. Một phép đo bất định tốt có thể giúp chuyển mẫu sang xem xét thủ công nhưng không tự tạo ra bằng chứng để chọn nhãn đúng. Các số liệu dưới đây là **phân tích offline trên log đã lưu**, không phải lượt Kaggle mới và chưa phải kết quả xác nhận trên tập độc lập.

## 1. Cổng ensemble hiện có thực sự làm gì

Trong [`run_ensemble.py`](../../experiments/ensemble/run_ensemble.py), điểm cuối của ba model MetaCLIP2, DFN5B, SigLIP2 được trung bình đều. `margin_gate` lấy khoảng cách giữa hai điểm lớp cao nhất **của MetaCLIP2**; chỉ khi margin `<= 0.10` mới thay dự đoán MetaCLIP2 bằng dự đoán của trung bình ba model. Ngưỡng trong [`launcher.yaml`](../../configs/launcher.yaml) cũng là `0.10`.

$$
\Delta(x)=p_{\mathrm{Meta},(1)}(x)-p_{\mathrm{Meta},(2)}(x),\qquad
\bar p_c(x)=\frac{1}{3}\sum_{m=1}^{3}p_{m,c}(x)
$$

$$
\hat y(x)=\begin{cases}
\arg\max_c\bar p_c(x), & \Delta(x)\le 0.10,\\
\arg\max_c p_{\mathrm{Meta},c}(x), & \Delta(x)>0.10.
\end{cases}
$$

Vì mọi điểm softmax thuộc `[0,1]`, margin cũng thuộc `[0,1]`. Nếu dùng ngưỡng `1`, cổng sẽ bật cho **mọi mẫu**, tương đương `equal_mean`; đó không phải cấu hình đang dùng. Điểm `0,48/0,52` có margin `0,04`, nhưng `0,52` **không phải** xác suất dự đoán đúng đã được hiệu chuẩn. [Nghiên cứu hiệu chuẩn CLIP zero-shot](https://arxiv.org/abs/2303.12748) cũng phân biệt điểm softmax với xác suất đúng thực nghiệm.

## 2. Phương pháp mới thử offline: đồng thuận hai model phụ

Quy tắc `aux_consensus_low_margin` không dùng nhãn thật để quyết định từng mẫu: chỉ đổi nhãn MetaCLIP2 nếu margin `<=0.10` **và** DFN5B cùng SigLIP2 dự đoán một nhãn khác giống nhau. Nếu không, giữ nhãn MetaCLIP2. Quy tắc không học từ nhãn, không so biên độ softmax khác backbone và có thể áp dụng cho 2 hoặc nhiều lớp. `aux_consensus_all` bỏ điều kiện margin: với bài toán hai lớp, đây chính là **bỏ phiếu đa số 2/3**, mỗi model một phiếu, không trung bình điểm. Với nhiều lớp, nếu cả ba chọn ba nhãn khác nhau thì giữ nhãn MetaCLIP2. Đây là heuristic của dự án, không phải thuật toán nguyên mẫu của một paper.

$$
\hat y_{\mathrm{consensus}}(x)=\begin{cases}
\hat y_{\mathrm{DFN}}(x), & \Delta(x)\le0.10\ \land\ \hat y_{\mathrm{DFN}}(x)=\hat y_{\mathrm{SigLIP}}(x)\ne\hat y_{\mathrm{Meta}}(x),\\
\hat y_{\mathrm{Meta}}(x), & \text{các trường hợp khác}.
\end{cases}
$$

| Log nguồn; số mẫu | Phương pháp | Accuracy | Macro-F1 | Sai → đúng / đúng → sai |
|---|---|---:|---:|---:|
| Fakeddit 2way; Kaggle 1.000 | MetaCLIP2 | 0,591 | 0,519 | — |
|  | Trung bình ba model | 0,603 | 0,466 | 72 / 60 |
|  | Cổng margin + trung bình | 0,598 | 0,494 | 42 / 35 |
|  | **Bỏ phiếu đa số 2/3, mọi mẫu** | **0,603** | **0,464** | **72 / 60** |
|  | **Cổng đồng thuận mới** | **0,588** | **0,486** | **30 / 33** |
| CrisisMMD informativeness; Kaggle 1.000 | MetaCLIP2 | 0,806 | 0,806 | — |
|  | Trung bình ba model | 0,765 | 0,762 | 52 / 93 |
|  | Cổng margin + trung bình | 0,803 | 0,803 | 26 / 29 |
|  | **Bỏ phiếu đa số 2/3, mọi mẫu** | **0,719** | **0,712** | **52 / 139** |
|  | **Cổng đồng thuận mới** | **0,800** | **0,800** | **21 / 27** |
| CrisisMMD humanitarian; **log local 100 mẫu** | MetaCLIP2 | 0,570 | 0,284 | — |
|  | Trung bình ba model | 0,540 | 0,252 | 2 / 5; 1 sai → sai khác |
|  | **Bỏ phiếu đa số 2/3, mọi mẫu** | **0,560** | **0,258** | **4 / 5; 1 sai → sai khác** |
|  | **Cổng đồng thuận mới** | **0,560** | **0,267** | **2 / 3; 1 sai → sai khác** |

Cổng đồng thuận có margin **chưa vượt baseline** ở ba log này. Bỏ phiếu đa số mọi mẫu làm Accuracy Fakeddit 2way tăng 0,012 nhưng Macro-F1 giảm 0,055; trên CrisisMMD informativeness, Accuracy giảm 0,087 và Macro-F1 giảm 0,094. Lượt humanitarian 100 mẫu là log local cũ, không có receipt Kaggle; chỉ đủ kiểm tra quy tắc chạy được với 8 lớp, không dùng để kết luận phương pháp trên toàn task.

Lý do ensemble khó sửa triệt để xuất hiện ngay trong log: ở Fakeddit 2way, MetaCLIP2 sai **409** mẫu, trong đó **261** mẫu cả ba model đều sai. Ở CrisisMMD informativeness, tương ứng là **194** và **107** mẫu. Trên những mẫu cả ba cùng chọn sai một nhãn, trung bình lồi điểm hoặc bỏ phiếu ba model không thể tự chuyển sang nhãn đúng. Đây là **giới hạn theo các dự đoán hiện có**, không phải giới hạn của mọi phương pháp tương lai. Fakeddit còn có thiên lệch nhãn: ba model dự đoán `real` lần lượt **772 / 873 / 908** trên 1.000 mẫu, nên bỏ phiếu/trung bình dễ khuếch đại xu hướng đó.

Nguồn tính toán: [Fakeddit 2way](../../results/analysis/consensus_gate_v1/fakeddit/2way/samples_1000/multi_model/2way_5c026d594111_consensus_gate_v1/summary.json), [CrisisMMD informativeness](../../results/analysis/consensus_gate_v1/crisismmd/informativeness/samples_1000/multi_model/informativeness_e595fbb7f79a_consensus_gate_v1/summary.json), [CrisisMMD humanitarian local 100](../../results/analysis/consensus_gate_v1/crisismmd/humanitarian/samples_100/multi_model/dataset_crisismmd_vkrvk61x_consensus_gate_v1/summary.json). Script tái chạy: [`compare_consensus_gate.py`](../../experiments/uncertainty/compare_consensus_gate.py). Hai log 1.000 mẫu đầu có `run_receipt.json` và `run_artifacts.zip` của lượt Kaggle; mọi metric của cổng mới đều **tái tính offline** từ log ấy.

## 3. Cách nào phát hiện mẫu dễ sai tốt hơn?

Đo AUROC với mục tiêu hậu nghiệm `1` nếu MetaCLIP2 dự đoán sai; **AUROC = 0,5** là mức xếp hạng ngẫu nhiên. Điểm bất định không dùng nhãn thật để tính; nhãn chỉ dùng đánh giá khả năng xếp lỗi. Với 2 lớp, `1-top1`, entropy và `1-margin` gần như cùng một thứ tự mẫu, nên chỉ đổi tên điểm đo không tạo ra tín hiệu mới. Với 8 lớp, thứ tự có thể khác.

| Task; log Kaggle 1.000 mẫu | Tỷ lệ lỗi MetaCLIP2 | AUROC: margin thấp | AUROC: model bỏ phiếu khác | AUROC: ảnh/tweet chọn khác | AUROC: JS ảnh–tweet |
|---|---:|---:|---:|---:|---:|
| Fakeddit 2way | 40,9% | **0,544** | 0,561 | 0,529 | 0,465 |
| CrisisMMD informativeness | 19,4% | **0,794** | 0,560 | 0,709 | 0,617 |
| CrisisMMD humanitarian | 26,9% | **0,809** | Chưa có ba model Kaggle cùng mẫu | 0,715 | 0,710 |

Ở humanitarian 8 lớp, `1-top1 score` đạt AUROC **0,810**, gần bằng margin; entropy chuẩn hóa chỉ **0,765**. Nếu chỉ cho xem lại **200/1.000** mẫu có top1 score thấp nhất, nhóm đó chứa **118/269** lỗi; 800 mẫu còn lại đạt Accuracy **0,811** thay vì Accuracy toàn bộ **0,731**. Với informativeness, 200 mẫu margin thấp nhất chứa **86/194** lỗi; 800 mẫu còn lại đạt **0,865** thay vì **0,806**. Đổi lại, 200 mẫu được chuyển sang xem lại **chưa có nhãn tự động cuối cùng**: đây là cải thiện độ tin cậy của phần máy tự quyết, không phải tăng Accuracy trên đủ 1.000 mẫu.

Fakeddit khác hẳn: margin có AUROC chỉ **0,544**, 200 mẫu margin thấp nhất chứa **98/409** lỗi. Nó không chọn được nhóm lỗi đủ tốt để gọi là giải pháp cho ca khó. Cả ba model có thể tự tin và cùng sai; pipeline hiện tại chỉ so độ phù hợp với prompt, không truy xuất chứng cứ để xác minh sự thật của bài đăng. Ở humanitarian, nhãn thật nằm trong **top-2 của MetaCLIP2 ở 900/1.000 mẫu**; trong 269 lỗi top-1, **169** mẫu có nhãn đúng ở vị trí thứ hai. Đây là cơ hội cho bước phân biệt hai lớp cuối, nhưng log hiện tại **không có quy tắc không nhãn đáng tin** để biết khi nào nên chọn lớp thứ hai.

Nguồn: [Fakeddit 2way](../../results/analysis/uncertainty_diagnostic_v1/fakeddit/2way/samples_1000/metaclip2_vith14/2way_5c026d594111_uncertainty_diagnostic_v1/summary.json), [CrisisMMD informativeness](../../results/analysis/uncertainty_diagnostic_v1/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_e595fbb7f79a_uncertainty_diagnostic_v1/summary.json), [CrisisMMD humanitarian](../../results/analysis/uncertainty_diagnostic_v1/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_e20dbf10ac1f_uncertainty_diagnostic_v1/summary.json). Script: [`diagnose_single_model.py`](../../experiments/uncertainty/diagnose_single_model.py).

### 3.1. Nhạy cảm với prompt nhánh ảnh

Một phép thử bổ sung ghép **MetaCLIP2 baseline** và **MetaCLIP2 + prompt DCLIP adapted** trên đúng cùng thứ tự `(id, text, image_path, y_true)` của 1.000 mẫu mỗi task. Hai lượt `informativeness` có `sample_set_id` khác nhau trong manifest, nên phép ghép còn kiểm toàn bộ bốn trường của từng dòng; hai lượt `humanitarian` có cả `sample_set_id` và dãy dòng khớp. Đây là hai lượt Kaggle đã có, còn phép đo bất đồng là **tái tính offline**.

| Task | Số mẫu đổi nhãn khi đổi prompt | Baseline sai trong nhóm đổi nhãn | Baseline sai trong nhóm giữ nhãn | AUROC phát hiện lỗi từ cờ đổi nhãn |
|---|---:|---:|---:|---:|
| `informativeness` | 102/1.000 | 65/102 = **63,7%** | 129/898 = 14,4% | 0,645 |
| `humanitarian` | 111/1.000 | 64/111 = **57,7%** | 205/889 = 23,1% | 0,587 |

Thay prompt ở nhánh ảnh khiến cả dự đoán cuối thay đổi ở một số mẫu, và nhóm đó nhiều lỗi baseline hơn. Nhưng **bất đồng không chọn được prompt đúng**: ở informativeness, DCLIP sửa 65 mẫu và làm sai 37; ở humanitarian, sửa 49 và làm sai 47. Nếu chỉ cần Accuracy ở bộ hiện tại, DCLIP toàn tập đã có điểm riêng trong báo cáo CrisisMMD; thêm cờ bất đồng chỉ phục vụ báo mức ổn định hoặc chuyển xem lại. Phép thử này có **hai họ prompt**, chưa bao gồm biến đổi ảnh và không phải phương pháp nguyên mẫu của paper tự nhất quán.

Nguồn: [informativeness](../../results/analysis/prompt_stability_v1/crisismmd/informativeness/samples_1000/metaclip2_vith14/informativeness_e595fbb7f79a_informativeness_2ff98e4f0871_prompt_stability_v1/summary.json), [humanitarian](../../results/analysis/prompt_stability_v1/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_e20dbf10ac1f_humanitarian_8688b1c0c9cf_prompt_stability_v1/summary.json). Script: [`compare_prompt_stability.py`](../../experiments/uncertainty/compare_prompt_stability.py).

## 4. Hướng nghiên cứu tiếp theo

1. **Dùng margin/top1 để chuyển ca khó sang xem xét thủ công ở CrisisMMD.** Đây là hướng có bằng chứng thăm dò rõ nhất về **phát hiện** lỗi. Cố định ngân sách xem lại (ví dụ 20%) trước khi đánh giá trên tập độc lập; báo cả Accuracy của phần tự động, tỷ lệ mẫu được tự động xử lý và số lỗi còn lại. [Selective classification](https://arxiv.org/abs/1705.08500) nghiên cứu trực tiếp quan hệ giữa mức bao phủ và rủi ro. Không nói rằng ngưỡng hoặc mức rủi ro hiện tại được bảo đảm khi chưa hiệu chỉnh trên dữ liệu mới.
2. **Thử tính nhất quán của dự đoán dưới nhiều prompt và biến đổi ảnh nhẹ, rồi thử làm rõ top-2.** [Ge và cộng sự, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ge_Improving_Zero-Shot_Generalization_and_Robustness_of_Multi-Modal_Models_CVPR_2023_paper.html) dùng độ nhất quán qua prompt/ảnh để tìm mẫu CLIP dễ sai, sau đó làm rõ nhãn bằng thông tin thứ bậc. Paper thử phân loại **chỉ ảnh**; việc giữ thêm nhánh tweet và áp dụng cho CrisisMMD là **đề xuất thích nghi của dự án, chưa có số liệu**. Giao thức kế tiếp nên cố định trước prompt tương đương của từng lớp, số biến đổi ảnh và quy tắc chỉ xét lại khi top-1/top-2 bất ổn; chạy một model, một task/lượt. Báo sai → đúng, đúng → sai, Macro-F1 và chi phí GPU. Không dùng nhãn của 1.000 mẫu hiện tại để chọn câu prompt hoặc ngưỡng.
3. **Conformal prediction khi cần trả tập nhãn.** Đây là cách nói “chưa chắc” bằng một tập ứng viên; nó không tự sửa nhãn đơn. Bảo đảm bao phủ đòi hỏi tập calibration có nhãn và điều kiện dữ liệu phù hợp. Phép thử Fakeddit 2way đã có trong [`README uncertainty`](../../experiments/uncertainty/README.md); không nên gọi nó là zero-shot thuần túy. Nguồn phương pháp: [Angelopoulos và Bates](https://arxiv.org/abs/2107.07511), [Romano, Sesia và Candès](https://papers.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html).
4. **Test-time prompt tuning** là hướng không dùng nhãn task nhưng có tối ưu prompt trên mẫu test, tốn compute và khác giao thức prompt đóng băng hiện tại. Chỉ thử sau phép tự nhất quán đơn giản, ghi rõ đây là *test-time adaptation*. Nguồn: [Shu và cộng sự, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html).

Không nên bật cổng đồng thuận mới vào runner chính từ các log này. Ưu tiên ngắn hạn là **phát hiện và báo cáo đúng mức phân vân của CrisisMMD**, sau đó thử một nguồn tín hiệu mới ở bước top-2. Với Fakeddit 2way, chỉ đổi công thức trộn ba model hiện tại khó cứu được nhiều mẫu vì ba model cùng sai và thiên lệch cùng hướng; cần thêm bằng chứng mới thay vì tăng scale điểm.

**Giới hạn đánh giá:** Các manifest 1.000 mẫu đều ghi `split_verified=false`, và bộ mẫu đã được dùng khảo sát nhiều ý tưởng. Việc chọn hướng sau khi nhìn các bảng là lựa chọn hậu nghiệm; điểm trên chính bộ mẫu này không xác nhận khả năng tổng quát. Lượt humanitarian ba model chỉ có 100 mẫu local và không đủ lớp hiếm. Chỉ bổ sung kết luận hiệu quả sau khi khóa phương pháp và đánh giá trên tập độc lập, cùng mẫu giữa các đối chứng.
