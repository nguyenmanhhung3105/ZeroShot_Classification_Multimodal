# Xử lý dự đoán thiếu chắc chắn trong dự án zero-shot

Ngày đối chiếu: **13-10-2026**. Đây là báo cáo nghiên cứu và đề xuất thí nghiệm, chưa phải thuật toán đã tích hợp hoặc kết quả cải thiện trên dataset của dự án.

## 1. Kết luận để chọn hướng nghiên cứu

Có phương pháp xử lý trường hợp `real=0.48, fake=0.52`, nhưng cần tách ba mục tiêu:

| Mục tiêu | Cách làm | Có thể đổi nhãn dự đoán? |
|---|---|---|
| Bổ sung bằng chứng để phân loại tốt hơn | Ensemble model/prompt; thích nghi khi suy luận như TDA, StatA, TPT | Có; cần đo xem đổi đúng hay đổi sai |
| Làm độ tin cậy phản ánh đúng tỷ lệ đúng | Calibration, chẳng hạn temperature scaling | Một nhiệt độ chung sau logits cuối giữ nguyên argmax |
| Xử lý ca chưa đủ bằng chứng | Selective classification; tập nhãn conformal | Có thể trì hoãn kết luận hoặc trả tập ứng viên; không tự sửa model |

**Đề xuất cho dự án:** bắt đầu bằng đo margin, entropy và bất đồng giữa nhánh ảnh/text; thử ensemble các backbone hiện có trên các mẫu thiếu chắc chắn; so sánh với một phương pháp thích nghi có công bố như TDA/StatA cho single-label. Nếu cho phép dùng nhãn ở tập calibration, bổ sung conformal cho single-label và kiểm soát rủi ro riêng cho multi-label. TPT/O-TPT là hướng tiếp theo khi chấp nhận gradient và chi phí bổ sung.

Các lựa chọn này là suy luận về độ phù hợp với code hiện tại. Chưa có chứng cứ chúng cải thiện Fakeddit, CrisisMMD hoặc MM-IMDb của bạn.

## 2. Đối chiếu pipeline đang có

| Dataset/task | Quy tắc nhãn | Điểm/decision hiện tại | Cách nhận diện phân vân |
|---|---|---|---|
| Fakeddit 2way | Một nhãn: fake/real | Softmax mỗi nhánh → fusion → argmax | Margin top-1/top-2; entropy; bất đồng |
| Fakeddit 6way | Một trong 6 loại nội dung | Như trên; relation có thể hiệu chỉnh nếu bật | Như trên, thêm phân tích cặp lớp hay nhầm |
| CrisisMMD informativeness | Một trong 2 lớp | Softmax → fusion → argmax | Như binary |
| CrisisMMD humanitarian | Một trong 8 lớp | Softmax → fusion → argmax | Như multiclass |
| MM-IMDb genres | 23 thể loại độc lập, nhiều thể loại cùng đúng | Mặc định `relative_z`, chọn điểm > 1.0 | Khoảng cách đến ngưỡng từng genre; mức ổn định từng genre |

Nguồn code: [inference.py](../src/inference.py), [logging_utils.py](../src/logging_utils.py), [evaluation_audit.py](../src/evaluation_audit.py), [cấu hình](../configs/experiment_config.yaml), [registry](../src/models/model_registry.py).

Đối với single-label, nếu relation tắt:

$$
P_I = \operatorname{softmax}(s_I S_I + b), \quad
P_T = \operatorname{softmax}(s_T S_T + b), \quad
B = wP_I + (1-w)P_T.
$$

`S_I`, `S_T` là điểm similarity với mô tả lớp. Dự đoán là `argmax(B)`. Text rỗng có fallback riêng trong code. Scale của mô hình được học cho ảnh–text; dùng nó cho text–text không chứng minh xác suất đã được calibration cho task.

Log đã có `confidence=max(B)` và `margin=top1-top2`, cùng điểm từng lớp/từng nhánh. Chữ `confidence` chỉ là tên trường log. Chưa có bước fit calibration chứng minh `0.52` tương ứng 52% dự đoán đúng.

MM-IMDb chuẩn hóa similarity qua 23 genre **trong từng mẫu, riêng từng nhánh**, rồi kết hợp. Điểm z có thể âm hoặc >1; không phải xác suất. Mode `fixed_threshold` dùng sigmoid nhưng cũng cần kiểm tra calibration; không tự đổi sang mode này để giải quyết uncertainty.

## 3. Phân vân có thể đến từ đâu?

Với đầu vào và điểm số cố định, argmax chọn `fake=0.52` một cách xác định. “Hên/xui” là cách diễn tả kết quả đúng/sai, không phải model bốc thăm. Chênh lệch nhỏ cho thấy quyết định dễ đổi khi prompt, ảnh, fusion hoặc model thay đổi.

Một điểm fusion gần 0.5 có ít nhất hai nguyên nhân khác nhau:

1. Cả ảnh và text đều gần `[0.5, 0.5]`: hai nhánh đều ít phân biệt được lớp.
2. Ảnh gần `[0.95, 0.05]`, text gần `[0.05, 0.95]`, fusion 50/50: hai nhánh tự tin nhưng đối lập.

Trường hợp 2 cần kiểm tra xung đột phương thức và định nghĩa nhãn, không chỉ tăng scale. Hai nhánh cùng tự tin cũng có thể cùng sai.

Fakeddit được gán nhãn bằng distant supervision; tác giả ghi nhận khó chọn đúng một loại cho một số mẫu. Điều này hỗ trợ khả năng có ca mơ hồ về nhãn, không chỉ về model. `False Connection` xét ảnh có hỗ trợ đúng ý nghĩa caption hay không; similarity cao không chứng minh tin thật. Nguồn: [paper Fakeddit, phần 3.2–3.3](https://arxiv.org/html/1911.03854).

**Suy luận cho dự án:** thiếu nguồn sự kiện, ngữ cảnh châm biếm hoặc dấu vết sửa ảnh có thể khiến title + image chưa đủ quyết định. Không có phép biến đổi hai điểm 0.48/0.52 nào tự tạo thêm các bằng chứng đó.

## 4. Công thức đo uncertainty

### 4.1 Binary và multiclass single-label

Với phân phối điểm cuối `p`, số lớp `C`, điểm lớn nhất và nhì `p_(1), p_(2)`:

\[
\Delta(x)=p_{(1)}-p_{(2)},\qquad
H_{norm}(x)=\frac{-\sum_{c=1}^{C}p_c\log p_c}{\log C}.
\]

Quy ước `0 log 0 = 0`. Margin nhỏ: hai ứng viên gần nhau. Entropy gần 1: điểm trải rộng qua nhiều lớp. Chúng đo hai khía cạnh khác nhau; không phải xác suất model sai.

Ví dụ `[0.48,0.52]`: margin `0.04`, entropy chuẩn hóa khoảng `0.99885`. Với 6/8 lớp vẫn dùng công thức trên; không dùng ngưỡng confidence của binary làm chuẩn chung cho mọi task.

Để đo bất đồng ảnh/text, đặt `M=(P_I+P_T)/2`:

\[
J(x)=\frac{\operatorname{KL}(P_I\Vert M)+\operatorname{KL}(P_T\Vert M)}{2\log2}.
\]

`J` gần 0: hai nhánh cho phân phối giống nhau; có thể cùng phẳng. `J` lớn: bất đồng mạnh. Không diễn giải `J` thành nhãn False Connection trực tiếp.

Nếu có `K` dự đoán từ các model, prompt hoặc view đã định nghĩa trước:

\[
\bar p=\sum_{k=1}^{K}a_kp^{(k)},\quad a_k\ge0,\quad\sum_k a_k=1,
\]

\[
D=\frac{H(\bar p)-\sum_k a_kH(p^{(k)})}{\log C}.
\]

Đây là bất đồng của tập dự đoán; không tự chứng minh phân rã uncertainty theo Bayes. Cần giữ entropy của `bar p`: tất cả model cùng phẳng sẽ có `D=0` nhưng vẫn thiếu chắc chắn.

Một gate thử nghiệm có thể là `margin < δ OR entropy > h OR disagreement > d`. **Các ngưỡng chưa được lựa chọn hoặc kiểm chứng.** Gate chỉ chuyển ca khó sang bước xử lý bổ sung, không được dùng như bằng chứng rằng nhãn cần đảo.

### 4.2 Multi-label MM-IMDb

Với điểm z cuối `z_c` và ngưỡng hiện tại `τ=1.0`:

\[
d_c(x)=|z_c(x)-\tau|.
\]

`d_c` nhỏ báo genre ở gần biên quyết định. Ví dụ `Action=1.02` gần biên hơn `Action=2.0`. Hai genre có điểm cao gần nhau không phải xung đột vì cả hai có thể đúng.

Qua các view/model đã định nghĩa, có thể đo:

\[
V_c=\frac1K\sum_k\mathbf1[z_c^{(k)}>\tau],\qquad
\sigma_c^2=\frac1K\sum_k(z_c^{(k)}-\bar z_c)^2.
\]

`V_c` gần 0.5 cho thấy quyết định genre hay đổi. Đây là tỷ lệ bỏ phiếu, không phải xác suất genre đúng. Khi ensemble nhiều backbone, vẫn phải xác minh điểm/threshold có ý nghĩa tương thích; không lấy trung bình tùy ý raw cosine hoặc logit khác scale.

Nếu dùng xác suất độc lập đã được calibration, mới xét Bernoulli entropy từng genre. Không áp Shannon entropy của phân phối categorical lên vector z hoặc ép tổng 23 genre bằng 1.

## 5. Các phương pháp đã kiểm tra nguồn

### 5.1 Ensemble model/prompt: hướng ít thay đổi pipeline

[Deep Ensembles — NeurIPS 2017](https://arxiv.org/abs/1612.01474) kết hợp các mạng được huấn luyện độc lập để ước lượng uncertainty. Dùng ba backbone pretrained hiện có là một biến thể ensemble, không phải tái hiện nguyên phương pháp tác giả.

**Đề xuất riêng:** giữ MetaCLIP2 làm bước đầu; ca thiếu chắc chắn mới gọi DFN5B/SigLIP2 nếu đủ bộ nhớ và thời gian. Gộp phân phối single-label theo công thức `bar p` phía trên, với thứ tự nhãn thống nhất. Trọng số bằng nhau là baseline có thể thử, chưa phải tối ưu. So sánh với việc chạy cả ba cho mọi mẫu để đo lợi ích và chi phí của gate.

Ví dụ hoàn toàn minh họa: ba model cho điểm `real` lần lượt `.48, .62, .67`; trung bình bằng `.59`, nên quyết định có thể chuyển từ fake sang real. Đây không phải kết quả chạy của dự án. Nếu cả ba cùng hiểu sai nội dung, ensemble vẫn có thể sai.

Không chọn model có confidence cao nhất một cách mặc định: scale và calibration khác nhau. DCLIP/WaffleCLIP/prompt hiện có có thể làm các nhánh so sánh, nhưng prompt sai ngữ nghĩa giống nhau vẫn gây lỗi chung. Hiện `mean_embed/mean_sim` đã gộp prompt; muốn đo bất đồng từng prompt cần lưu thêm dự đoán riêng, không suy ra được từ một vector lớp đã gộp.

### 5.2 Nhiều view ảnh: có ích để đo ổn định, chưa chắc tăng accuracy

Phụ lục A.2 của [TPT](https://arxiv.org/html/2209.07511v1) có baseline lấy trung bình view: average accuracy `42.69`, thấp hơn baseline `44.18` trong bảng đó. Bằng chứng này bác bỏ việc mặc định augmentation averaging luôn có lợi.

**Đề xuất riêng:** nếu thử, giữ ảnh gốc và các biến đổi nhẹ bảo toàn nội dung. Crop có thể mất vùng sửa ảnh, nạn nhân hoặc chú thích; flip có thể đảo chữ. Không paraphrase title/plot tùy ý vì có thể đổi chi tiết sự kiện. Mọi phép biến đổi phải được cố định và kiểm tra cho từng task.

### 5.3 TDA: thích nghi không cần backprop, có negative cache

[TDA — CVPR 2024](https://arxiv.org/html/2403.18293v1) dùng cache feature với pseudo-label từ luồng test, gồm positive/negative cache, để hiệu chỉnh điểm CLIP; không cần gradient. Negative pseudo-label là cơ chế riêng của tác giả, không phải đảo nhãn đang có điểm thấp.

**Độ phù hợp suy luận:** ứng viên cho Fakeddit/Crisis single-label vì nhận feature và điểm lớp. Đưa vào pipeline ảnh + text của bạn là mở rộng cần kiểm thử; paper đánh giá phân loại ảnh, không xác nhận kết quả trên các task này. Không áp nguyên one-hot cache lên MM-IMDb multi-label. Cache có trạng thái, nên phải ghi thứ tự mẫu, warm-up, reset và hyperparameter.

### 5.4 StatA: cần đối chứng khi luồng không lý tưởng

[Realistic Test-Time Adaptation / StatA — CVPR 2025](https://arxiv.org/html/2501.03729) đánh giá batch thiếu lớp và stream có tương quan; chỉ ra hạn chế của TTA trong các điều kiện này, đề xuất anchor giữ tri thức text encoder.

**Độ phù hợp suy luận:** đưa StatA vào danh sách đối chứng cho TDA, đặc biệt khi stream có nhiều mẫu cùng lớp hoặc lớp hiếm. Không giả định batch có đủ 6/8 lớp hay cân bằng. Không lấy kết quả trên CLIP ViT-B/16 làm cam kết cho MetaCLIP2/DFN5B/SigLIP2.

### 5.5 TPT/O-TPT: có thể đổi quyết định nhưng cần tối ưu prompt

[TPT — NeurIPS 2022](https://arxiv.org/abs/2209.07511) tối ưu prompt bằng entropy trên các view được chọn theo confidence, không dùng nhãn test. Có gradient và tăng chi phí inference. Đây là continuous prompt tuning, khác việc thêm câu prompt bằng tay.

[O-TPT — CVPR 2025](https://arxiv.org/html/2503.12096v1) xử lý calibration trong prompt tuning bằng ràng buộc orthogonality. Nghiên cứu này ghi nhận TPT có thể tạo dự đoán quá tự tin: entropy giảm không đủ chứng minh nhãn đúng hơn.

**Độ phù hợp suy luận:** cần adapter token/prompt theo backbone và đường inference cho phép gradient; không thể chỉ bật flag trong hàm hiện tại. Với MM-IMDb, tối thiểu hóa categorical entropy sẽ xung đột với nhiều genre cùng đúng; phải thiết kế objective khác và kiểm chứng riêng.

### 5.6 Temperature scaling: hữu ích cho độ tin cậy

\[
p_T(c|x)=\frac{\exp(\ell_c/T)}{\sum_j\exp(\ell_j/T)},\quad T>0.
\]

[Guo et al. — ICML 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), phần 4.2: một nhiệt độ chung giữ thứ tự logits, nên giữ argmax. Cần tập có nhãn để fit calibration. Không thể dùng việc sharpen 0.52 thành 0.9 như bằng chứng tăng accuracy.

**Lưu ý riêng của dự án:** hiệu chỉnh hai nhánh trước fusion có thể đổi quyết định cuối vì thay đổi mức đóng góp tương đối. Hiệu chỉnh riêng lớp hoặc đổi threshold multi-label cũng có thể đổi nhãn; phải đánh giá như một thay đổi decision, không gộp vào kết luận giữ nguyên argmax ở trên.

### 5.7 Selective classification: trì hoãn ca chưa đủ chắc chắn

[Selective Classification — NeurIPS 2017](https://arxiv.org/abs/1705.08500) nghiên cứu đánh đổi giữa tỷ lệ mẫu được nhận và lỗi trên mẫu được nhận. Một reject gate dùng margin là baseline; không tương đương thuật toán có chứng nhận rủi ro của paper.

Trong dự án có thể giữ nhãn argmax cho benchmark, thêm trạng thái `uncertain/review_required` riêng. Nếu loại mẫu khó khỏi metric, phải báo coverage và error trên phần được nhận; không tuyên bố accuracy toàn bộ tăng từ metric của tập con.

### 5.8 Conformal cho binary/multiclass: trả tập ứng viên

Một baseline split conformal đơn giản, theo [Angelopoulos & Bates](https://arxiv.org/html/2107.07511): trên `n` mẫu calibration có nhãn, tính

\[
s_i=1-p_{y_i}(x_i),\quad k=\lceil(n+1)(1-\alpha)\rceil,
\quad q=s_{(k)},\quad\Gamma(x)=\{c:1-p_c(x)\le q\}.
\]

`s_(k)` là phần tử thứ k sau khi sắp tăng; nếu `k>n`, dùng `q=+∞`. Công thức này là threshold conformal đơn giản, **không phải APS**. [APS — NeurIPS 2020](https://arxiv.org/abs/2006.02544) dùng score theo tổng điểm tích lũy thứ hạng để tạo tập thích nghi.

Với điểm `[real=.48,fake=.52]` và **q minh họa .6**, tập ứng viên là `{real,fake}`. Với 6/8 lớp, tập có thể gồm vài lớp. Tập ứng viên không khẳng định các lớp đều đúng.

Bảo đảm dạng `P(Y ∈ Γ(X)) ≥ 1−α` là **marginal**, cần exchangeability của calibration/test và predictor phù hợp được cố định. Không suy ra từng mẫu, từng lớp hiếm hoặc riêng nhóm singleton đạt tỷ lệ đó. Temperature calibration không phải điều kiện bắt buộc để conformal có coverage. Nguồn về điều kiện và phân biệt marginal/conditional: [Romano et al., phần 1–2](https://papers.nips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf).

Tập rỗng/nhiều lớp cần chính sách xử lý. Không thêm nhãn `uncertain` vào taxonomy benchmark hoặc coi tập ứng viên là dự đoán multi-label. Với chỉ 5 mẫu calibration và `α=.1`, `k=6>5`, baseline trên trả tất cả lớp; các smoke test 5 mẫu không đủ chứng nhận hữu ích ở mức này.

### 5.9 Conformal risk control cho multi-label

Với tập genre thật `Y`, tập dự đoán `S`, chọn loss tỷ lệ bỏ sót:

\[
L_{FN}(S,Y)=\frac{|Y\setminus S|}{\max(1,|Y|)}.
\]

[Conformal Risk Control](https://arxiv.org/html/2208.02814v4) chọn ngưỡng từ calibration để kiểm soát expected loss đơn điệu. Với loss trên, mở rộng tập giảm bỏ sót nhưng có thể tăng genre sai; không bảo đảm mỗi phim không bỏ sót.

[Fisch et al. — ICML 2022](https://proceedings.mlr.press/v162/fisch22a.html) nghiên cứu conformal multi-label với số ứng viên sai được khống chế theo tolerance. Đây là hướng khi cần tránh gán quá nhiều genre thay vì chỉ tối đa recall.

**Đề xuất riêng:** giữ score z, dùng họ tập `S_t={c:z_c>t}` với threshold có thể hạ để mở rộng tập, thay vì giả vờ z là xác suất. Predictor/score phải cố định; chọn loss, chiều parameter và quy tắc ties/empty/fallback đáp ứng điều kiện lý thuyết. Việc áp dụng CRC lên z là mở rộng cho dự án, chưa được triển khai hoặc kiểm chứng. F1 không đơn điệu khi thêm genre, nên không đưa trực tiếp vào công thức CRC cơ bản.

## 6. Phần quan hệ ảnh–text đã làm có vai trò gì?

[pair_relation.py](../src/pair_relation.py) đã có cosine, JS và hệ số reliability từ entropy, nhưng flag mặc định tắt. Khi bật `auto`, chỉ Fakeddit 6way dùng class-aware, các task khác diagnostic.

Reliability thấp làm bằng chứng relation nhỏ lại khi các nhánh phẳng. Đây là cách tránh khuếch đại bằng chứng yếu; không phải cơ chế tạo thêm bằng chứng để giải quyết phân vân. Quan hệ match/mismatch cũng không áp chung lên mọi fake subtype. Xem [báo cáo quan hệ](quan_he_anh_text_20261012.md).

**Đề xuất riêng:** nếu nghiên cứu cascade, giữ relation là yếu tố thí nghiệm tách biệt. Không bật đồng thời ensemble, TTA, relation và đổi threshold rồi quy cải thiện cho uncertainty handling.

## 7. Thiết kế thí nghiệm để xác định có cải thiện thật

1. **Đóng băng baseline:** cùng ID mẫu, thứ tự nhãn, prompt, tiền xử lý, weights, threshold, text chunking và relation flag. Nhóm thiếu chắc chắn phải được xác định từ điểm baseline, trước khi xem phương pháp mới sửa được mẫu nào.
2. **Tách protocol:** label-free zero-shot dùng tham số đã quy định, nhãn chỉ dùng đánh giá. Chọn gate/weights trên nhãn validation là model selection có nhãn; temperature/conformal là bước calibration có nhãn. Ghi rõ dù backbone vẫn không fine-tune.
3. **Các đối chứng:** baseline; gate chỉ ghi uncertainty; ensemble; augmentation averaging; TDA; StatA; TPT khi đủ tài nguyên. Thử độc lập trước khi kết hợp. MM-IMDb cần nhánh nghiên cứu riêng, không chạy nguyên thuật toán categorical.
4. **Metric trên toàn bộ cohort:** binary/multiclass báo accuracy, macro-F1, balanced accuracy, từng lớp; multi-label báo micro/macro/samples-F1, precision/recall, số genre và fallback rate. Chia thêm nhóm theo margin/distance nhưng vẫn giữ metric toàn bộ.
5. **Đếm tác động thay đổi:** `sai→đúng`, `đúng→sai`, còn sai, giữ đúng trên cùng mẫu; báo rõ ca uncertain và phần còn lại. Với multi-label, đếm TP/FP/FN thay đổi từng genre. Confidence tăng mà lỗi không giảm chưa phải cải thiện phân loại.
6. **Đo độ tin cậy:** ECE cùng reliability diagram, NLL/Brier cho xác suất phù hợp. Selective báo coverage/risk curve; conformal báo coverage, set size, singleton/empty rate, từng lớp. Không dùng ECE categorical cho z.
7. **Dữ liệu độc lập:** nếu tune trước rồi conformal, cần calibration sạch chưa dùng để chọn pipeline và test độc lập. `evaluation.split: val` hiện chỉ là khai báo; chưa tự xác minh split. Tách theo tweet/sự kiện/phim hoặc nguồn phù hợp để tránh leakage từ mẫu liên quan.
8. **TTA có trạng thái:** đối chiếu nhiều thứ tự stream, reset cache giữa dataset/task/model, ghi warm-up và hyperparameter. Để dùng split conformal cơ bản sau TTA, phải có predictor phù hợp cố định; không vừa cập nhật cache trên calibration/test rồi tự áp bảo đảm như predictor bất biến.
9. **Ngân sách:** báo số lần encode, latency và bộ nhớ. Cascade chỉ đáng chọn khi trade-off đo được. Nếu sử dụng randomness, cố định seed và kiểm tra mức dao động.

Các smoke test cũ chỉ xác minh luồng trên 5 dòng đầu, không dùng để chọn phương pháp tốt nhất hay chứng minh nâng accuracy. Turn nghiên cứu này chỉ đọc code/cấu hình và nguồn học thuật; không đọc dataset hoặc chạy inference/train.

## 8. Thứ tự ưu tiên thực tế

| Ưu tiên | Đề xuất cho dự án | Điều kiện |
|---|---|---|
| 1 | Chẩn đoán uncertainty từ log và phân biệt hai nhánh phẳng/bất đồng | Không cần model mới |
| 2 | Ensemble các backbone/prompt hiện có; thử gate ca khó | Có thêm encode; không hứa trước cải thiện |
| 3 | TDA với StatA làm đối chứng cho single-label | Chấp nhận TTA có trạng thái và protocol stream |
| 4 | Conformal binary/multiclass; risk control multi-label | Có calibration có nhãn độc lập |
| 5 | TPT/O-TPT | Chấp nhận gradient, sửa adapter backbone và chi phí |

Nếu benchmark buộc trả đúng một nhãn cho mọi mẫu, conformal/reject là output chẩn đoán bổ sung; ưu tiên nghiên cứu các cách có thể thay đổi quyết định đúng hơn. Nếu ứng dụng cho phép trì hoãn, tập ứng viên hoặc review giúp tránh khẳng định khi chưa đủ bằng chứng.

Không có phương pháp được các nguồn trên chứng minh giải quyết mọi mẫu phân vân trên mọi dataset. Kết luận lựa chọn cuối phải dựa trên thí nghiệm đúng protocol của dự án.
