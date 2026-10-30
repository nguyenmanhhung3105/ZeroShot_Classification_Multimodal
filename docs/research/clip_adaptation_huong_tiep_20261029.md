# Hướng phát triển CLIP sau thí nghiệm transductive binary

## Phạm vi và quyết định hiện tại

Người dùng cho phép mở rộng nghiên cứu và đề xuất triển khai phương pháp có
khả năng giúp dự án. Vẫn ưu tiên CLIP/MetaCLIP2; few-shot Tip-Adapter đã được
lưu riêng, chưa triển khai support có nhãn. Không coi đổi prompt là hướng duy
nhất. Chỉ dùng kết quả dự án để xác nhận cải thiện, không chuyển mức tăng trên
ImageNet thành phần trăm kỳ vọng của CrisisMMD.

## Các bài và mã nguồn đã xem

| Hướng | Cơ chế | Điều kiện thực tế cho dự án | Ưu tiên |
|---|---|---|---|
| StatA, CVPR 2025 | Neo thống kê phân phối ảnh về prototype từ prompt CLIP khi thích nghi | Cần embedding ảnh và text, không đủ nếu chỉ có score hai chiều; phù hợp để kiểm tra trôi cụm/nhóm thiếu lớp | Đã chạy dev: Macro-F1 47,21%; không chọn thay baseline |
| TDA, CVPR 2024 | Bộ nhớ động gồm embedding và nhãn giả, có nhánh positive/negative | Cần embedding ảnh; reset cache mỗi lượt/seed, kiểm tra thứ tự ảnh và nhãn giả sai | Đã đo train/dev và binary thật; chưa chọn thay ensemble |
| DPE, NeurIPS 2024 | Prototype ảnh và text tiến hóa; học residual để căn chỉnh | Cần embedding nhiều view và tối ưu residual; chi phí cao hơn cache, không phải chỉ trung bình score | Để sau hai hướng trên |
| Frolic, NeurIPS 2024 | Phân phối prototype prompt, confidence matching và sửa label bias | Core repo phụ thuộc nhiều module chưa được cung cấp đầy đủ; cần kiểm tra giả định bias với dữ liệu lệch lớp | Chưa ưu tiên khi thời gian gấp |
| InMaP, NeurIPS 2023 | Tối ưu proxy ảnh từ nhãn giả, encoder frozen | Hai biến thể có/không Sinkhorn; prior đều gây hại tám lớp | Đã thử; giữ cổng không Sinkhorn, binary train 81,75% |
| LAME, CVPR 2022 | Hiệu chỉnh posterior theo lân cận ảnh, không đổi trọng số | RBF/kNN trong nhóm 75, cùng seed đối chứng | Đã thử, giảm kết quả; không chọn cấu hình này |

Đã đọc phần phương pháp StatA, TDA, DPE; rà soát abstract và core source
Frolic. Không tuyên bố đã tái lập benchmark các paper này.

### StatA

[Paper](https://arxiv.org/abs/2501.03729),
[phần phương pháp](https://arxiv.org/html/2501.03729v1#S4),
[repo](https://github.com/MaxZanella/StatA).

Paper kiểm tra các thiết lập số lớp thực sự xuất hiện trong batch thay đổi
và dòng ảnh có tương quan. Statistical Anchor regularize tham số Gaussian
bằng kiến thức text encoder. Chọn hướng này là suy luận từ vấn đề quan sát
ở dự án: lớp hiếm bị dự đoán quá nhiều khi dùng EM-Dirichlet tầng 2; cổng
binary có mức tăng nhỏ nhưng giảm recall. Đây chưa phải bằng chứng StatA
sẽ khắc phục các vấn đề đó trên CrisisMMD.

Đã kiểm tra `solvers/StatA.py`, parser mặc định của `main.py`, hướng dẫn repo;
commit `2635f9355f9389756064becb5d9ed54c2756809c`. Triển khai ở
`experiments/clip_stata/` và cờ opt-in trong runner hai tầng. Có baseline
chuẩn hóa prototype riêng để tránh quy nhầm thay đổi aggregation thành
lợi ích của thuật toán. Chi tiết/lệnh chạy trong `experiments/clip_stata/README.md`.

### TDA

[Paper](https://arxiv.org/abs/2403.18293),
[phần phương pháp](https://arxiv.org/html/2403.18293v1),
[repo](https://github.com/kdiAAA/TDA).

Cache dùng pseudo-label do model dự đoán, khác support ground truth của
Tip-Adapter. Không cần backpropagation của backbone. Tuy nhiên cache tích
lũy có thể lưu sai và phụ thuộc thứ tự dòng ảnh; paper StatA cũng đánh giá
hạn chế của các phương pháp memory khi dòng dữ liệu tương quan. Nếu triển
khai, phải tính baseline cùng ảnh, thử thứ tự ngẫu nhiên và theo sự kiện,
reset cache mỗi lượt, không chọn thứ tự tốt nhất.

### DPE

[Paper](https://arxiv.org/abs/2410.12790),
[phần phương pháp](https://arxiv.org/html/2410.12790v1),
[repo](https://github.com/zhangce01/DPE-CLIP).

Textual prototype và visual prototype được cập nhật online. Residual tối ưu
qua entropy trên nhiều augmented view và ràng buộc căn chỉnh prototype.
Không nên gọi bản chỉ cache/không residual là tái lập DPE đầy đủ. Phần
"multimodal" trong paper liên quan prototype ảnh và prompt, không tự có
nghĩa là đã dùng tweet của CrisisMMD.

### Frolic

[Paper NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/03cdf8e212ba92a3f36bffe1391928bd-Abstract-Conference.html),
[repo](https://github.com/zhuhsingyuu/Frolic),
[core source](https://github.com/zhuhsingyuu/Frolic/blob/main/main_zeroshot.py).

Rà soát thấy repo công bố core implementation, hướng dẫn nói các thành phần
khác sẽ cung cấp sau. Core source import các dataset/utils/config bên ngoài
repo hiện có. Chưa ưu tiên dựng lại toàn bộ phần thiếu hoặc dùng tùy tiện
logit adjustment trên các lớp CrisisMMD vốn đã rất lệch.

## Thử nghiệm cổng bảo vệ đã hoàn thành

Một ablation của dự án: giữ informative của CLIP gốc khi margin cosine lớn;
các ảnh còn lại theo cổng transductive. Ngưỡng là median của các margin train
dương, lấy **chỉ từ score**, không dùng nhãn tối ưu tham số. Đã khóa quy tắc
trước khi tính metric dev, không dò nhiều quantile sau khi xem kết quả.

Margin = score(informative) - score(not_informative), threshold thực tế
0,020149379968643175 từ 1.841 margin dương trong 3.000 ảnh train.

Kết quả trên 2.219 ảnh dev, ba grouping seed:

- Transductive binary hiện tại: Accuracy 69,1453%, Macro-F1 48,7146%.
- Thêm guard: Accuracy 69,1603%, Macro-F1 48,7252%.
- Chỉ khôi phục một quyết định ở seed 2; các lượt khác không thay đổi.

Không coi đây là cải tiến đáng kể; không dò thêm ngưỡng trên dev. Code ở
`experiments/clip_transductive/confidence_guard.py`, cấu hình `config_guard.json`,
artifact `results/analysis/clip_transductive_guard/cc9b18bef2b0/`. Đây là giả
thuyết riêng của dự án, không phải StatA hoặc phương pháp đã công bố.

## Kế hoạch đánh giá đã cụ thể hóa

Cập nhật theo yêu cầu tiếp tục thử phương pháp mới: InMaP không Sinkhorn
ở cổng + DN-unit tầng chi tiết đạt Macro-F1 train/dev 50,36/49,75%, binary
train 81,75%, recall cứu trợ dev 70,36%. Majority + DN vẫn có dev Macro-F1
cao hơn 50,39% nhưng recall cứu trợ 60,00%; chưa có ứng viên thắng mọi mặt.
LAME ở RBF/kNN nhóm 75 giảm kết quả, đã đối chiếu mã tác giả và không chọn.
Theo dõi kết quả/giả thuyết và hàng đợi DMN-ZS, DPE ở
`docs/research/so_theo_doi_phuong_phap.md`. Test thật vẫn chưa chạy.

Cập nhật lượt tiếp theo: đã triển khai ZLaP (CVPR 2024) và adaptation DN-unit
từ DN (NeurIPS 2023) trên embedding đã lưu. ZLaP cả hai tầng chỉ 41,78%
Macro-F1 dev, không chọn. Majority + DN-unit tầng chi tiết đạt 50,39% dev,
so với majority 49,24%; train 49,22% so với majority 49,34%, chưa thắng gốc
49,88%. Binary giữ nguyên majority 80,83%. Đối chứng chỉ chuẩn hóa text
đạt dev 50,40% nhưng train 47,63%; không quy mọi lợi ích cho DN.
Bootstrap theo nhóm query còn chứa 0, chưa kết luận chắc chắn hoặc đổi
mặc định pipeline. Chi tiết scope, source commit, đối chiếu tác giả và các
đánh đổi ở `experiments/clip_zlap/README.md`. DN-unit dùng vector unit sẵn
có, không tuyên bố tái lập đầy đủ DN trên encoder output chưa chuẩn hóa.

Đã thử thêm soft mean và raw CLIP OR majority trên cùng artifact train/dev.
Soft mean đạt binary train 82,34% và 8 nhãn train 50,81% Macro-F1, nhưng
8 nhãn dev chỉ 47,89%, thấp hơn majority 49,24%. Raw CLIP OR majority giống
hệt baseline. Chưa có tổ hợp cải thiện đồng thời các phạm vi; không chọn
soft mean từ mức tăng train để thay pipeline 8 nhãn. Chi tiết kiểm chứng ở
`experiments/clip_transductive/FUSION_EXPERIMENT.md`.

Cập nhật sau các lượt full: ensemble majority hiện đạt Macro-F1 dev 49,24%,
vẫn được giữ làm đối chứng. TDA đã được triển khai ở `experiments/clip_tda/`.
Binary normalized CLIP trên 3.000 train đạt 80,22% Macro-F1, baseline 79,58%;
chuẩn hóa trước EM chỉ 79,63%, không thay EM cũ. Đã chuẩn bị job
`experiments/clip_two_stage/launcher_binary_features.yaml` vừa export feature
vừa chạy TDA với nhãn informative thật. Chi tiết nguồn, kiểm thử, kết quả và
lệnh ở `experiments/clip_tda/README.md`. Các bước bên dưới là kế hoạch gốc;
StatA và bước tái sử dụng feature cho TDA đã thực hiện.

1. StatA smoke năm ảnh để xác minh MetaCLIP2, prototype/score, serialization
   và replay trên môi trường Kaggle; sau đó mới chạy đủ 2.219 dev.
2. So cùng ảnh: hai tầng gốc, normalized gate control, StatA gate trung bình
   ba grouping seed; giữ humanitarian argmax. Không chọn seed tốt nhất.
3. Xem Accuracy, Macro-F1 và F1/recall từng lớp, đặc biệt cứu trợ. Chỉ thêm
   binary metric khi có nhãn binary thật; cohort dev hiện không có.
4. Nếu cần thử TDA, tái dùng embedding được lưu từ lượt StatA; không encode
   lại ảnh nếu checkpoint/prompt/cohort không đổi.
5. Khi chốt phương pháp, kiểm tra loại trùng train+dev trước test cuối. Không
   dùng kết quả test để chọn ngưỡng, phương pháp hoặc seed.
