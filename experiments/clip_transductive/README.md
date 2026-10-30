# Thử Transductive CLIP trên CrisisMMD

## Trạng thái và kết luận hiện tại

Đã triển khai và chạy một phương pháp: **EM-Dirichlet zero-shot** ở tầng
humanitarian, giữ cổng informative của phương pháp hai tầng. Dùng lại điểm
cosine MetaCLIP2 đã lưu cho đúng 2.219 ảnh dev, prompt Waffle seed 0.

Kết quả ngày 28-10-2026:

| Phương pháp | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Hai tầng hiện tại, cùng ảnh và điểm | 66,20 | 48,06 |
| Transductive, grouping seed 0 | 60,43 | 40,14 |
| Transductive, grouping seed 1 | 60,07 | 40,19 |
| Transductive, grouping seed 2 | 60,88 | 40,92 |
| Trung bình ba cách chia nhóm | 60,46 | 40,41 |

Macro-F1 trung bình giảm **7,64 điểm phần trăm**. Độ lệch chuẩn tổng thể giữa
ba cách chia nhóm là 0,36 điểm phần trăm Macro-F1. Giữ phương pháp hai tầng
hiện tại làm ứng viên chính; chưa có cơ sở chọn cấu hình transductive này.
Không điều chỉnh nhiệt độ, lambda, kích thước nhóm hoặc chọn seed tốt nhất
sau khi xem kết quả.

Artifact đầy đủ: `results/analysis/clip_transductive/f543d3f765b7/`.
Có `analysis.json`, `summary.tsv`, `run_manifest.json`, `audit.json` và từng
`grouping_seed_N/{predictions.tsv,metrics.json,diagnostics.json}`.

## Câu hỏi nghiên cứu và cách làm

Với cùng backbone, cùng prompt và cùng ảnh, suy luận chung trên các query
chưa có nhãn có cải thiện hơn dự đoán độc lập hay không?

1. Xác thực artifact hai tầng cũ: protocol/cohort fingerprints, ID, thứ tự,
   ground truth, score và tính lại dự đoán baseline. Chỉ nhận artifact dev
   image-only, nhãn ảnh; từ chối test trong thử nghiệm thăm dò này.
2. Giữ cổng informative nguyên trạng. 802 ảnh bị chặn vẫn nhận not_humanitarian.
3. Với 1.417 ảnh qua cổng, lấy điểm bảy lớp dương từ bank tám lớp cũ;
   không sinh lại prompt. Tạo feature bằng `softmax(30 * cosine_scores)`.
4. Xáo trộn chỉ số ảnh qua cổng bằng grouping seeds cố định `[0,1,2]`, không
   dùng nhãn hoặc sự kiện. Mỗi seed có 18 nhóm 75 ảnh và một nhóm 67 ảnh.
5. Fit EM-Dirichlet cho từng nhóm: 20 vòng ngoài, tối đa 1.000 vòng MM bên
   trong. Lambda theo code gốc `floor(K/5)*N`, tức lambda/N=1 với K=7.
   Lưu ý code gốc ghi đè `lambd: 0` của YAML bằng công thức này.
6. Ghép các cụm có ảnh với tên lớp bằng Hungarian trên trung bình feature
   CLIP của cụm, rồi đưa nhãn về đúng thứ tự ảnh ban đầu.

Không dùng nhãn thật trong `predict_two_stage` hoặc `em_dirichlet`; nhãn
chỉ phục vụ xác thực artifact và tính metric. Không có support có nhãn.
Nhiệt độ dương giữ nguyên argmax baseline; runner kiểm tra đối chứng này.

Mỗi ảnh vẫn có nhãn riêng. Hungarian ghép một-một các cụm đang có với lớp,
không đặt số ảnh bằng nhau cho các lớp và không bắt mọi lớp phải xuất hiện.
Tuy nhiên, nếu thuật toán tạo nhiều cụm, bước ghép vẫn có thể gán một cụm
cho lớp hiếm hoặc thực tế không có trong nhóm. Đây là một giới hạn cần chú ý.

Grouping seed thay cách chia nhóm ảnh; prompt seed vẫn là 0. Trung bình là
trung bình metric ba lượt, không phải ensemble dự đoán hoặc chọn seed tốt nhất.

## Kiểm tra tính đúng

- 85 kiểm thử liên quan đạt, gồm feature/argmax, objective giảm, matching
  cụm đang có, tính tương đương khi hoán vị ảnh, nhóm cuối, singleton, cổng
  không đổi, từ chối artifact lỗi/test và không ghi đè kết quả.
- Kiểm thử end-to-end thay ground truth trong hai artifact tổng hợp: nhãn
  dự đoán và batch ID giữ nguyên, metric được tính riêng từ ground truth.
- Đối chiếu trực tiếp class EM_DIRICHLET và matching gốc ở commit
  `9066ad371757c794e4b8255e852f837ac9a1d3b1`: toàn bộ các nhóm của cả ba
  seed dev và hai ca bổ sung đều cho nhãn giống hệt. Sai số assignment lớn
  nhất dưới 1,4e-12 khi dùng float64 cho cả hai. Code gốc vẫn giữ các cast
  float tường minh. Chi tiết: `reference_verification.json`.
- Audit sau chạy xác minh 11 hash output, hash code và source artifact;
  tính lại metric, kiểm tra mọi ID/nhãn/thứ tự/batch ID/gate, fixed/broken
  và objective trên cả ba seed. Kết quả trong `audit.json`.

Đối chiếu dùng AST của các class thuật toán và hàm matching/one-hot đã đọc
từ repo gốc; không import toàn bộ benchmark hoặc nạp model/dataset tác giả.
Mã tính toán dự án viết từ phương trình paper, dùng NumPy/SciPy float64,
nghiệm MM được tính theo dạng tương đương để giảm triệt tiêu số học.

## Phân tích kết quả và giới hạn

F1 rescue tăng từ 51,02% lên trung bình 58,65%, nhưng F1 các lớp khác giảm,
đặc biệt injured/dead từ 30,77% xuống 5,13%, vehicle từ 50,67% xuống 29,20%.
Lớp missing/found chỉ có hai ảnh thật, trong khi mỗi lượt dự đoán trung bình
khoảng 155 ảnh vào lớp này. Số lượng dự đoán lớp hiếm bị tăng mạnh là hiện
tượng quan sát được; chưa tách ablation để quy riêng cho EM hay Hungarian.

Dev đã dùng trong các lần chọn phương án trước; đây là kết quả phát triển,
không phải đánh giá test cuối cùng. Đây cũng là thay đổi sang thiết lập
transductive: nhãn một ảnh phụ thuộc các query khác. Việc áp dụng EM-Dirichlet
ở tầng 2 với MetaCLIP2/prompt dự án không phải tái lập toàn bộ benchmark paper.

Ba seed chỉ kiểm tra độ nhạy chia nhóm, không đủ chứng minh ý nghĩa thống kê.
Không báo CI bootstrap từng ảnh vì các dự đoán cùng nhóm có phụ thuộc.
Giữ cổng tầng 1 cũng có nghĩa phương pháp này không thể sửa false negative
của cổng. Với nhóm chỉ một ảnh, code giữ argmax gốc và ghi fallback; lượt dev
thực tế không có nhóm singleton.

## Chạy lại và kiểm thử

Lượt trên đã chạy xong bằng CPU. Có thể tái lập từ thư mục gốc:

```sh
venv/bin/python -B experiments/clip_transductive/run.py
```

Mỗi lượt tạo thư mục kết quả mới. Cấu hình đã khóa ở `config.json`.
Không cần gửi lại job Kaggle hoặc encode lại ảnh để kiểm tra thuật toán này.

```sh
PYTHONPATH=src:tests venv/bin/python -B -m unittest \
  experiments.clip_transductive.test_transductive \
  experiments.clip_two_stage.test_two_stage \
  experiments.clip_two_stage.test_multimodal \
  experiments.clip_two_stage.test_views \
  test_launch_runs test_results_layout -b -q
```

Đối chiếu số lại nếu đã có checkout repo tham chiếu đúng commit:

```sh
venv/bin/python -B experiments/clip_transductive/verify_reference.py \
  --reference-dir /private/tmp/vlm-transductive-clip-reference \
  --source-run results/clip_two_stage_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/humanitarian_77187aa56f13 \
  --all-groupings
```

## Nguồn

- [Transductive Zero-Shot and Few-Shot CLIP, CVPR 2024](https://arxiv.org/abs/2405.18437).
- [Implementation chính thức tại commit tham chiếu](https://github.com/SegoleneMartin/transductive-CLIP/tree/9066ad371757c794e4b8255e852f837ac9a1d3b1).
- File đối chiếu: `src/methods/zero_shot/em_dirichlet.py`, `src/utils.py`,
  `config/methods_config/em_dirichlet.yaml`, `config/main_config.yaml`.
