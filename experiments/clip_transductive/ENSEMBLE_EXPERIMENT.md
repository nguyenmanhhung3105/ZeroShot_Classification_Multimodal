# Kết hợp cổng normalized CLIP, EM-Dirichlet và StatA

## Phạm vi

Theo yêu cầu người dùng, thử kết hợp các cách quyết định cổng informativeness
đã có. Cùng MetaCLIP2, prompt, 2.219 ảnh dev, tám nhãn đầu ra humanitarian
và argmax bảy lớp dương ở tầng 2. Dùng artifact đã lưu, không encode lại ảnh
và không gửi thêm job Kaggle. Đây là kết hợp ba cách suy luận dùng chung
backbone, không phải ensemble ba model được huấn luyện độc lập.

Nguồn khóa trong `config_ensemble.json`:

- Hai tầng gốc: `humanitarian_77187aa56f13`.
- Cổng EM-Dirichlet: `results/analysis/clip_transductive_gate/c3fa825bc57b/`.
- StatA và normalized control: `humanitarian_ca2d96c17e89`.

Loader kiểm tra/replay artifact StatA, hash output EM, ID, nhãn, thứ tự,
cohort, backbone, bank prompt và cosine giữa hai lượt. Ghép cùng grouping
seed 0/1/2, kiểm tra cùng batch membership của EM và StatA. Không coi ba
cách chia nhóm là ba dataset độc lập và không chọn seed tốt nhất.

## Hai quy tắc đã cố định trước khi xem metric kết hợp

1. **Majority:** ít nhất hai trong ba cổng dự đoán informative thì cho ảnh
   đi tiếp. Quy tắc "giữ EM, chỉ đổi khi normalized CLIP và StatA đồng ý"
   tương đương chính xác majority ở bài toán binary; chỉ tính là một phương án.
2. **Recall OR:** ít nhất một cổng dự đoán informative thì cho đi tiếp.
   Quy tắc này giữ mọi ảnh mà bất kỳ nhánh nào cho đi tiếp, nhằm hạn chế
   mất recall, nhưng có thể tăng false positive.

Không trộn trực tiếp các posterior khác thang đo, không dò trọng số hoặc
ngưỡng trên dev. Hàm `combine` chỉ nhận ma trận boolean ba cổng, không nhận
nhãn thật. Nhãn chỉ phục vụ xác thực artifact và đánh giá đầu ra tám lớp.
Đây là ablation của dự án, không tuyên bố là phương pháp mới đã công bố.

## Kết quả 29-10-2026

| Phương pháp | Accuracy (%) | Macro-F1 (%) | Recall cứu trợ (%) |
|---|---:|---:|---:|
| Hai tầng gốc | 66,20 | 48,06 | 76,07 |
| Normalized gate, argmax độc lập | 68,32 | 48,85 | 67,50 |
| EM-Dirichlet gate, trung bình | 69,15 | 48,71 | 58,10 |
| StatA gate, trung bình | 71,17 | 47,21 | 46,79 |
| Majority ba cổng, trung bình | **69,84** | **49,24** | 60,48 |
| Recall OR ba cổng, trung bình | 68,60 | 48,97 | **71,07** |

Majority tăng 0,52 điểm phần trăm Macro-F1 và 0,69 điểm Accuracy so với
EM riêng; tăng 0,39 điểm Macro-F1 so với normalized control. Recall cứu trợ
tăng 2,38 điểm so với EM riêng, nhưng vẫn thấp hơn 15,60 điểm so với hai tầng
gốc. Không gọi đây là cải thiện đồng thời mọi tiêu chí.

Recall OR có Macro-F1 thấp hơn majority khoảng 0,26 điểm, nhưng recall cứu
trợ cao hơn 10,60 điểm. So với hai tầng gốc, recall cứu trợ vẫn thấp hơn 5
điểm. Phương án phù hợp phụ thuộc mức ưu tiên Macro-F1 hay phát hiện cứu trợ.

| Grouping seed | Majority Accuracy / Macro-F1 (%) | Recall OR Accuracy / Macro-F1 (%) |
|---|---:|---:|
| 0 | 69,81 / 49,11 | 68,59 / 49,01 |
| 1 | 69,90 / 48,93 | 68,77 / 49,03 |
| 2 | 69,81 / 49,68 | 68,45 / 48,88 |

Majority sửa đúng/làm sai so với EM: 21/3, 17/3, 23/9 theo ba seed. Các cổng
bất đồng trên 338, 322, 337 ảnh tương ứng. Những ảnh cả ba cổng đồng ý giữ
nguyên; việc kết hợp chỉ có tác động ở phần bất đồng.

## Xác minh và độ bất định

- Ba kiểm thử đạt: toàn bộ tám tổ hợp vote, tương đương priority/majority,
  tính bao phủ của OR, giữ đúng argmax lớp dương và từ chối cohort lệch.
- Audit xác minh hash output/code, đủ 2.219 ID mỗi seed, tính lại metric,
  fixed/broken và tái dựng quy tắc vote bằng biểu thức độc lập.
- Bootstrap ghép cặp 2.000 lần theo 30 nhóm query của từng seed, giữ nguyên
  dự đoán đã có, không refit EM/StatA khi resample. Khoảng 95% của mức tăng
  Macro-F1 majority so với EM, tính theo điểm phần trăm:
  seed 0 [0,19; 1,09], seed 1 [-0,30; 1,49], seed 2 [-0,38; 1,43].
- Vì hai khoảng còn chứa 0 và dev đã được xem nhiều lần, chỉ xem đây là
  tín hiệu thăm dò, chưa kết luận mức tăng ổn định trên dữ liệu chưa thấy.
  Các nhóm có thể còn phụ thuộc theo tweet/sự kiện; không hiệu chỉnh nhiều
  phép so sánh và không gộp seed thành các quan sát độc lập.

Artifact: `results/analysis/clip_gate_ensemble/8ee92bf95f2c/` gồm frozen plan,
manifest, summary, analysis, audit và vote/nhãn dự đoán đầy đủ theo từng seed.

## Tái lập

Lượt trên đã chạy xong. Chạy lại tạo thư mục mới:

```sh
venv/bin/python -B experiments/clip_transductive/ensemble.py
```

Kiểm thử:

```sh
venv/bin/python -B -m unittest experiments.clip_transductive.test_ensemble -b -q
```

Chưa đổi mặc định pipeline/Kaggle sang ensemble. Cần chốt mục tiêu đánh giá,
khóa quy tắc và kiểm chứng tập độc lập trước khi chọn phương án cuối cùng.
