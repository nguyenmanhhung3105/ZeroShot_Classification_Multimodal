# InMaP và tổ hợp cổng InMaP + tầng chi tiết DN-unit

[Paper NeurIPS 2023](https://arxiv.org/abs/2310.19752),
[repo tác giả](https://github.com/idstcv/InMaP), commit
`5a6f7c94fac040401a5afdf06adc0a31b734534e`.

## Giả thuyết và cấu hình

InMaP tối ưu vector đại diện lớp trong không gian ảnh với nhãn giả từ CLIP.
Encoder không cập nhật, không dùng nhãn thật hoặc support few-shot. Có tối
ưu classifier bằng projected gradient; không gọi đây là chỉ đổi prompt hay
không có tối ưu tham số nào.

Khóa hai biến thể có trong repo trước khi đo: không Sinkhorn (ablation
InMaP^0.5 của paper) và có Sinkhorn với prior đều. Giữ 2.000 vòng proxy,
20 vòng Sinkhorn, lr=10, tau_t=0,01, tau_i=0,04, alpha=0,6, gamma=0.
Không dò gamma/temperature để nâng dev. Prior đều là giả định đáng ngờ
với CrisisMMD lệch lớp, nên không mặc định dùng nó.

Fit mỗi task/split trên toàn cohort ảnh không nhãn, không trộn train/dev.
Không mở test hoặc dataset ảnh, chỉ dùng cached feature cùng prompt đã có.
Mean prototype được unit-normalize như tác giả, có đối chứng riêng.

## Kết quả ngày 29-10-2026

Macro-F1 (%); train 3.000 ảnh, dev 2.219 ảnh. Majority là trung bình ba seed;
InMaP thuần xác định, không coi ghi lặp theo seed là ba quan sát độc lập.

| Phương án | Binary train | 8 nhãn train | 8 nhãn dev |
|---|---:|---:|---:|
| Hai tầng gốc | 79,58 | 49,88 | 48,06 |
| Majority, tầng chi tiết gốc | 80,83 | 49,34 | 49,24 |
| Majority + DN-unit tầng chi tiết | 80,83 | 49,22 | 50,39 |
| InMaP không Sinkhorn ở cổng, tầng chi tiết gốc | 81,75 | 50,67 | 48,64 |
| Majority + InMaP không Sinkhorn tầng chi tiết | 80,83 | 48,90 | 48,99 |
| InMaP không Sinkhorn cả hai tầng | 81,75 | 50,22 | 48,44 |
| InMaP có prior đều ở cổng, tầng chi tiết gốc | 81,16 | 48,93 | 48,77 |
| InMaP có prior đều cả hai tầng | 81,16 | 43,62 | 42,40 |
| **Cổng InMaP không Sinkhorn + DN-unit tầng chi tiết** | **81,75** | **50,36** | **49,75** |

Không có binary ground truth dev; không suy nó từ humanitarian. InMaP
không Sinkhorn giúp binary train 0,93 điểm so với majority nhưng chưa vượt
soft mean 82,34% đã thử trước đó. Không gọi nó là tốt nhất tuyệt đối.

Tổ hợp mới ghép đúng hai phần có tín hiệu bổ sung, không fit trọng số:
cổng InMaP không Sinkhorn, rồi DN-unit argmax bảy lớp dương. So với hai tầng
gốc, Macro-F1 tăng trên cả train (+0,48 điểm) và dev (+1,69 điểm). Recall cứu
trợ dev là **70,36%**, so với 60,00% của majority+DN-unit và 76,07% của gốc.
Tradeoff: Macro-F1 dev 49,75% vẫn thấp hơn majority+DN-unit 50,39%; không
tuyên bố thắng trên mọi tiêu chí hoặc chọn phương án bằng test.

Ép prior đều làm giảm mạnh kết quả tám lớp, phù hợp với cảnh báo về phân
bố lệch. Đây là kết quả trên cấu hình cụ thể, không chứng minh mọi biến thể
OT hoặc mọi cách học proxy đều không phù hợp.

## Kiểm chứng

- Ba unit test: mass Sinkhorn, vector proxy unit/không sửa input và tính
  xác định, từ chối embedding/config không hợp lệ.
- Trích đúng hai hàm `image_opt`, `sinkhorn` từ AST repo chính thức, không
  import main/model/dataset. Bốn ca giả và đủ tám fit thực (2/8 lớp ×
  train/dev × có/không Sinkhorn), 2.000 vòng, khớp mọi vector proxy và nhãn
  trên CPU FP32 một thread. Giữ classifier [D,C] contiguous như tác giả
  để tránh khác accumulation FP32 khi tạo pseudo-label.
- Audit đối chiếu hash source/output, NPZ ID, score từ proxy, gate, prediction
  và metric từng lớp của toàn bộ seed. Mean và prototype DN không fit lại.
- Dev đã được xem nhiều lần; kết quả thăm dò, chưa kiểm chứng test độc lập.

Artifact chính: `results/analysis/clip_inmap/37d5a5db4a8a/`.
Tổ hợp: `results/analysis/clip_inmap_dn/1b61ddf777c1/`.
Các lượt đã hoàn tất, không gửi Kaggle hoặc encode lại ảnh.

Tái lập (tạo output mới):

```sh
venv/bin/python -B experiments/clip_inmap/run.py
venv/bin/python -B experiments/clip_inmap/combine_dn.py \
  --inmap-run results/analysis/clip_inmap/37d5a5db4a8a \
  --dn-run results/analysis/clip_dn_unit/d70959a0a21c
venv/bin/python -B -m unittest experiments.clip_inmap.test_inmap -b -q
```
