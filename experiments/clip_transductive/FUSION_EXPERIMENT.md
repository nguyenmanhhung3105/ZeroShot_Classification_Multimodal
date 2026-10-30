# Kiểm tra thêm cách kết hợp cổng CLIP, EM-Dirichlet và StatA

## Phạm vi

Hai quy tắc được cố định trong `config_fusion.json` trước khi tính metric
kết hợp. Dùng embedding và score đã lưu, cùng MetaCLIP2, prompt seed 0,
grouping seed 0/1/2, nhóm 75 ảnh. Không chạy lại backbone, không gửi Kaggle,
không fit trọng số/ngưỡng theo nhãn, không đổi cách chọn một trong bảy lớp
dương ở tầng 2. Đây là ablation của dự án, không phải phương pháp mới đã
được paper chứng minh.

- **Soft mean:** trung bình bằng nhau của phân phối normalized CLIP
  (logit scale checkpoint 100), EM-Dirichlet (đầu vào scale 30 như đối chứng)
  và StatA. Cho đi tiếp khi điểm informative trung bình >= 0,5. Cột EM là
  cụm tiềm ẩn: ghép về nhãn bằng Hungarian; với binary, cụm chưa có mẫu thắng
  nhận lớp còn lại. Không giả định ba nhánh đã được hiệu chuẩn xác suất.
- **Protected majority:** raw CLIP gốc OR majority ba nhánh. Giữ nguyên lớp
  tầng 2 cho mọi ảnh raw CLIP đã cho qua, nhằm bảo vệ recall các lớp dương.
  Khác phép OR cũ giữa normalized CLIP, EM và StatA.
- **Đối chứng majority:** normalized CLIP, raw EM và StatA biểu quyết 2/3;
  đối chiếu từng dự đoán với các lượt trước, không chỉ so metric.

## Kết quả ngày 29-10-2026

Macro-F1 (%); các tổ hợp lấy trung bình ba grouping seed, không chọn seed tốt nhất:

| Phương pháp | 2 nhãn, train 3.000 | 8 nhãn, train 3.000 | 8 nhãn, dev 2.219 |
|---|---:|---:|---:|
| Hai tầng gốc | 79,58 | 49,88 | 48,06 |
| Majority ba cổng | 80,83 | 49,34 | **49,24** |
| Soft mean ba cổng | **82,34** | **50,81** | 47,89 |
| Protected majority | 79,58 | 49,88 | 48,06 |

Không có nhãn informativeness thật trên cohort dev này nên binary dev để
null, không suy nhãn từ humanitarian. Train chỉ là tên split nguồn: không
huấn luyện backbone hoặc học trọng số bằng nhãn ở thí nghiệm này.

Soft mean tăng 2,76 điểm phần trăm binary Macro-F1 và 0,93 điểm 8 nhãn trên
train so với gốc. Tuy nhiên 8 nhãn dev giảm 0,17 điểm so với gốc, giảm 1,34
điểm so với majority. Accuracy dev soft mean là 70,93%, cao hơn majority
69,84%; recall cứu trợ lại chỉ 50,71%, so với majority 60,48% và gốc 76,07%.
Vì dữ liệu lệch lớp, không chọn phương án chỉ dựa trên Accuracy.

8 nhãn train của soft mean theo seed: 51,21 / 48,87 / 52,34%; độ lệch chuẩn
dân số 1,44 điểm. Train có hai mẫu missing/found; recall lớp này là 50 / 0 /
50% ở ba seed. Macro-F1 rất nhạy với lớp hiếm này, mức tăng trung bình chưa
đủ để kết luận ổn định. Trên dev soft mean chỉ 47,87 / 47,92 / 47,89%.

Protected majority giống hệt baseline ở cả hai split, cả ba seed: mọi ảnh
majority cho qua đều đã nằm trong tập raw CLIP cho qua. Phép OR khôi phục
các ảnh baseline giữ lại, nhưng không bổ sung quyết định mới.

## Xác minh và quyết định

- Bốn unit test đạt: ánh xạ cụm, đối chiếu hard EM (có nhóm cuối singleton),
  soft vote theo confidence và bảo toàn dự đoán khi bảo vệ cổng.
- Audit kiểm tra hash code/output/source; đủ ID, đúng thứ tự và nhãn;
  tính lại soft mean, vote, batch ID, dự đoán 2/8 nhãn và metric từng lớp.
- Majority khớp chính xác dự đoán các lượt train/dev đã có; không có soft
  tie EM trên cả hai split. Protected gate khớp baseline từng ảnh.
- Giữ majority làm ứng viên tốt nhất **trong các cấu hình đã thử theo
  Macro-F1 8 nhãn trên dev**. Không tự đổi mặc định pipeline. Soft mean là
  ứng viên binary cần kiểm chứng thêm, chưa chọn cho đầu ra 8 nhãn.
- Train/dev đã được xem nhiều lần; chưa có đánh giá test độc lập, không
  tuyên bố đã tìm tổ hợp tối ưu hoặc cải thiện trên mọi tiêu chí.

Artifact đầy đủ: `results/analysis/clip_gate_fusion/74ed4f8f37f4/`, gồm
`analysis.json`, `summary.tsv`, `audit.json`, frozen plan, manifest và dự đoán
từng ảnh/seed.

Lượt này đã chạy xong. Tái lập từ artifact (tạo thư mục kết quả mới):

```sh
venv/bin/python -B experiments/clip_transductive/fusion.py
venv/bin/python -B -m unittest experiments.clip_transductive.test_fusion -b -q
```
