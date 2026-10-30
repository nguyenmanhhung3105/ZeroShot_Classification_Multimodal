# Đối chứng bỏ missing_or_found_people

## Phạm vi

Theo yêu cầu người dùng, kiểm tra hai tầng độc lập và EM-Dirichlet transductive
khi bỏ lớp `missing_or_found_people`. Không đổi taxonomy hoặc dữ liệu của
pipeline chính; lưu kết quả ablation ở thư mục riêng.

- Bỏ đúng hai ảnh có ground truth thuộc lớp này, giữ 2.217/2.219 ảnh dev.
- Loại cột điểm của lớp này khỏi tập ứng viên cho cả hai phương pháp.
- Bài toán mới có bảy lớp tổng cộng, gồm not_humanitarian và sáu lớp dương.
- Giữ nguyên MetaCLIP2, prompt Waffle seed 0, ảnh, điểm cosine và gate.
  Dùng điểm từ bank tám lớp đã sinh, không sinh lại prompt ít lớp hơn.
- Tính lại argmax hai tầng và chạy lại EM-Dirichlet từ feature sáu lớp;
  không chỉ bỏ một dòng trong bảng F1 cũ, không lọc ảnh theo nhãn dự đoán.
- Giữ temperature=30, query size=75, 20 vòng ngoài, tối đa 1.000 vòng MM;
  lambda/N theo code gốc vẫn bằng floor(6/5)=1.
- Cả ba grouping seed 0,1,2 giữ cùng tập ảnh; chỉ đổi cách chia nhóm.
  Tầng 1 cho qua 1.415 ảnh: 18 nhóm 75 và một nhóm 65 ở mỗi seed.

## Kết quả

| Phương pháp | Accuracy (%) | Macro-F1 bảy lớp (%) |
|---|---:|---:|
| Hai tầng | 66,35 | 53,66 |
| Transductive, grouping seed 0 | 59,99 | 42,98 |
| Transductive, grouping seed 1 | 60,67 | 43,75 |
| Transductive, grouping seed 2 | 60,13 | 44,39 |
| Transductive, trung bình | 60,26 | 43,71 |

Transductive thấp hơn hai tầng **9,95 điểm phần trăm Macro-F1** và
**6,09 điểm phần trăm Accuracy**. Độ lệch chuẩn tổng thể Macro-F1 giữa
ba cách chia nhóm là 0,57 điểm phần trăm.

Lớp injured_or_dead_people có tám ảnh thật, trong khi transductive dự đoán
trung bình khoảng 206 ảnh thuộc lớp này. F1 lớp này: hai tầng 32,00%,
transductive trung bình 3,26%. F1 vehicle_damage: 47,50% so với 22,60%.
F1 rescue tăng từ 50,12% lên 56,24%, nhưng không bù được suy giảm các lớp khác.

Bỏ missing/found chưa giải quyết được suy giảm của transductive. Không chọn
seed tốt nhất hoặc tiếp tục bỏ các lớp khác dựa trên bảng kết quả này.

## Xác minh

- 12 kiểm thử transductive/ablation đạt, gồm lọc đúng ground truth và cột
  score, không thay dữ liệu nguồn, giữ gate, tính lại nhãn cho ảnh trước
  đó bị dự đoán vào lớp đã bỏ, đủ cohort và không ghi đè output.
- Kiểm tra hash code, artifact nguồn và 12 output của lượt mới.
- Tính lại metric/fixed/broken, kiểm tra ID/thứ tự/nhãn/gate/batch membership
  trên cả ba seed. Mọi dự đoán đều thuộc bảy lớp mới.
- Đối chiếu mọi nhóm sáu lớp dương với EM_DIRICHLET + Hungarian gốc ở commit
  `9066ad371757c794e4b8255e852f837ac9a1d3b1`: toàn bộ nhãn dự đoán lưu trên
  2.217 ảnh của mỗi seed khớp; sai số assignment float64 tối đa dưới 5,2e-12.

Artifact: `results/analysis/clip_transductive_drop_class/f41f8173dfd5/`.
`audit.json` chứa kết quả xác minh và hash nguồn tham chiếu; `subset.json`
lưu chỉ số nguồn/ảnh được giữ và ID của hai ảnh bị bỏ. Các dự đoán, metric
và objective từng nhóm lưu riêng theo grouping seed.

## Giới hạn diễn giải

Macro-F1 mới trung bình trên bảy lớp, nên mức tăng so với Macro-F1 tám lớp
trước đó không tự chứng minh model tốt hơn. So sánh trực tiếp có ý nghĩa ở
đây là hai phương pháp trên cùng task bảy lớp và cùng 2.217 ảnh.

Ground truth được dùng để định nghĩa task bỏ lớp và tính metric; thuật toán
dự đoán vẫn không nhận nhãn thật. Đánh giá này không giải quyết trường hợp
ảnh missing/found vẫn xuất hiện khi triển khai. Giữ task tám lớp gốc cho kết
quả chính nếu yêu cầu dự án vẫn bao gồm lớp này. Đây là thăm dò trên dev đã
xem trước đó, chưa phải kết quả test độc lập.

## Tái lập

Lượt mới đã chạy bằng CPU từ điểm CLIP đã lưu, không cần chạy Kaggle:

```sh
venv/bin/python -B experiments/clip_transductive/drop_class.py
```

Lệnh tạo thư mục mới, không sửa artifact cũ. Kiểm thử:

```sh
venv/bin/python -B -m unittest \
  experiments.clip_transductive.test_drop_class \
  experiments.clip_transductive.test_transductive -b -q
```
