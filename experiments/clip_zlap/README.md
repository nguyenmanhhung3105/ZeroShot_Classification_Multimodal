# ZLaP và hiệu chỉnh phân phối embedding cho CLIP hai tầng

## Mục tiêu và nguồn nghiên cứu

Thử cải thiện cổng hai nhãn và lớp humanitarian chi tiết, dùng MetaCLIP2 và
prompt của dự án, không đổi backbone hoặc thêm mẫu support có nhãn.

1. **ZLaP, CVPR 2024:** [paper](https://arxiv.org/abs/2404.04072),
   [repo](https://github.com/vladan-stojnic/ZLaP), commit
   `89ab14bef8904a9397f8b6a7e5f3e6af2c288b93`.
   Tạo đồ thị ảnh–ảnh và ảnh–text riêng; chỉ lũy thừa cạnh ảnh–text;
   đối xứng/chuẩn hóa đồ thị rồi giải `(I-alpha*W) F = Y`. `Y` chỉ gán nhãn
   cho **prototype text**, không nhận ground truth ảnh. Không ép cân bằng lớp.
   Giữ cấu hình CLIP mẫu của tác giả: k=5, gamma=5, alpha=0,3, CG tol=1e-6,
   tối đa 50 vòng. Đồ thị chứa cả cohort mỗi split, khác nhóm 75 ảnh của EM.
2. **DN, NeurIPS 2023:** [paper](https://arxiv.org/abs/2302.11084),
   [repo](https://github.com/fengyuli-dev/distribution-normalization), commit
   `be4bdb6a158871a15ed60e4e4852b9de6db28da1`.
   Thử **DN-unit**, adaptation trên embedding unit đang có. Giữ hệ số 0,25
   từ `compute_zeroshot.py`. Dùng mean ảnh train không nhãn, khóa cho cả
   train/dev; trừ mean tương ứng của ảnh/text rồi chuẩn hóa và tính cosine.
   Không thử nhiều hệ số, không dùng nhãn để ước lượng mean.

DN gốc dùng vector encoder chưa chuẩn hóa; artifact hiện có chỉ lưu vector
ảnh unit và mean prompt. Vì vậy DN-unit là áp dụng cơ chế trên representation
của dự án, **không phải tái lập đầy đủ benchmark DN**. Các mức tăng của paper
không được chuyển thành kỳ vọng cho CrisisMMD.

## Dữ liệu và đối chứng

- Train: `humanitarian_471dc4a07814`, đủ 3.000 ảnh có hai bộ nhãn ảnh thật.
- Dev: `humanitarian_ca2d96c17e89`, 2.219 ảnh có nhãn humanitarian; binary
  ground truth không có, giữ metric null.
- Cổng majority: artifact `clip_gate_fusion/74ed4f8f37f4`, ba grouping seed
  0/1/2 của normalized CLIP + EM + StatA. Kiểm tra hash, ID, thứ tự, truth,
  cosine và prediction trước khi kết hợp.
- Pack dev cũ thiếu human text prototypes. Tái dùng **text vectors** đã cache
  trong pack train, kiểm tra cùng backbone, bank prompt và thứ tự nhãn;
  xác minh `X_dev @ mean_text.T` tái dựng mọi human cosine dev. Không lấy
  vector ảnh/nhãn train làm support cho dev.
- Chỉ đọc artifact kết quả; không mở dataset gốc hoặc encode lại ảnh.
  Mỗi graph ZLaP riêng từng split; không trộn ảnh train và dev.
- Có đối chứng chuẩn hóa prototype riêng để tách tác động aggregation khỏi
  thuật toán đồ thị/hiệu chỉnh phân phối. Không đổi mặc định runner chính.

## Kết quả ngày 29-10-2026

Macro-F1 (%). Với cổng majority, trung bình cả ba seed; ZLaP và DN thuần
xác định nên không coi việc ghi lặp cùng dự đoán ở ba seed là ba lần độc lập.

| Phương án | Binary train 3.000 | 8 nhãn train 3.000 | 8 nhãn dev 2.219 |
|---|---:|---:|---:|
| Hai tầng gốc | 79,58 | 49,88 | 48,06 |
| Majority, tầng chi tiết gốc | 80,83 | 49,34 | 49,24 |
| ZLaP cổng, tầng chi tiết gốc | 79,83 | 48,51 | 47,93 |
| Majority + ZLaP tầng chi tiết | 80,83 | 44,97 | 42,95 |
| ZLaP cả hai tầng | 79,83 | 44,75 | 41,78 |
| ZLaP phân loại thẳng 8 nhãn | — | 39,15 | 37,45 |
| Majority + chuẩn hóa text prototype tầng chi tiết | 80,83 | 47,63 | **50,40** |
| DN-unit cổng, tầng chi tiết gốc | 80,38 | 49,44 | 48,99 |
| DN-unit cả hai tầng | 80,38 | 49,26 | 49,74 |
| Majority + DN-unit tầng chi tiết | 80,83 | 49,22 | **50,39** |

ZLaP không được chọn: đồ thị tầng chi tiết làm giảm Macro-F1 cả train/dev.
Đối chiếu với mã tác giả khớp dự đoán trên cả hai task/split; không diễn giải
kết quả kém thành thuật toán đã có lợi nhưng triển khai chưa đúng.

**Ứng viên mới:** majority + DN-unit tầng chi tiết tăng dev 1,16 điểm Macro-F1
và 0,75 điểm Accuracy so với majority cũ (69,84% → 70,59%). Train giảm 0,12
điểm Macro-F1 (49,34% → 49,22%); so với hai tầng gốc 49,88% vẫn thấp hơn.
Không gọi đây là cải thiện đồng thời mọi bộ đánh giá. Binary giữ nguyên
majority 80,83% vì chỉ thay tầng chi tiết, không sửa cổng.

Đối chứng chỉ chuẩn hóa text có dev 50,3969%, DN-unit có 50,3947%: gần như
bằng nhau; không tuyên bố DN thắng đối chứng đó trên dev. Tuy nhiên chỉ
chuẩn hóa làm train giảm xuống 47,63%, trong khi DN-unit giữ 49,22%. Đây là
lý do giữ DN-unit làm ứng viên cân nhắc, chưa chốt mô hình cuối.

Recall cứu trợ dev: gốc 76,07%, majority 60,48%, majority+DN-unit 60,00%,
DN-unit cả hai tầng 64,29%. Phương án Macro-F1 cao chưa giải quyết hoàn toàn
việc bỏ sót cứu trợ. DN-unit cả hai tầng là đối chứng đơn giản không cần EM,
StatA hay tập query đồng thời, sau khi mean train đã được khóa.

## Độ tin cậy và kiểm chứng

- 13 kiểm thử đạt (ZLaP/DN, fusion, ensemble): nghiệm CG đối chiếu nghiệm
  dense, hoán vị ảnh khi không tie, đồ thị có cạnh cross-modal, nút cô lập,
  DN theo công thức và mean cố định không phụ thuộc batch truy vấn.
- ZLaP: chạy các hàm thuần từ AST repo tác giả, thay CuPy/SciPy và FAISS/CPU
  kNN theo cùng backend số học. Hai ca giả và cả bốn graph thực đều khớp
  toàn bộ score/prediction. Không import dataset/model của repo ngoài.
- Đây là kiểm tra thuật toán với backend CPU chung, không phải tái lập CUDA.
  kNN dot-product FP32 theo block 256, tie chọn index nhỏ. GEMM với kích
  thước batch khác có thể đổi hàng xóm gần hòa; diagnostics ghi số tie.
- DN: trích đúng bốn phép hiệu chỉnh từ AST tác giả, so trên cùng input
  unit của dự án. Cả bốn task/split khớp mọi argmax; sai số cosine tối đa
  1,94e-7. Không ngầm coi kiểm tra này là tái lập input encoder raw.
- Audit xác minh hash code/output/source, đủ ID và nhãn, các gate và từng
  prediction/metric mỗi lớp; DN mean đúng mean train đã khóa.
- Bootstrap ghép cặp 2.000 lần theo nhóm query của majority, giữ prediction
  cố định: khoảng 95% delta Macro-F1 DN-unit leaf so với majority, theo điểm
  phần trăm, seed 0 [-1,31; 3,81], seed 1 [-1,21; 4,19], seed 2 [-1,69; 4,54].
  Đều chứa 0, nên chưa kết luận cải thiện chắc chắn. Các nhóm có thể còn
  phụ thuộc theo sự kiện/tweet, dev đã xem nhiều lần, không hiệu chỉnh nhiều
  phép so sánh. Không gộp seed thành quan sát độc lập.
- Chưa dùng test để chọn tham số/phương pháp. Cần khóa ứng viên rồi đo test
  đã loại trùng với cả train và dev trước khi kết luận cuối cho báo cáo.

## Artifact và cách chạy

Các lượt đã hoàn tất, không cần gửi Kaggle để lấy các kết quả bên trên:

- ZLaP: `results/analysis/clip_zlap/07ba79dff51f/`.
- DN-unit: `results/analysis/clip_dn_unit/d70959a0a21c/`.
- Đối chiếu tác giả: `reference_verification.json`,
  `dn_reference_verification.json` trong thư mục code này.
- `analysis.json` chứa đủ metric từng lớp/seed; `audit.json` chứa kiểm chứng.
  NPZ lưu score/mean và TSV lưu prediction, không pickle, không trọng số model.

Tái lập từ artifact đã lưu, tạo thư mục kết quả mới:

```sh
venv/bin/python -B experiments/clip_zlap/run.py
venv/bin/python -B experiments/clip_zlap/dn_unit.py
```

Kiểm thử và đối chiếu (repo tham chiếu chỉ cần khi chạy lệnh verify):

```sh
venv/bin/python -B -m unittest experiments.clip_zlap.test_zlap experiments.clip_transductive.test_ensemble experiments.clip_transductive.test_fusion -b -q
venv/bin/python -B experiments/clip_zlap/verify_reference.py --reference-dir /private/tmp/vlm-zlap-reference --output /private/tmp/zlap_verification.json --config experiments/clip_zlap/config.json
venv/bin/python -B experiments/clip_zlap/verify_dn_reference.py --reference-dir /private/tmp/vlm-dn-reference --output /private/tmp/dn_verification.json
```
