# LAME trên CLIP hai tầng: kết quả không cải thiện

[Paper CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html),
[repo](https://github.com/fiveai/LAME), commit
`d2e5f63090bc1c8129bf7cbd781029a5955e1a67`.

LAME hiệu chỉnh posterior theo quan hệ lân cận giữa embedding ảnh; không
đổi trọng số encoder/classifier và không dùng nhãn thật. Khác ZLaP, không
tạo node text trong đồ thị; khác InMaP, không học lại proxy.

## Cấu hình cố định trước khi đo

- Hai affinity tác giả hỗ trợ: RBF (mặc định) và kNN, k=5. Không sweep k,
  lambda, temperature hoặc chọn seed sau kết quả.
- bound_lambda=1, tối đa 100 bước, dừng theo energy như code tác giả,
  force_symmetry=False, scale CLIP checkpoint 100.
- RBF dùng sigma tự tính từ khoảng cách hàng xóm thứ năm, đúng code tác
  giả; giá trị sigma=1 trong config không thay công thức này.
- Nhóm 75 ảnh, grouping seed 0/1/2 như các đối chứng EM/StatA của dự án.
  Đây là protocol của dự án, không phải tái lập benchmark LAME gốc.
- Cổng dùng score semantic CLIP gốc; tầng chi tiết dùng score DN-unit đã
  cache. Kiểm tra source hash và ID, không đọc dataset hoặc encode lại ảnh.
- Đo root-only, majority + LAME ở tầng DN-unit, và LAME cả hai tầng.
  Majority + DN-unit nguyên trạng làm đối chứng trong mọi lượt.

## Kết quả ngày 29-10-2026

Macro-F1 (%), trung bình ba grouping seed trên cùng cohort:

| Kernel / phương án | Binary train 3.000 | 8 nhãn train 3.000 | 8 nhãn dev 2.219 |
|---|---:|---:|---:|
| Majority + DN-unit đối chứng | 80,83 | 49,22 | 50,39 |
| RBF, chỉ cổng LAME | 35,03 | 34,20 | 33,75 |
| RBF, majority + LAME tầng DN-unit | 80,83 | 37,50 | 36,08 |
| RBF, LAME cả hai tầng | 35,03 | 24,61 | 22,69 |
| kNN, chỉ cổng LAME | 46,47 | 37,67 | 37,48 |
| kNN, majority + LAME tầng DN-unit | 80,83 | 46,09 | 47,11 |
| kNN, LAME cả hai tầng | 46,47 | 34,80 | 35,86 |

Không có binary ground truth dev, giữ null. Kết quả giảm mạnh, không chọn
LAME ở các cấu hình này. Không tiếp tục hạ lambda hoặc chọn batch/seed theo
dev để gọi là cải thiện đã kiểm chứng. Kết luận chỉ áp dụng cho protocol
nhóm 75, score CLIP/DN-unit và hai affinity đã thử, không phủ định mọi
thiết lập LAME hoặc mọi thuật toán dùng đồ thị.

## Kiểm chứng và giới hạn

- Bốn unit test đạt: lambda=0 giữ prior, hàng xóm sửa mẫu mơ hồ, bao phủ
  mọi ID/batch, singleton/zero-sigma fallback và từ chối dữ liệu không hợp lệ.
- Đối chiếu trực tiếp class kernel và hàm optimizer thuần bằng AST repo tác
  giả, không import Detectron2/model/dataset. Bốn ca giả và **840 nhóm task
  thực** khớp toàn bộ posterior và nhãn trên CPU FP32.
- Giữ cách topk rồi bỏ hàng xóm đầu tiên như tác giả, kể cả tình huống tie;
  diagnostics ghi số phần tử đường chéo khác zero. RBF giữ self-affinity.
- Singleton hoặc sigma=0 dùng prior ở bản dự án; không xảy ra trong các
  nhóm thực của hai cohort này, được kiểm tra riêng bằng fixture.
- Audit đối chiếu hash source/output, NPZ ID và xác suất, grouping,
  prediction và mọi metric từng lớp trên đủ các seed/kernel/split.
- Train/dev đã xem nhiều lần, không phải test độc lập.

Artifact: `results/analysis/clip_lame/e257fe3950a8/`, có frozen plan,
manifest, summary, diagnostics, posterior/prediction và audit.

Tái lập (lượt hiện tại đã hoàn tất, không cần Kaggle):

```sh
venv/bin/python -B experiments/clip_lame/run.py
venv/bin/python -B -m unittest experiments.clip_lame.test_lame -b -q
```
