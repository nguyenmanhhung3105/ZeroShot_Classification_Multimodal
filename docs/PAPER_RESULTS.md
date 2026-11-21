# Kết quả paper đã lưu để đối chiếu offline

Nguồn số dùng chung: [configs/paper_results/references.json](../configs/paper_results/references.json).
Bản chụp CSV tác giả: [configs/paper_results/sources/](../configs/paper_results/sources/).
Đã kiểm chứng ngày 03/11/2026. Chỉ bảng **OpenAI CLIP ViT-L/14, ảnh 224px**
đang dùng để kiểm chứng dự án; việc lưu bảng không kích hoạt thêm model hay job.

Các số dưới đây là số **paper công bố**, đơn vị %, không phải kết quả chạy của dự án.

| Dataset | Waffle: CLIP photo | Waffle: DCLIP photo | Waffle | DCLIP: CLIP gốc | DCLIP gốc | CLIP paper gốc |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet | 73.37 | 75.26 | 75.30 ± 0.04 | 71.58 | 75.00 | 75.3 |
| ImageNetV2 | 67.90 | 69.72 | 69.48 ± 0.08 | 65.33 | 69.30 | — |
| CUB200 | 62.24 | 63.53 | 64.18 ± 0.13 | 63.08 | 63.46 | — |
| EuroSAT | 56.03 | 58.72 | 61.17 ± 0.35 | 41.48 | 48.66 | 59.9 |
| Places365 | 40.46 | 42.60 | 42.26 ± 0.10 | 39.00 | 40.55 | — |
| Food101 | 92.55 | 92.81 | 93.31 ± 0.09 | 91.79 | 92.44 | 92.9 |
| Oxford Pets | 93.30 | 93.89 | 91.98 ± 0.11 | 88.25 | 92.23 | 93.5* |
| DTD | 52.87 | 56.60 | 53.94 ± 0.29 | 51.33 | 54.36 | 55.3 |

Nguồn chính: [Waffle bảng 10, v2](https://arxiv.org/html/2306.07282),
[DCLIP bảng 1, v2](https://arxiv.org/html/2210.07183),
[CLIP bảng 11, trang PDF 43](https://arxiv.org/pdf/2103.00020).
CLIP dùng hàng L/14, không dùng hàng L/14-336px.
“—” nghĩa không có ô của dataset đó trong bảng 11 đã lưu; JSON ghi `null`, không đoán số.

## Cách đọc đúng

- Ba cột Waffle dùng cùng giao thức photo prompt trong paper Waffle.
  Waffle dùng 15 cặp descriptor, trung bình 7 seed; ± là standard deviation.
- Hai cột DCLIP dùng prompt gốc: tên lớp và tên lớp kèm mô tả.
  ImageNet có dấu chấm cuối câu theo repo tác giả. Phải xác minh split khi tái lập;
  chẳng hạn loader DCLIP DTD dùng `val`, còn dự án dùng test partition 1.
- Cột CLIP paper gốc có prompt riêng theo dataset và ensemble; ImageNet dùng
  80 templates. Cột này là mốc tham khảo riêng, không phải target cho `clip_photo`.
  (*) Oxford Pets dùng mean per class theo bảng 9; các số còn lại trong cột này dùng accuracy.
  EuroSAT paper CLIP có tập đánh giá 5.000 ảnh theo bảng 9, khác 27.000 ảnh của dự án.
- Số paper đã làm tròn và số CSV tác giả được lưu riêng. Ví dụ CLIP photo trên CUB:
  paper **62.24%**, CSV **62.27131485939026%**. Giữ nguyên cả hai, không coi đây
  chỉ là sai số làm tròn. Khi đối chiếu runtime, dùng CSV đầy đủ nếu có như trước.
- CSV tác giả lưu population std (ddof=0); `summary.tsv` dự án dùng sample std (ddof=1).
  Không trộn hai loại độ lệch chuẩn khi báo cáo.
- Kết quả trên MetaCLIP2 và Fakeddit nằm trong các run riêng. Không gán các số
  OpenAI ViT-L/14 này cho model hoặc taxonomy khác. Các bảng lưu ở đây không cung cấp
  mốc Fakeddit tương đương trực tiếp.

## Cách dùng trong code

`paper_reference.reference_targets(dataset)` đọc JSON và bản chụp CSV trong dự án,
kiểm tra checksum, không tra web hoặc đọc repo tác giả để lấy lại con số.
Đã khai báo job tham chiếu cho đủ 8 dataset: DTD, ImageNetV2, ImageNet, CUB,
Food101, Pets, EuroSAT và Places365.
Bundle code Kaggle tự kèm registry và ba CSV.

Pets giữ ID lớp trong dataset đã chuẩn bị. Prompt được sinh theo thứ tự JSON tác giả
rồi mới sắp cột theo tên lớp chính xác về ID dataset; không sinh lại RNG theo thứ tự mới.
`author_class_order` và `author_to_dataset_label` được ghi vào manifest/frozen plan.
Sai hoặc thiếu tên lớp làm dừng; không đoán nhãn theo số lớp.
EuroSAT dùng đủ 27.000 ảnh theo Waffle upstream; hai đối chứng prompt gốc trên tập
này không xác nhận tái lập split `EuroSATVal` trong loader DCLIP gốc.

Mỗi run mới lưu `paper_targets.json`, giá trị tham chiếu và SHA256 registry vào
`frozen_plan.json`/`run.json`; `paper_comparison.json` tiếp tục giữ số paper, số CSV,
chênh lệch và ngưỡng gần khớp đặt trước 0.10 điểm phần trăm.
Kết quả run cũ được giữ nguyên.

Registry ghi URL paper, phiên bản/bảng, backbone, giao thức, protocol đánh giá
của dự án, URL/commit/SHA256 của ba CSV đã chụp. Không cần tải lại paper/CSV
cho các lần đối chiếu sau; chỉ cập nhật chủ động khi muốn đổi nguồn tham chiếu.
