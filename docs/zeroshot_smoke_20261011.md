# Tiếp tục nghiên cứu zero-shot ngày 11-10-2026

Kế thừa phiên `01a0ee28-726b-7de2-810a-40b92bacc381`, tên “Phân tích tổng thể
dự án”. Phiên tiếp quản đã đổi tên thành “Nghiên cứu zeroshot”. Phần thiếu lúc
bàn giao là chạy checkpoint thật, kiểm tra dữ liệu dự án và đối chiếu phép chấm
điểm. Những bước chạy thử này đã hoàn thành; đánh giá quy mô lớn vẫn chưa làm.

## Phạm vi và môi trường

- Python 3.12 từ `venv/bin/python`; Python mặc định của máy chưa có phụ thuộc.
- OpenAI CLIP `ViT-B/32`, checkpoint chính thức, thiết bị CPU ở các lượt kiểm tra
  sau sửa. Checkpoint đã tải vào cache mặc định `~/.cache/clip`.
- Giữ nguyên ba repo và commit trong `configs/research/sources.json`.
- Bốn phương pháp: `clip`, `waffle`, `dclip_labels`, `dclip`; Waffle dùng seed 0
  và 15 cặp. Descriptor, prompt text, fusion và quy tắc quyết định không đổi.
- Đọc tối đa 5 dòng đầu mỗi bảng processed, không truy cập dữ liệu thô. Cả 5
  dòng Fakeddit đầu đều vượt bộ lọc độ dài text hiện tại, nên loader dừng sau
  batch đầu; không cần đọc thêm dòng để bù.
- Mỗi tác vụ có 5 ảnh hợp lệ, không bỏ ảnh, xuất 8 dòng kết quả: 4 phương pháp
  × 2 chế độ `original_image`/`multimodal`.
- Không đổi YAML chính hoặc YAML nghiên cứu đang dùng. Các cấu hình chạy thêm
  được lưu riêng trong thư mục suite kết quả.

## Kết quả chạy thử

Bảng dưới chỉ ghi **Macro-F1 của chế độ ảnh + text**, dùng cùng bộ mẫu cho các
phương pháp trong mỗi tác vụ. Đây là kiểm tra hoạt động của pipeline trên 5 mẫu,
không phải đánh giá chất lượng đại diện hay bảng xếp hạng phương pháp.

| Dataset/tác vụ | CLIP | WaffleCLIP | DCLIP tên lớp | DCLIP mô tả |
|---|---:|---:|---:|---:|
| CrisisMMD humanitarian | 0.350000 | 0.229167 | 0.187500 | 0.350000 |
| CrisisMMD informativeness | 0.444444 | 0.444444 | 1.000000 | 0.444444 |
| Fakeddit 6way | 0.083333 | 0.083333 | 0.083333 | 0.083333 |
| Fakeddit 2way | 0.285714 | 0.166667 | 0.375000 | 0.285714 |
| MM-IMDb genres | 0.050725 | 0.085507 | 0.050725 | 0.137681 |

Macro-F1 vẫn tính trên toàn bộ taxonomy đã cấu hình, kể cả lớp không có mặt
trong 5 mẫu. Vì thế không so trực tiếp giá trị này giữa các tác vụ. Với MM-IMDb,
4/5 plot dài hơn context 77 token; chunking chia tổng cộng 12 đoạn với số đoạn
từng mẫu `[2, 1, 2, 2, 5]`. Audit đo độ dài plot gốc; con số này không có nghĩa
runner đã bỏ phần còn lại của plot khi bật chunking.

Kết quả và cấu hình đã khóa:

- [Humanitarian sau sửa](../results/upstream_suite/crisismmd/humanitarian/samples_5/openai_clip_vit_b_32/humanitarian_experiment_djym_cbk/summary.tsv).
- [Informativeness, Fakeddit 6way, MM-IMDb](../results/upstream_suite/mixed/dataset/samples_5/openai_clip_vit_b_32/dataset_smoke_suite_ua4bjvve/experiment_62_d5hs1/summary.tsv).
- [Fakeddit 2way](../results/upstream_suite/mixed/dataset/samples_5/openai_clip_vit_b_32/dataset_smoke_suite_ua4bjvve/experiment_yo4gy0ph/summary.tsv).
- [Báo cáo kiểm tra cả năm tác vụ](../results/upstream_suite/mixed/dataset/samples_5/openai_clip_vit_b_32/dataset_smoke_suite_ua4bjvve/validation_report.json).
- [Thời gian suite và cấu hình chạy thêm](../results/upstream_suite/mixed/dataset/samples_5/openai_clip_vit_b_32/dataset_smoke_suite_ua4bjvve/suite_report.json).

Lượt humanitarian sau sửa mất khoảng 7,4 giây trên CPU, tính từ lúc nạp runner
đến khi ghi xong kết quả, không tính tải checkpoint. Suite ba tác vụ mất khoảng
16,5 giây; lượt Fakeddit 2way mất khoảng 3,1 giây. Đây là thời gian chạy thử,
chưa đo peak bộ nhớ hoặc hiệu năng trên 1.000 mẫu.

## Kiểm tra độ đúng và sửa báo cáo

Đã kiểm tra log của cả năm tác vụ: cùng fingerprint bộ mẫu giữa các phương pháp,
điểm nhánh text giống hệt nhau, điểm hữu hạn, xác suất một nhãn có tổng bằng 1,
và metric trong `seed_summary.tsv` khớp `summary.tsv`.

Riêng humanitarian, nạp lại checkpoint trên CPU và đối chiếu toàn bộ 8 output
với mã hóa CLIP trực tiếp rồi lấy mean cosine. Các hàm tác giả cho cosine khớp
phép tính trực tiếp; nhãn dự đoán của tất cả mẫu khớp log. Sai số điểm đầu ra
lớn nhất khoảng `2.48e-6`.
Báo cáo đối chiếu `parity_report.json` được nhắc trong ghi chú cũ không có trong
workspace hiện tại; không còn tệp để kiểm tra nguồn/commit, bộ mẫu, thiết bị và
sai số từng output. Đây là kiểm tra phép
chấm điểm trên cùng prompt đã khóa, không tái lập benchmark hoặc GPT-3 descriptor
gốc của paper. Script kiểm tra được lưu cạnh báo cáo để chạy lại:

```sh
venv/bin/python -B results/upstream_projects/verification_k_8bp_ms/verify_smoke.py \
  results/upstream_suite/crisismmd/humanitarian/samples_5/openai_clip_vit_b_32/humanitarian_experiment_qzll_j5j
```

Hai thay đổi nhỏ trong `src/research/run_upstream.py`:

1. Tổng hợp seed lấy metric qua `compact_summary`, cùng quy tắc với summary
   chính. Trước đó balanced accuracy/samples-F1 có tiền tố `fusion_` từ audit
   nên bị bỏ sót khi tìm tên metric không tiền tố.
2. Ghi `device` thực tế vào summary và manifest để phân biệt `auto` trong YAML
   với thiết bị đã dùng. Không thay đổi công thức suy luận hoặc metric.

Các kiểm thử của runner kiểm tra thêm hai trường hợp này. Toàn bộ 88 kiểm thử
đều đạt với dữ liệu giả; lượt chạy thật sau sửa giữ nguyên metric humanitarian
so với trước sửa. Các thư mục kết quả cũ và checkout upstream được giữ nguyên.

```sh
venv/bin/python -B -m unittest discover -s tests -b
venv/bin/python -B src/research/run_upstream.py --prepare-only
venv/bin/python -B src/research/run_upstream.py
```

## Việc còn lại

- Chưa đánh giá trên bộ mẫu lớn hoặc sweep nhiều seed. Lượt này giữ giới hạn
  5 dòng và không chạy inference lớn theo quy tắc làm việc hiện tại.
- Chưa chạy MetaCLIP2/DFN/SigLIP2 trên đúng bộ mẫu này để đối chiếu với nhánh
  upstream; chưa thể kết luận nghiên cứu tốt hơn hệ thống chính.
- Chưa có kiểm định thống kê, peak bộ nhớ, hoặc kết luận về descriptor tốt hơn.
- Descriptor vẫn là mô tả cố định theo taxonomy, không phải descriptor GPT-3
  gốc. Phần fusion và MM-IMDb vẫn là mở rộng của dự án.
- Khi được yêu cầu đánh giá lớn hơn, khóa cùng bộ mẫu, backbone và nhánh text
  giữa các phương pháp, chạy nhiều seed Waffle và báo cáo mean/std; không chọn
  seed tốt nhất hoặc sửa prompt theo lỗi đánh giá vừa xem.

Các ID tweet trong 5 dòng đầu được lưu dưới dạng ký hiệu khoa học với dấu phẩy,
có ID lặp. Lượt này giữ nguyên dữ liệu và dùng cả đường dẫn ảnh trong fingerprint
bộ mẫu; không dùng mỗi ID tweet làm khóa duy nhất để ghép kết quả. Việc xác minh
hoặc sửa ID nguồn là nhiệm vụ riêng vì không truy cập dữ liệu thô trong lượt này.

Không commit hoặc push.
