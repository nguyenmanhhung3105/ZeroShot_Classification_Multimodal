# Fakeddit: StatA real/fake ở tầng 1, CALIP ở tầng 2

Nhánh theo yêu cầu mới: giữ StatA làm gate binary image-only, thêm CALIP ở bước
phân loại kiểu fake. Không thay gate bằng CALIP hoặc trộn EM vào gate của job này.
Nhánh Majority gốc và kết quả trước được giữ riêng.

## Luồng và các đối chứng

1. Một MetaCLIP2 ViT-H/14 cố định mã hóa ảnh thành global và patch feature đã
   project cùng không gian text. Title mã hóa một lần. Có một forward ảnh tham
   chiếu ở batch đầu để xác nhận global giống `model.encode_image`.
2. StatA nhận global feature và hai prototype unit fake/real, nhóm query không
   nhãn tối đa 75, seed 0/1/2. Gate này giống nhau trong mọi đối chứng tầng 2.
3. Nếu real, trả `True` (6way=0). Nếu fake, chọn trong ID 1..5 bằng một trong:
   CLIP; CALIP; hoặc fusion xác suất CLIP/CALIP 0,5/0,5.

CALIP chỉ chấm sáu lớp fine, không chấm hoặc thay gate binary. Dù tính điểm đủ
sáu lớp để có flat control, hierarchy chỉ chọn năm loại fake sau gate.

Có hai modality: image-only và ảnh–tiêu đề. Với MM, mọi nhánh giữ cùng posterior
title từ raw mean text, weight ảnh 0,75/text 0,25. Image prototype CLIP/CALIP đều
unit để tách tác động normalization. Vì vậy, không coi chênh lệch với **legacy
raw-mean MM** từ run cũ là lợi ích thuần CALIP; so với đối chứng CLIP trong run mới.

| Nhánh | Flat | Hierarchy dùng gate StatA |
|---|---|---|
| CLIP | flat_clip_image / flat_clip_mm | hierarchy_stata_clip_image / hierarchy_stata_clip_mm |
| CALIP | flat_calip_image / flat_calip_mm | hierarchy_stata_calip_image / hierarchy_stata_calip_mm |
| Fusion 0,5/0,5 | flat_clip_calip_image / flat_clip_calip_mm | hierarchy_stata_clip_calip_image / hierarchy_stata_clip_calip_mm |

Sáu flat tất định một lượt, sáu hierarchy ba seed: 12 method, 24 tổ hợp method/seed.
Tiêu chí chính macro-F1 đủ 6 lớp, kèm accuracy/recall từng lớp. Không chọn seed tốt nhất.

## CALIP và artifact

Tái sử dụng `calip.py` đã kiểm tra: attention scale 2, beta 1/1,12/0,02, logit
scale 100 theo commit `83e21b6b985934dce6e8414c26300671b4825841`. Đây là tham số
tham chiếu, chưa tối ưu cho Fakeddit/MetaCLIP2. Temperature 1 và fusion cố định,
không bảo đảm posterior CALIP được calibration ngang CLIP.

Không lưu patch cho cả cohort: chỉ giữ batch trong RAM/GPU và tối đa 3 ảnh ở
`patch_verification.npz`. `extraction.json` ghi grid/projection/check global.
`replay_scores.npz` giữ global/title feature, logits CALIP, posterior các nhánh,
group IDs, binary posterior StatA và dự đoán. Không tải lại dataset.

`binary_summary.tsv` và `binary_metrics_by_seed.tsv` xuất metric 2way CLIP/StatA
riêng (accuracy, macro-F1, balanced accuracy); không nhầm với bảng `summary.tsv`
6way. `comparisons.tsv` so CALIP/fusion với CLIP cùng gate, seed và modality.
`branch_confidence.json` giúp kiểm tra posterior quá sắc. Oracle dùng binary
label thật chỉ ở `oracle_diagnostic.json`, không đưa vào bảng mô hình.

## Lệnh chạy

Smoke 5 ảnh trước:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job fakeddit_stata_calip_smoke --use-kaggle true --wait
```

Sau khi smoke hợp lệ, full 3.000 dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job fakeddit_stata_calip_dev --use-kaggle true --wait
```

Mỗi lệnh kiểm tra quota, chặn trùng, gửi job riêng, theo dõi 30 giây/lần, tải ZIP,
xác thực checksum và in bảng. Lỗi kết nối dùng `--resume-stage` đã in, không gửi lại.
Thêm `--prepare-only` chỉ đóng gói code và kiểm tra cấu hình, không gọi API/model/data.
Không chạy full ngầm sau smoke. Không dùng test, không tạo test lock, không tìm grid.

## Kiểm tra triển khai

Đã test gate bất biến khi đổi logits CALIP, real luôn về True, các nhánh dùng
cùng gate/seed, fusion đúng hệ số, fallback nhóm <=3, end-to-end bằng solver giả,
metric binary tách 6way, oracle không xếp hạng và whitelist notebook riêng.
Extractor được kiểm tra trực tiếp bằng ViT nhỏ ngẫu nhiên, không tải weights:
global parity, khôi phục output_tokens và replay score từ patch đã lưu.
Các test CALIP công thức/hoán vị đã có cũng được chạy lại.
Chưa chạy smoke/full trên ảnh thật trong lượt triển khai.
