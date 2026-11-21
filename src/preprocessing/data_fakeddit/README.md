# Nguồn và preprocessing Fakeddit

Entry point dùng từ thư mục gốc: `scripts/prepare_fakeddit_splits.py`. Xem tham số bằng `venv/bin/python -B scripts/prepare_fakeddit_splits.py --help`; lệnh help không đọc dataset. Lần dọn này chỉ đưa code trở lại cấu trúc chính, không chạy preprocessing trên dữ liệu thật.

| Split của dự án | TSV nguồn chính thức dưới `data/raw/fakeddit/` | `source_split` |
|---|---|---|
| train | `multimodal_train.tsv` | train |
| dev | `multimodal_validate.tsv` | validate |
| test | `multimodal_test_public.tsv` | test_public |

Ba pool độc lập; không chia ngẫu nhiên train thành dev/test. Mặc định mỗi split 3.000 mẫu, seed 42, lấy quota theo tỷ lệ sáu lớp của pool sau lọc. Giữ `2_way_label` nguồn (1=Real, 0=Fake), `6_way_label` nguồn (0=True, 1..5=Fake subtype). Few-shot chuyển sang binary 0=Real, 1=Fake trong runner, không sửa nhãn TSV nguồn.

`prepare_splits.py` giữ nguyên logic builder từ `archive/legacy_20261103/src/preprocessing/data_fakeddit/prepare_splits.py`: chuẩn hóa HTML/Unicode NFC/khoảng trắng, kiểm tra ID/nhãn/URL/hasImage, lọc độ dài tiêu đề, xác thực ảnh, loại trùng trong và giữa split theo ID/URL/title/bytes/pixel/dHash. Thứ tự xây test → dev → train, hỗ trợ resume, chỉ công bố khi verify thành công. Việc đọc annotations và tải ảnh chỉ diễn ra khi chủ động chạy builder.

- `helpers.py`: nguyên thân năm hàm `digest`, `text_value`, `quotas`, `image_ok`, `write_tsv` trích từ script cohort cũ; bỏ phụ thuộc import CrisisMMD/MM-IMDb.
- `preprocess_fakeddit.py`, `check_fakeddit_ready.py`: wrapper preprocessing/verify giữ từ bản cũ. Khi tạo cohort hiện tại ưu tiên entry point trong `scripts/` để hợp toàn bộ danh sách dev đã xem từ lịch sử và các run hiện tại.
- `configs/preprocessing/fakeddit_legacy_base.yaml`: bản sao cấu hình nền cũ để giữ định dạng `experiment.yaml` xuất ra; không phải cấu hình suy luận mặc định hiện tại. Các dataset khác trong base bị builder đặt `enabled=false`.
- Downloader đời đầu vẫn nguyên trạng tại `archive/legacy_20261103/src/preprocessing/data_fakeddit/image_downloader.py`. Builder hiện dùng `fetch_image` với timeout và kiểm tra ảnh; không cần chạy downloader cũ.

Đầu ra mỗi split gồm `fakeddit_sample.tsv`, `manifest.json`, `experiment.yaml`, `images/`; thư mục chung có `audit.json`. TSV ghi `source_file`, `source_row` (dòng gốc, bắt đầu từ 2), `source_split`, `split`, `label_source`. Manifest ghi hash nguồn, quota, số mẫu bị loại, hash TSV, hash từng ảnh/text và `cohort_id`.

Giữ phân biệt `data/processed/fakeddit_splits_v2` và `data/processed/fakeddit_splits_min5_v1`. Entry point hiện mặc định min5, lọc `len(title.replace(' ', '')) > 5`; nguồn và checksum của cohort thực tế phải lấy từ manifest đã tạo, không suy từ default code. Thí nghiệm few-shot pin hash manifest train/dev trong `configs/fakeddit_fewshot_task1_min5.yaml`. Không tạo lại dữ liệu chỉ để làm gọn thư mục.

Kiểm thử `tests/test_fakeddit_splits.py` dùng TSV/ảnh tổng hợp trong thư mục tạm, kiểm tra nguồn tách biệt, quota, dedup, checksum, lỗi mạng và timeout; không chạm dataset thật.
