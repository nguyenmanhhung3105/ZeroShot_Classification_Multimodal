# Cấu trúc thư mục kết quả

Mọi **lượt chạy mới** được đặt theo đúng sáu cấp:

```text
results/<phương_pháp>/<dataset>/<task>/samples_<số_mẫu>/<model>/<task>_<mã_lượt>/
```

Ví dụ lượt MetaCLIP2 dùng nhánh ảnh–text trực tiếp trên CrisisMMD humanitarian:

```text
results/direct_pair_fusion/crisismmd/humanitarian/samples_1000/metaclip2_vith14/humanitarian_45948c7b7aa6/
├── summary.tsv             chỉ số tổng hợp
├── run_artifacts.zip       log dự đoán, manifest và chi tiết trong archive
├── run_receipt.json        thông tin xác nhận lượt Kaggle
└── vlm-runs.log            log worker
```

Tên dataset là `crisismmd`, `fakeddit` hoặc `mmimdb`. Task có thư mục riêng: CrisisMMD dùng `informativeness` hoặc `humanitarian`, Fakeddit dùng `2way` hoặc `6way`, MM-IMDb dùng `genres`. Trong bốn lượt MM-IMDb cũ, tên lượt còn tiền tố `multi_label_` nhưng thư mục task là `genres` vì đó là cùng bài toán. `samples_N` là số mẫu trong summary nếu lượt hoàn tất; với lượt lỗi không có dòng kết quả, dùng số mẫu đã yêu cầu trong job/receipt. Bốn lượt cũ ở `samples_unknown` chỉ có summary rỗng đã được xoá.

Thư mục phương pháp được tạo tự động từ YAML của job. Chạy chuẩn dùng `baseline`; bật quan hệ dùng `agreement_fusion`, `direct_pair_fusion` hoặc `class_aware`; nghiên cứu dùng `adapted_dclip`, `adapted_waffle`, `upstream_clip`, `upstream_dclip`…; ensemble dùng `ensemble`. Khi adapted/upstream bật thêm quan hệ ảnh–text, tên ghép như `adapted_dclip_agreement_fusion` tách nó khỏi phương pháp gốc. Mỗi phương pháp mới có thư mục riêng khi launcher nhận một tên phương pháp mới. Lượt một model dùng tên model như `metaclip2_vith14`; OpenAI CLIP ViT-B/32 dùng `openai_clip_vit_b_32`. Lượt có nhiều model giữ nguyên trong `multi_model`, không tách archive chung. Seed vẫn là thông tin của từng lượt.

Job khóa giao thức ảnh-only dùng tiền tố `image_only_`: `image_only_baseline`, `image_only_adapted_<method>` hoặc `image_only_upstream_<method>`. Standard/adapted đặt `image_weight=1.0` và tắt quan hệ ảnh–text; upstream chỉ xuất `original_image`. Năm lượt upstream lịch sử chỉ chứa `original_image` đã được chuyển vào nhóm này, còn archive có cả `original_image` và `multimodal` giữ nguyên ở `upstream_suite`. Xem [phép so sánh CrisisMMD ảnh-only](crisismmd_image_only_comparison.md).

Các gói lịch sử chứa **nhiều phương pháp trong cùng một archive** được giữ nguyên dưới `adapted_suite` hoặc `upstream_suite`; tách file sẽ làm mất mối liên hệ giữa các phép thử. Gói chứa nhiều dataset hoặc task dùng `mixed/dataset` vì không thể đặt toàn bộ gói dưới một task riêng. Thử nghiệm 2, 5, 30, 100 mẫu có `samples_2`, `samples_5`, `samples_30`, `samples_100` tương ứng. Báo cáo phân tích offline được gom dưới `results/analysis/<tên_phân_tích>/<dataset>/<task>/samples_N/<model>/<lượt>/`, tách khỏi thư mục các phương pháp chạy model.

`results/_system/` chứa cache, staging Kaggle, báo cáo nhãn và [bảng đường dẫn cũ → mới](../../results/_system/migration_map.json). Đây là dữ liệu vận hành, không phải kết quả model. Các `run_receipt.json`, log và archive lịch sử có thể vẫn ghi đường dẫn tại thời điểm chạy; file gốc được giữ nguyên để không sửa bằng chứng thí nghiệm. Khi cần tìm vị trí hiện tại, dùng bảng migration. `results1/` nằm ngoài phạm vi lần sắp xếp này.

Mở trực tiếp thư mục phương pháp/dataset/task/số mẫu/model để xem các lượt. Nếu muốn lọc bằng terminal mà không tạo file chỉ mục:

```bash
python -B scripts/list_results.py --method direct_pair_fusion --dataset crisismmd --samples 1000
```

Khi Kaggle mất kết nối sau khi đã gửi job, `--resume-stage` vẫn nhận cả đường dẫn staging cũ đã in trước khi chuyển và đường dẫn hiện tại `results/_system/launches/staging/<stage>`; nó tải về cây kết quả mới.
