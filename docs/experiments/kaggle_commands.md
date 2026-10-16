# Lệnh chạy Kaggle riêng theo model, dữ liệu và phương pháp

Từ thư mục gốc dự án, chạy lệnh sau để **in toàn bộ lệnh cụ thể**. Lệnh liệt kê không đọc dataset, không tải model và không gửi job lên Kaggle:

```sh
venv/bin/python -B scripts/list_kaggle_commands.py
```

Danh sách hiện có 363 lệnh độc lập trong [`kaggle_commands_all.txt`](kaggle_commands_all.txt). Chọn **một dòng** cần chạy rồi dán vào terminal; đợi `[RESULT]` và kết quả tải về local trước khi chạy dòng tiếp theo vì các job dùng chung kernel slug `vlm-runs`. Nếu sửa YAML, chạy lại lệnh liệt kê để lấy danh sách mới.

Launcher từ chối gửi job Kaggle nếu một lệnh chọn nhiều model, phương pháp, backbone, output hoặc nhiều seed Waffle. Chọn một cấu hình bằng `--model`, `--method`, `--backbone`, `--output`, `--seed` tương ứng. Ngoại lệ là các job ensemble và `margin_majority` có chủ ý chạy ba model tuần tự.

Để tìm lệnh nhanh hơn:

```sh
venv/bin/python -B scripts/list_kaggle_commands.py --kind standard
venv/bin/python -B scripts/list_kaggle_commands.py --kind ensemble
venv/bin/python -B scripts/list_kaggle_commands.py --kind uncertainty
venv/bin/python -B scripts/list_kaggle_commands.py --kind adapted
venv/bin/python -B scripts/list_kaggle_commands.py --kind upstream
venv/bin/python -B scripts/list_kaggle_commands.py --kind demo
venv/bin/python -B scripts/list_kaggle_commands.py --kind relation_ablation
venv/bin/python -B scripts/list_kaggle_commands.py --job crisismmd_info
venv/bin/python -B scripts/list_kaggle_commands.py --job adapted_crisismmd_info
venv/bin/python -B scripts/list_kaggle_commands.py --job upstream_crisismmd_info
venv/bin/python -B scripts/list_kaggle_commands.py --job crisismmd_info_image_only
venv/bin/python -B scripts/list_kaggle_commands.py --job upstream_crisismmd_info_image_only
```

| Dữ liệu/tác vụ | Standard và ensemble | Adapted | Upstream |
|---|---|---|---|
| Fakeddit 2 nhãn | `fakeddit_2way` | `adapted_fakeddit_2way` | `upstream_fakeddit_2way` |
| Fakeddit 6 nhãn | `fakeddit_6way` | `adapted_fakeddit_6way` | `upstream_fakeddit_6way` |
| CrisisMMD informativeness | `crisismmd_info` | `adapted_crisismmd_info` | `upstream_crisismmd_info` |
| CrisisMMD humanitarian | `crisismmd_humanitarian` | `adapted_crisismmd_humanitarian` | `upstream_crisismmd_humanitarian` |
| MM-IMDb genres | `mmimdb_genres` | `adapted_mmimdb_genres` | `upstream_mmimdb_genres` |

- `standard`: từng model cho từng tác vụ, gồm các job quan hệ, ảnh-only và entropy; 23 lệnh.
- `ensemble`: ba model chung một lượt, một tác vụ mỗi lượt; 5 lệnh.
- `uncertainty`: 4 lệnh CrisisMMD riêng cho bỏ phiếu 2/3 có cổng margin và chuyển prompt khi margin thấp; xem [giao thức và trạng thái](crisismmd_uncertainty_methods.md).
- `adapted`: từng model × phương pháp/seed; các job ảnh-only giới hạn vào phương pháp đã khai báo; 185 lệnh.
- `upstream`: từng backbone × phương pháp × output; các job ảnh-only chỉ in `original_image`; 128 lệnh. Waffle dùng seed 0 theo `configs/research/upstream.yaml`.
- `demo`: từng model × năm tác vụ, tối đa 5 mẫu; 15 lệnh. `relation_ablation`: từng model trên Fakeddit 6way, tối đa 5 mẫu; 3 lệnh.

Ba ví dụ chạy thật, mỗi lệnh chỉ một model và một tác vụ:

```sh
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info --model metaclip2_vith14 --ensemble false --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job adapted_crisismmd_info --model dfn5b_vith14 --method dclip --max-samples 1000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --job upstream_crisismmd_humanitarian --backbone ViT-B/32 --method clip --output original_image --use-kaggle true
```

Giới hạn mặc định lấy từ từng job trong `configs/launcher.yaml`: phần lớn job chuẩn là 1.000 mẫu, nghiên cứu thông thường là 30 mẫu, các job CrisisMMD ảnh-only là 1.000 mẫu; demo và relation là 5 mẫu. Muốn đổi số mẫu cho **một lệnh**, thêm `--max-samples N`. Ví dụ để chạy riêng MetaCLIP2 với 3.000 mẫu CrisisMMD, thêm `--max-samples 3000` vào lệnh standard tương ứng; không cần tạo job hay YAML mới. Có thể thêm `--dry-run` vào bất kỳ lệnh nào để kiểm tra gói trước khi gửi.

Sáu job CrisisMMD ảnh-only mặc định 1.000 mẫu, có thư mục kết quả `image_only_*` riêng. Xem [các lệnh đối chứng ảnh-only](crisismmd_image_only_comparison.md) trước khi so với WaffleCLIP/DCLIP upstream.

Mọi kết quả mới nằm ở `results/<phương_pháp>/<dataset>/<task>/samples_N/<model>/<task>_<id>/`; xem [cấu trúc kết quả](results_layout.md). Khi CLI mất kết nối **sau khi đã gửi** job, dùng `--resume-stage <stage>` theo đường dẫn launcher in ra thay vì gửi job mới. Chỉ file code thay đổi mới cần cập nhật dataset code; dataset đã xử lý được dùng lại khi SHA không đổi.
