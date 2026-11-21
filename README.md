# Phân loại nội dung với MetaCLIP2

Mã nguồn phục vụ các thí nghiệm và báo cáo `report/main.tex`: benchmark CLIP/DCLIP/WaffleCLIP, kết hợp Waffle–semantic–StatA, Fakeddit Task 1 zero-shot/few-shot min5 và đối chứng phân cấp trên Fakeddit/CIFAR-100. Backbone chính là MetaCLIP2 ViT-H/14; nhánh OpenAI ViT-L/14 dùng kiểm chứng tham chiếu. Few-shot fit linear head trên feature của backbone đóng băng.

Đọc [bản đồ code](docs/CODEX-NAVIGATION-GUIDE.md), [nguồn dữ liệu Fakeddit](src/preprocessing/data_fakeddit/README.md) và [kiểm tra/dọn dự án](docs/REPORT_CLEANUP_20261116.md).

## Cài đặt và cập nhật cho thành viên

Chạy tại thư mục gốc với Python 3.12 (phiên bản đã kiểm thử local):

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -B run.py --help
venv/bin/python -B -m unittest discover -s tests -q
```

Xem [hướng dẫn bàn giao GitHub](docs/GITHUB_HANDOFF.md) để cập nhật checkout cũ, cấu hình tài khoản Kaggle và lấy dữ liệu. GitHub chứa source, báo cáo, cấu hình, test và [summary đã xuất](results/README.md); dataset, weights, archive, backup và artifact đầy đủ được giữ ngoài Git. Máy mới chạy được test tổng hợp và prepare-only mà không cần các thư mục đó. Cần Internet và dung lượng cho thư viện khi cài đặt; chạy inference thật cần dữ liệu và model riêng.

## Cấu trúc đang dùng

```text
run.py                         CLI local
src/vlm_project/                backbone, prompt, adaptation, runner, metric, Kaggle
src/preprocessing/data_fakeddit/ tạo và kiểm tra train/dev/test, giữ nguồn và dòng gốc
scripts/                       entry point, chuẩn bị dữ liệu/job, kiểm tra kết quả
configs/                       protocol/prompt/descriptor và cấu hình từng thí nghiệm
results/final/                 artifact hiện tại, tách full / pilot / smoke
results/published/             summary nhẹ và manifest checksum chia sẻ qua GitHub
results/_system/               receipt, staging, sample lock đang có
report/                        main.tex, sáu chương, tài liệu tham khảo và hình
archive/                       code/kết quả lịch sử, có manifest khôi phục
backups/                       snapshot trước khi thay đổi
```

Không thay đổi các tập dữ liệu, checkpoint, seed, prompt hoặc công thức thí nghiệm trong lần dọn ngày 16/11/2026. Các module phụ trợ vẫn được giữ khi runner của báo cáo gọi tới chúng. Kết quả giữa kỳ và nhánh cũ nằm tại `archive/report_cleanup_20261116/results/`; `results1` nằm cạnh đó. Các lần dọn trước ở `archive/legacy_20261103/`.

## Thí nghiệm dùng cho báo cáo

| Nội dung | Code chính | Tài liệu |
|---|---|---|
| Prompt benchmark / tham chiếu paper | `benchmark.py`, `paper_reference.py` | [tiến độ](docs/BENCHMARK_PROGRESS.md), [tham chiếu](docs/PAPER_RESULTS.md) |
| Waffle–semantic–StatA, DTD/EuroSAT | `waffle_stata.py`, `waffle_semantic_stata.py` | [giao thức](docs/WAFFLE_SEMANTIC_STATA.md), [EuroSAT](docs/WAFFLE_SEMANTIC_STATA_EUROSAT_REVIEW.md) |
| Fakeddit Task 1 zero-shot min5 | `fakeddit_task1.py` | [protocol](docs/FAKEDDIT_TASK1.md), [kết quả](docs/FAKEDDIT_TASK1_MIN5_REVIEW.md) |
| Few-shot Task 1 và ngân sách support | `fakeddit_fewshot_task1.py`, `fakeddit_fewshot_support.py` | [protocol](docs/FAKEDDIT_FEWSHOT_TASK1_MIN5.md), [kết quả](docs/FAKEDDIT_FEWSHOT_SUPPORT_MIN5_REVIEW.md) |
| Few-shot hai tầng min5 / flat 6way | `fakeddit_fewshot_cascade.py` | [runner và job dev](docs/FAKEDDIT_FEWSHOT_CASCADE_MIN5.md), [full dev đã audit](docs/FAKEDDIT_FEWSHOT_CASCADE_MIN5_REVIEW.md) |
| Protocol/test hai tầng min5 | `cascade_protocol.py`, `cascade_test_runner.py` | [protocol đã niêm phong và runtime đích](docs/CASCADE_TEST_PROTOCOL.md); chưa mở test |
| Hierarchy, CALIP, CIFAR-100 | `fakeddit_hierarchy.py`, `fakeddit_calip_ablation.py`, `cifar100_*.py` | [Fakeddit](docs/FAKEDDIT_HIERARCHY_DEV_REVIEW.md), [CIFAR](docs/CIFAR100_EM_STATA_DEV_REVIEW.md) |

Min5 và cohort v2 cũ là hai tập khác nhau. Bảng min5 hiện dùng **dev**, đã có kết quả cascade few-shot cùng cohort; chưa xác nhận phương án trên test min5. Pilot/smoke chỉ phục vụ phát triển hoặc vận hành. Không lấy chúng thay số full, không chọn seed tốt nhất; độ lệch chuẩn giữa seed không phải khoảng tin cậy.

## Kiểm tra local

```sh
venv/bin/python -B run.py --help
venv/bin/python -B run.py benchmark --dataset dtd --prepare-only
venv/bin/python -B scripts/prepare_fakeddit_splits.py --help
venv/bin/python -B -m unittest discover -s tests -q
venv/bin/python -B scripts/audit_report_results.py
```

Help, prepare-only và unittest không train hoặc nạp model/dataset thật. Lệnh audit cuối chỉ chạy khi đã lấy artifact đầy đủ trong `results/final/` từ máy lưu kết quả; summary GitHub không đủ để tính lại prediction. Audit xác minh checksum và tính lại accuracy/Macro-F1/BA từ prediction, kiểm tra mean/sample std. Top-5 chỉ kiểm tra phép tổng hợp. Xem [audit bàn giao](docs/results_audit_20261121.json), [audit lịch sử](docs/results_audit_20261116.json) và [danh mục run](results/README.md).

## Chuẩn bị và chạy Kaggle

Launcher vẫn là `scripts/launch_runs.py`. Dùng `--prepare-only` để tạo bundle/notebook offline. Ví dụ job few-shot support hiện tại:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_fewshot_min5.yaml \
  --job fakeddit_fewshot_support_min5 --max-samples 5 --prepare-only
```

Khi chủ động gửi job, thay `--prepare-only` bằng `--use-kaggle true --wait`; bỏ `--max-samples 5` để chạy đủ cohort theo cấu hình. Giữ nguyên dataset ảnh đã có trên Kaggle. Launcher tải kết quả vào `results/final/kaggle/`, lưu staging/receipt tại `results/_system/launches/final/`. Nếu gián đoạn sau khi gửi, dùng `--resume-stage <stage>` để theo dõi đúng job cũ. Không gửi lại chỉ vì lỗi đọc trạng thái.

Trước khi gửi job từ tài khoản khác, tạo bản sao launcher với `kaggle.username` của mình và cấu hình quyền đọc các dataset trong `configs/project*.yaml`. Kaggle CLI cài riêng theo [hướng dẫn bàn giao](docs/GITHUB_HANDOFF.md). Benchmark dùng prompt tác giả cần chạy `venv/bin/python -B run.py sources` một lần để lấy đúng commit; lệnh này cần mạng, không nằm trong kiểm thử offline.

Benchmark dùng `configs/kaggle_launcher.yaml`; zero-shot min5 dùng `configs/kaggle_launcher_min5.yaml`; few-shot dùng `configs/kaggle_launcher_fewshot_min5.yaml`. Lệnh riêng nằm trong tài liệu từng thí nghiệm ở bảng trên. `run.py package` xuất bundle code và notebook vào `bundles/final/`, gồm cả preprocessing; không kèm dữ liệu/model/secret.

CLI `run.py fakeddit`, `freeze`, `report` vẫn giữ để tái hiện pipeline v2 cũ theo [PROTOCOL.md](docs/PROTOCOL.md). Lock đó không phải protocol cho min5/few-shot. Dọn code làm đổi code hash; không dùng lock của code cũ để xác nhận test của bản mới.

Few-shot min5 có [protocol và runner test riêng](docs/MIN5_TASK1_PROTOCOL.md): `scripts/prepare_min5_protocol.py` tạo draft/niêm phong, `scripts/run_min5_test.py --prepare-only` chỉ kiểm kế hoạch. Thực thi yêu cầu protocol sealed và đường dẫn train/test tường minh; không dùng slug test v2 trong cấu hình chung. Hiện mới kiểm bằng dữ liệu tổng hợp, chưa niêm phong hoặc chạy test min5 thật, chưa tích hợp launcher Kaggle.

## Backup và nguồn dữ liệu

Snapshot trước lần dọn này: `backups/before_report_cleanup_20261116/source_before.tar.gz`, có manifest SHA256. Archive lịch sử: `archive/report_cleanup_20261116/manifest.json`. Cách khôi phục ở [báo cáo dọn dự án](docs/REPORT_CLEANUP_20261116.md). Bản báo cáo đã chuyển vào `report/`; bản trước khi biên tập được lưu tại `backups/report_revision_20261116/`.

Fakeddit giữ mapping nguồn chính thức: train → `multimodal_train.tsv`, dev → `multimodal_validate.tsv`, test → `multimodal_test_public.tsv`. Code giữ `source_file`, `source_row`, `source_split`, checksum và cohort ID. Lần rà soát này không truy cập dữ liệu thô, không dựng lại split, không chạy inference hoặc gửi job.
