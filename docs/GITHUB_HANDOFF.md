# Bàn giao GitHub — 21/11/2026

Đây là bản source sau hợp nhất sang `src/vlm_project/`, dùng `run.py` và launcher trong `scripts/`. Các đường dẫn `src/run_experiment.py`, `src/run/`, `configs/launcher.yaml` đời trước đã ngừng dùng. Code lịch sử vẫn tra được qua lịch sử Git; archive và backup trên máy tác giả không nằm trong bản chia sẻ.

## Thành viên cập nhật

Nếu checkout cũ có sửa đổi, commit vào nhánh riêng hoặc lưu bản sao trước khi cập nhật. Kiểm tra bằng `git status`; không reset hoặc ghi đè thay đổi local. Khi working tree sạch:

```sh
git switch main
git pull --ff-only origin main
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -B run.py --help
venv/bin/python -B run.py benchmark --dataset dtd --prepare-only
venv/bin/python -B scripts/prepare_fakeddit_splits.py --help
venv/bin/python -B -m unittest discover -s tests -q
```

Máy mới clone repo trước rồi chạy từ thư mục gốc. Python local đã kiểm là 3.12.0; môi trường này phục vụ phát triển, không thay thế runtime Kaggle đã niêm phong cho test cascade. Dependency hiện dùng khoảng phiên bản; không coi cài mới là tái lập chính xác môi trường inference lịch sử.

## Dữ liệu, kết quả và tài khoản

- Dùng lại cohort đã chuẩn bị đúng manifest/hash trong `configs/project*.yaml` và recipe của từng thí nghiệm. Dataset, trọng số, feature, prediction và archive được chuyển riêng qua kho lưu trữ của nhóm hoặc Kaggle, không đưa vào Git. Không dựng lại split chỉ vì máy mới chưa có dữ liệu: builder còn hợp exclusions từ lịch sử local, mà archive đó không có trong bản clone.
- GitHub có [summary nguyên byte và manifest checksum](../results/published/manifest.json), [danh mục kết quả](../results/README.md), [audit bàn giao](results_audit_20261121.json), source và hình báo cáo trong `report/`. Summary không thay thế artifact đầy đủ khi tính lại metric.
- `scripts/audit_report_results.py` cần cây `results/final/` đầy đủ. Không chạy audit trên bản clone chỉ có summary rồi ghi đè bản audit đã xuất.
- Benchmark/prompt tham chiếu cần `venv/bin/python -B run.py sources` để clone ba repo tác giả đúng commit. Lệnh cần mạng, không tải dataset hoặc weights. Trọng số được tải khi chủ động chạy backbone thật.
- Launcher Kaggle cần CLI riêng: `venv/bin/python -m pip install 'kaggle==2.2.5'` (phiên bản có trên máy rà soát). Xác thực tài khoản của mình ngoài repo, kiểm tra bằng `venv/bin/kaggle --help`; không commit credential. Tạo bản sao launcher trong máy, sửa `kaggle.username` trước khi gửi job. Các dataset thuộc tài khoản tác giả cần được cấp quyền đọc hoặc gắn đúng Input.
- Chỉ dùng `--prepare-only` để kiểm tra offline. Cấu hình launcher hiện bật `use_kaggle: true`; chạy launcher không có cờ này có thể gửi job thật. Không sửa prompt, cohort, seed hoặc recipe đã khóa để xử lý vấn đề tài khoản/đường dẫn.

Protocol cascade tại `configs/cascade_min5_test_lock.json` đã sealed, chưa mở test. Đã kiểm checksum/fingerprint mà không đọc dữ liệu test. Local trả `runtime_matches=false`, `executable=false` đúng dự kiến; dùng [quy trình protocol](CASCADE_TEST_PROTOCOL.md) trong môi trường đích. Không tạo lại lock để bỏ qua lỗi runtime.

## Phạm vi đưa lên GitHub

Giữ `.gitignore`, `AGENTS.md`, `README.md`, `requirements.txt`, `run.py`, `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, `notebooks/`, `report/`, `results/README.md`, `results/published/`. Dataset, backup, archive, staging, bundle và các file ZIP xuất từ Overleaf giữ local. Logo dùng trong báo cáo đã có dưới `report/images/`.

Dữ liệu và backup đã được theo dõi ở phiên bản cũ sẽ được bỏ khỏi index bằng `--cached`, giữ nguyên file local. Thành viên cần sao lưu dữ liệu trên checkout cũ trước khi pull vì Git áp dụng các deletion của commit. Lịch sử Git trước đó vẫn được giữ nguyên.

## Xác minh bản bàn giao

- Workspace: 192 tests, 0 lỗi, 1 skip theo môi trường; dùng dữ liệu/encoder tổng hợp.
- Bản source giải nén riêng, không có Git/dataset/archive/third_party: cũng 192 tests, 0 lỗi, 1 skip. CLI help, benchmark prepare-only, preprocessing help và cascade test prepare-only đều chạy thành công bằng venv hiện có; chưa xác minh cài dependency mới từ đầu.
- Audit kết quả thật đã lưu: 36/36 run passed, 391 artifact, 698 tổ hợp method–seed, 4.953.696 prediction; không chạy inference, train hoặc đọc dữ liệu thô. Audit xác minh checksum và metric theo [phạm vi script](../scripts/audit_report_results.py); không phải replay mọi posterior/feature.
- 36 bảng summary được sao nguyên byte, ghi checksum và nguồn trong manifest; các artifact gốc không bị sửa.
- Lock cascade vẫn khớp code/config; chỉ sửa tài liệu và quy tắc chia sẻ, không sửa `src/vlm_project/`, `run.py`, requirements hoặc công thức thí nghiệm trong lượt bàn giao này.
- Chưa chạy linter/type checker vì các công cụ đó chưa được cài trong venv; chưa cài mới môi trường từ mạng, chưa kiểm GPU/test thật hoặc biên dịch LaTeX. Không coi các mục này là đã PASS.

Tự rà soát theo skill `agent-self-evaluation`:

| Tiêu chí | Điểm | Bằng chứng và phần cần cải thiện |
|---|---:|---|
| Chính xác | 4/5 | Test/audit/lock có bằng chứng; chưa kiểm GPU thật. |
| Đầy đủ | 4/5 | Có source, summary, hướng dẫn cập nhật; artifact nặng cần chuyển riêng. |
| Rõ ràng | 4/5 | Nêu phạm vi GitHub và runtime sealed; tài liệu lịch sử vẫn có liên kết archive local. |
| Khả thi | 4/5 | Test offline đã chạy; inference cần Input/quyền đọc và môi trường đích. |
| Súc tích | 4/5 | Giữ luồng xử lý, tập trung bàn giao; phần hướng dẫn chủ repo dài vì có dữ liệu đã track từ trước. |

Trung bình **4,0/5**. Ưu tiên tiếp: xác minh môi trường Kaggle khi nhóm chủ động chạy test. Người dùng có thể đối chiếu đánh giá qua test, manifest và audit; đây không phải chứng nhận kết quả trên test độc lập.
