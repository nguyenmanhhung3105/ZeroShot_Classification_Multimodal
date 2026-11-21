# Rà soát và làm gọn dự án — 16/11/2026

Phạm vi giữ lại dựa trên `main.tex` hiện tại: kiểm chứng prompt benchmark, Waffle–semantic–StatA trên DTD/EuroSAT, Task 1 Fakeddit zero-shot/few-shot min5, cùng các đối chứng hierarchy/CALIP/CIFAR. Không thay công thức, prompt, seed, dataset, nhãn hoặc số liệu báo cáo.

## Backup và phần được chuyển

- Snapshot source/config/script/test/doc/notebook/report trước khi sửa: `backups/before_report_cleanup_20261116/source_before.tar.gz`. `manifest.json` cạnh snapshot ghi checksum từng file và checksum TAR; đã đọc lại TAR để xác minh đủ **161 file**. Có git status và patch các thay đổi đã tồn tại từ trước.
- Archive: `archive/report_cleanup_20261116/manifest.json`, **2.355 file**, **644.542.761 byte** (~615 MiB), checksum trước/sau khớp. Phần lớn là kết quả lịch sử; không sao chép lặp lại 64 GiB staging/cache của `_system`.
- Chuyển các thư mục kết quả cũ khỏi `results/`, giữ `final/` và `_system/`. Chuyển `results1/`, `experiments/fakeddit_fewshot/`, kết quả `results/final/single_stage/` và hai tài liệu SINGLE_STAGE cũ vào archive. Bundle cũ `bundles/final/` được chuyển vào archive trước khi tạo bundle mới. Bản nguyên gốc `single_stage.py` cũng được sao lưu trước khi bỏ replay độc lập.
- Giữ nguyên toàn bộ run còn lại trong `results/final/`, receipt job, `main.tex`, PDF và hình báo cáo. Không ghi đè các thay đổi của người dùng từ trước; không commit/push/gửi Kaggle job.

## Code và truy vết dữ liệu

Khôi phục builder Fakeddit, wrapper preprocessing/verify vào `src/preprocessing/data_fakeddit/`. Năm helper được trích nguyên thân từ code cohort cũ, bỏ import thừa của hai dataset khác. Cấu hình nền legacy được sao vào `configs/preprocessing/` để builder không cần nạp code cấu hình từ archive. `scripts/prepare_fakeddit_splits.py` dùng builder này và vẫn hợp exclusions của dev lịch sử, kể cả khi kết quả cũ chuyển vào archive.

Giữ train/dev/test tách nguồn chính thức; TSV/manifest vẫn ghi file nguồn, dòng gốc, split, checksum và cohort ID. Xem [README preprocessing](../src/preprocessing/data_fakeddit/README.md). Downloader đời đầu giữ nguyên trong archive trước đây; builder hiện có downloader bounded riêng. Không dựng lại dataset, không thay dữ liệu đã có.

Bỏ lệnh CLI `single-stage-replay` và hàm replay đã kết thúc. Giữ `single_stage.candidates` vì diagnostic còn gọi tới phép tính này. Các module khác được giữ khi cần tái hiện bảng trong báo cáo hoặc là phụ thuộc thật của runner. Không gom lại solver hay đổi cách quyết định khi dọn thư mục.

Thí nghiệm few-shot cũ chuyển vào archive; regression test sampler/linear head dùng bản trích nguyên thân trong `tests/fixtures/legacy_fewshot.py` cùng nguồn và hash. Test hiện tại không cần import code archive. Bundle mới kèm preprocessing, cấu hình nền và entry point kiểm tra.

## Kiểm tra kết quả

`scripts/audit_report_results.py` chỉ đọc artifact của `results/final/`, bỏ `_system`. Audit xác minh completion/hash đầu ra, stream prediction để kiểm tra ID duy nhất/cùng cohort/nhãn, tính lại accuracy, Macro-F1, balanced accuracy từng method/seed; kiểm tra mean và sample std với bảng seed, summary TSV và run manifest. Top-5 chỉ đối chiếu phép tổng hợp, không tái dựng thứ hạng từ feature/logits.

Kết quả **32/32 run passed**, **329 artifact**, **598 tổ hợp method–seed**, **4.803.546 dòng dự đoán**. Chi tiết ở [results_audit_20261116.json](results_audit_20261116.json); danh mục run ở [results/README.md](../results/README.md). Smoke, pilot và diagnostic vẫn được ghi riêng trong danh mục, không được coi là full evaluation. Kết quả lịch sử trong archive được kiểm tra bảo toàn byte/checksum khi chuyển; không tuyên bố đã tính lại mọi metric lịch sử.

Số full chính trong `main.tex` được đối chiếu với summary: Waffle–semantic–StatA DTD ~71,10% accuracy; Waffle–semantic EuroSAT ~64,42%; StatA ảnh min5 ~72,02%; few-shot ảnh–tiêu đề support 192/192 ~82,41%. Đây là kết quả phát triển theo phạm vi cohort đã ghi, chưa xác nhận test min5 hay tầng hai few-shot.

## Khôi phục

Để xem code trước dọn, giải nén `source_before.tar.gz` vào **thư mục mới** rồi đối chiếu manifest. Không giải nén đè lên working tree. Để lấy một kết quả lịch sử, tra `source` → `archive` trong manifest archive; có thể đọc trực tiếp tại vị trí archive. Muốn tái chạy code lịch sử, dùng snapshot trong workspace riêng và cấu hình lại đường dẫn dữ liệu; archive không được xem là một checkout chạy độc lập.

Snapshot không chứa dataset, weights, venv, secrets, Git objects, archive/backup lặp; kết quả lịch sử được bảo toàn bằng các lần chuyển có checksum. Đây là backup trong cùng máy, không phải bản dự phòng ngoài máy. Dọn source làm đổi code hash: lock cũ không hợp lệ cho code mới; không tạo lại lock hoặc mở test trong lần dọn này.

## Xác minh mã nguồn và bundle

150 test chạy ở workspace, không có lỗi, 1 skip theo điều kiện môi trường. Sáu test split Fakeddit dùng dữ liệu tổng hợp; ba test audit kiểm tra metric sai dù checksum hợp lệ, prediction bị sửa và export deterministic thiếu cột seed. Thân năm helper Fakeddit được so AST với bản legacy và khớp nguyên vẹn. Cấu hình nền được sao nguyên byte. Syntax, liên kết tài liệu bàn giao và diff whitespace đều hợp lệ.

Kiểm tra bundle độc lập phát hiện test Waffle còn hash descriptor từ checkout tác giả và test tham chiếu còn đọc CSV third_party; đã đổi test sang descriptor tổng hợp trong thư mục tạm và CSV snapshot có checksum đã đóng gói, giữ nguyên assertion công thức/nguồn. Thuật toán suy luận không đổi. Bundle kèm toàn bộ script cần thiết, không cần archive để chạy regression tests.
