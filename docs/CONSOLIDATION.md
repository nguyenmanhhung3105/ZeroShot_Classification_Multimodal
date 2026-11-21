# Lưu trữ trước khi hợp nhất

Snapshot tạo trước mọi thay đổi bản cuối:

- `backups/before_final_consolidation_20261103_032000/project_snapshot.tar.gz`
- SHA256: `4ba68e8e3ee857b65ac387179a034f3733aec42f9fb87cecbbb97d502d8359ba`
- Manifest, git status trước sửa và patch thay đổi có sẵn nằm cạnh snapshot.
- HEAD lúc chụp: `45fc00cac55a6f7fe8d84347ed4e4153cbcca198`.

Snapshot chứa mã nguồn, cấu hình, tài liệu, kết quả lịch sử và source nghiên cứu; không chứa dataset, venv, cache, weights, secrets, git objects hoặc backup lặp. Các dataset, môi trường, checkpoint và kết quả gốc ngoài vùng di chuyển được giữ nguyên, không xóa. Đây là backup phần dự án; không phải bản sao toàn bộ dữ liệu/weights.

Các thư mục ứng dụng cũ (`src`, `scripts`, `experiments`, `configs`, `tests`, `requirements`, `docs`) và `README.md`, `requirements.txt` được chuyển nguyên trạng sang `archive/legacy_20261103/`. Không sửa nội dung tài liệu/report đang viết. `archive/legacy_20261103/archive_manifest.json` lưu checksum file trước/sau di chuyển. `data/`, `third_party/`, `results/`, `results1/`, `venv/`, backup cũ và AGENTS.md không bị di chuyển. Những thay đổi/xóa dữ liệu đã có từ trước không được tự khôi phục hoặc ghi đè.

Để xem lại bản cũ, dùng archive. Các import/đường dẫn của chương trình cũ vẫn giả định cấu trúc gốc, nên đừng chạy trực tiếp từ archive trên dữ liệu hiện tại. Muốn khôi phục để chạy, tạo một thư mục dự án riêng, giải nén snapshot vào đó, trỏ tới dữ liệu cần thiết và thiết lập lại môi trường. Không giải nén đè lên bản cuối.

Trong Git, việc chuyển có thể hiện thành xóa ở đường dẫn cũ và thêm ở archive; kiểm tra rename khi review. Chưa commit/push. Kết quả bản cuối tách ở `results/final/`; các số lịch sử không bị thay bằng kết quả mới.

## Cập nhật 16/11/2026

Các mô tả `results/`, `results1/` ở trên phản ánh lần hợp nhất 03/11. Lần dọn tiếp theo đã chuyển kết quả lịch sử sang `archive/report_cleanup_20261116/`, giữ `results/final` và `results/_system`; đưa preprocessing Fakeddit trở lại src. Xem [báo cáo mới](REPORT_CLEANUP_20261116.md), [bản đồ code](CODEX-NAVIGATION-GUIDE.md) và [danh mục kết quả](../results/README.md). Snapshot 03/11 và toàn bộ archive cũ vẫn giữ nguyên.
