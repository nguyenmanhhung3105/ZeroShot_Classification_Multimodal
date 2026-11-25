# Kiểm tra demo localhost / Kaggle worker

## Bổ sung Docker Compose

- RED: test `test_container_binds_web_only_and_keeps_gateway_private` báo
  `TypeError: serve() got an unexpected keyword argument 'web_host'` trước thay đổi.
- GREEN: sáu test `test_demo_service.py` PASS sau khi thêm bind web riêng cho container;
  gateway vẫn bind loopback. Không thay đổi runner/model trên Kaggle.
- `docker compose config --quiet`, build web và `docker compose up -d`: PASS thực tế
  trên Docker Desktop Mac arm64. Build context 44,58 KB, không gửi dataset/weights/secret.
- Container web healthy; trang gốc HTTP 200; gateway thiếu xác thực HTTP 401;
  `importlib.util.find_spec('torch')` trong image trả None.
- Web chỉ publish `127.0.0.1:8765`; gateway không publish port. Tunnel dùng chung
  network namespace với web và đã đăng ký kết nối QUIC thành công.
- Đã xác nhận và dừng PID server Python cũ trước khi chuyển, tránh trùng cổng.
- `docker compose down` rồi `up -d`: PASS; hai container được tạo lại, web healthy.
  Đã dừng tunnel native cũ đúng cổng demo; không còn cần terminal Python/cloudflared riêng.
- Cần cập nhật URL/token trong Kaggle Secrets sau chuyển container. Không chạy lại
  GPU inference trong kiểm tra Docker này; notebook hiện có không cần đóng gói lại.

Tự đánh giá phần Docker: accuracy/completeness/clarity/actionability/conciseness đều 4/5
(trung bình 4,0): build và kiểm tra service thật đạt; còn bước người dùng cập nhật
Secrets và xác nhận lượt suy luận Kaggle qua URL mới. Không tuyên bố đã kiểm chứng lại GPU.

## Phạm vi và bằng chứng TDD

User journeys: upload ảnh và tiêu đề; chọn các phương pháp/backbone/pipeline; worker nhận
yêu cầu và trả kết quả từng phần; xem chi tiết/tải JSON; hủy yêu cầu; chuẩn bị notebook GPU
không kèm credentials hoặc ảnh; dùng head đã học mà không fit lại.

RED: chạy `venv/bin/python -B -m unittest discover -s tests -p 'test_demo*.py' -v`
trước khi triển khai, bảy test FAIL vì chưa có queue/validation và inference adapters.
GREEN: cùng nhóm test sau triển khai và mở rộng đạt **17/17 PASS**. Kiểm thử HTTP cần
quyền mở socket localhost; sandbox mặc định chặn bind nên lần kiểm API đầu báo PermissionError,
sau đó được chạy lại ngoài sandbox. Không tạo checkpoint commit vì quy tắc repo chỉ cho commit
khi người dùng yêu cầu; bằng chứng RED/GREEN được ghi ở đây thay thế.

| Bảo đảm | Bằng chứng | Kết quả |
|---|---|---|
| Ảnh hợp lệ, giới hạn đầu vào, chỉ chọn method/backbone đã khai báo | `tests/test_demo_service.py` | PASS |
| Một lease mỗi job, từ chối worker cũ, hủy loại kết quả muộn | `tests/test_demo_service.py` | PASS |
| HTTP từ web đến queue đến worker, xác thực và kiểm origin/host | `tests/test_demo_service.py` | PASS |
| Binary head và softmax nhiều lớp, đúng mapping Fake/Real khác Real/Fake | `tests/test_demo_inference.py` | PASS |
| Cascade dùng hard gate; không thay nhãn bằng argmax joint | `tests/test_demo_inference.py` | PASS |
| Một lần encode ảnh được dùng chung, đổi tiêu đề dùng lại feature ảnh | `tests/test_demo_inference.py` | PASS |
| Thiếu asset không giả lập dự đoán; checksum sai bị từ chối | `tests/test_demo_inference.py` | PASS |
| StatA/EM/hierarchy chạy trên nhóm feature tổng hợp, không singleton fallback | `tests/test_demo_inference.py` | PASS |
| Head mọi policy/seed được giữ trong kết quả chi tiết | `tests/test_demo_inference.py` | PASS |
| Bundle private/GPU chỉ đóng gói source/config cho phép, không ghi đè bundle cũ | `tests/test_demo_package.py` | PASS |
| Export chỉ nhận completed full dev; nhóm tham chiếu không chứa dev feature/nhãn | `tests/test_demo_package.py` | PASS |
| Worker yêu cầu HTTPS, Bearer header, không fallback CPU | `tests/test_demo_worker.py` | PASS |

Coverage đo bằng coverage.py trong môi trường QA tạm: `demo_service.py` **88%**,
`demo_inference.py` **81%**, tổng hai module **83%**. Không diễn giải số này là coverage
GPU thực tế hoặc coverage toàn repo. GPU runtime/model download/CALIP extractor chưa được
chạy trên Kaggle trong lượt kiểm tổng hợp.

`node --check src/vlm_project/demo_web/app.js`: PASS. Python AST parse cho chín file mới:
PASS. `git diff --check`: PASS. Không thay đổi runner/config thí nghiệm hiện có.

Kaggle CLI 2.2.5 đã trả được danh sách notebook của tài khoản `nguyenhung3105`.
Đã xuất head binary/cascade từ hai completed dev run được ghi trong `LOCAL_DEMO.md`;
không đọc feature cache thực tế, không tạo nhóm tham chiếu thật, không nạp backbone local.
Server local đã khởi động trên cổng 8765/8766. HTTPS gateway qua Cloudflare đã trả
**401** cho yêu cầu không xác thực, đúng ranh giới worker. Kiểm tra import service xác nhận
`torch`, `open_clip`, `transformers` không được nạp vào process local.
Kiểm thử Chrome và thao tác tải notebook private ở chế độ `--no-run` đã được yêu cầu,
nhưng chưa có kết quả cuối tại thời điểm ghi báo cáo; không tính hai bước này là PASS.

## Giới hạn còn lại

Tốc độ và bộ nhớ GPU thật cần đo sau khi notebook được cấp Secrets và khởi chạy.
Một backbone nằm trong GPU tại một thời điểm; so sánh cả hai có thể phải nạp lại.
Nhánh OpenAI là adapter demo OpenCLIP; head few-shot hiện có chỉ dành cho MetaCLIP2.
StatA/EM cần nhóm tham chiếu được cấp riêng. Độ chính xác của ảnh tùy ý ngoài phân phối
Fakeddit chưa được đánh giá; điểm mô hình không phải xác suất xác minh sự thật.

## Tự đánh giá

Áp dụng skill `agent-self-evaluation` sau triển khai, dựa trên kiểm thử và giới hạn trên.

| Tiêu chí | Điểm / 5 | Bằng chứng / cải thiện |
|---|---|---|
| Accuracy | 4 | Mapping lớp, head và hard gate có test; chưa xác nhận GPU runtime thật |
| Completeness | 4 | Có UI/API/worker/bundle/head; cần người dùng cấp Secrets và nhóm tham chiếu |
| Clarity | 4 | UI phân biệt chưa chạy, điểm và nhãn; bảng nhiều phương pháp cần cuộn ngang trên mobile |
| Actionability | 4 | Có CLI, notebook và hướng dẫn; lần đầu cần thao tác trên Kaggle Secrets |
| Conciseness | 4 | Tách hướng dẫn vận hành khỏi báo cáo; tài liệu chi tiết dài do nhiều chế độ |

Trung bình: **4,0/5**. Ưu tiên tiếp theo: xác nhận kết nối và đo một lượt GPU thật;
sau đó cấp nhóm tham chiếu nếu muốn demo các nhánh thích nghi. Người dùng có cơ sở
đánh giá mức này vì các giới hạn chưa kiểm chứng được nêu tường minh.
