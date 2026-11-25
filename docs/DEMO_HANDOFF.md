# Bàn giao giao diện demo phân loại nội dung ảnh–tiêu đề

Tài liệu gửi cho thành viên dựng, chỉnh sửa và vận hành giao diện. Chạy các lệnh dưới đây tại thư mục gốc repository.

## 1. Cài đặt từ GitHub

Cần Git và Docker Desktop (đã mở). Không cần GPU local, Node.js, React hoặc cài thư viện model để xem giao diện.

```sh
git clone https://github.com/nguyenmanhhung3105/ZeroShot_Classification_Multimodal.git
cd ZeroShot_Classification_Multimodal
docker compose up -d --build web
```

Mở http://127.0.0.1:8765/. Lệnh trên chỉ chạy web để xem, sửa giao diện và xem kết quả thực nghiệm đã lưu. Chưa chạy suy luận khi chưa kết nối worker Kaggle.

```sh
docker compose ps                       # Trạng thái
docker compose logs --tail=50 web       # Log web
docker compose stop                    # Dừng container
docker compose up -d web                # Bật lại bằng image đã build
docker compose down                    # Dừng và gỡ container
```

Sau khi sửa HTML/CSS/JS, build lại cả hai dịch vụ nếu đang dùng tunnel:

```sh
docker compose up -d --build
```

Sau đó nhấn Cmd+Shift+R trên macOS hoặc Ctrl+Shift+R trên Windows/Linux. Tunnel chia sẻ network namespace của web; khi thay container web nên cập nhật cả stack. Khởi động lại web làm mất lịch sử trong RAM và đổi token; tunnel khởi động lại có thể đổi URL.

## 2. Thành phần hệ thống

```text
Trình duyệt localhost:8765
          |
          v
Python demo_service: giao diện, API, hàng đợi trong RAM
          ^
          | gateway :8766 qua HTTPS Cloudflare Tunnel
          |
Kaggle worker: nhận job → nạp backbone trên CUDA → trả kết quả
```

- Local: phục vụ trang, kiểm tra upload, giữ trạng thái và hiển thị kết quả; không nạp backbone.
- Kaggle: chạy encoder và các phương pháp phân loại. CLI hỗ trợ chuẩn bị/gửi notebook; CLI không biến GPU Kaggle thành GPU local.
- Thực nghiệm đã lưu: đọc snapshot JSON, không cần notebook hay GPU.
- Không mở `index.html` trực tiếp: giao diện dùng các API do server cung cấp.

## 3. Bản đồ mã nguồn

| File | Trách nhiệm |
|---|---|
| `src/vlm_project/demo_web/index.html` | Khung trang, nhãn, form, bảng, dialog và ID phần tử |
| `src/vlm_project/demo_web/style.css` | Màu, typography, bố cục, trạng thái, responsive |
| `src/vlm_project/demo_web/app.js` | Điều hướng, upload, chọn phương pháp, gọi API, polling và render kết quả |
| `src/vlm_project/demo_web/analytics.js` | Biểu đồ SVG và bảng thống kê từ snapshot |
| `src/vlm_project/demo_web/experiment-results.json` | Danh mục kết quả thực nghiệm |
| `src/vlm_project/demo_web/experiments/*.json` | Snapshot thống kê từng run; không sửa tay để thay số liệu |
| `src/vlm_project/demo_service.py` | HTTP server, catalog phương pháp, validation, queue và worker gateway |
| `src/vlm_project/demo_worker.py` | Kết nối outbound từ Kaggle, heartbeat, nhận/trả job |
| `src/vlm_project/demo_inference.py` | Adapter backbone, prompt, few-shot head và suy luận |
| `scripts/demo.py` | CLI chạy web, chuẩn bị notebook, xuất assets |
| `scripts/export_demo_results.py` | Xuất snapshot từ artifact đã kiểm tra; không cần chạy khi chỉ dựng UI |
| `docker/demo/Dockerfile` | Image web nhẹ, không chứa model runtime |
| `compose.yaml` | Web và tunnel, cổng, healthcheck, giới hạn tài nguyên |
| `.dockerignore` | Chỉ cho phép mã web cần thiết vào build context |

Frontend là HTML/CSS/JavaScript thuần, không có bước npm build. Docker đóng gói server và tài nguyên tĩnh.

## 4. Các trang và element

ID là liên kết giữa HTML và JavaScript; đổi ID phải sửa cả nơi gọi tương ứng.

| Trang / ID | Element và chức năng |
|---|---|
| `view-overview` | Bài toán, đầu vào, taxonomy và cơ chế phương pháp |
| `view-setup` | Form cấu hình cho một yêu cầu suy luận |
| `view-results` | Kết quả của job đang mở, không phải đánh giá toàn dataset |
| `view-experiments` | Kết quả đánh giá đã lưu, có nguồn và phạm vi |
| `view-history` | Danh sách job của phiên server |
| `classify-form` | Form tạo yêu cầu |
| `image`, `upload-zone` | Chọn hoặc kéo thả ảnh JPEG/PNG/WebP, tối đa 8 MB |
| `preview`, `file-name`, `image-change` | Xem ảnh, tên file và thay ảnh |
| `title` | Tiêu đề gốc; form/API hiện yêu cầu 1–2.000 ký tự kể cả khi chỉ chọn phương pháp ảnh |
| `task` | Nhị phân Real/Fake hoặc sáu lớp Fakeddit |
| `pipeline` | Một tầng, hai tầng hoặc cả hai; các phương pháp được lọc theo task |
| `backbones` | Lựa chọn backbone; head few-shot phải khớp backbone |
| `methods` | Danh sách checkbox sinh từ catalog API |
| `method-count`, `selection-count`, `backbone-count` | Số cấu hình đang chọn |
| `select-all`, `select-none` | Chọn/bỏ các phương pháp phù hợp đang hiển thị |
| `run`, `form-error` | Gửi job và lỗi kiểm tra form |
| `job-status`, `progress`, `job-message`, `cancel` | Trạng thái, tiến độ và hủy yêu cầu |
| `results`, `table-wrap` | Bảng phương pháp, dự đoán, điểm và thời gian |
| `details`, `details-body` | Dialog chi tiết lớp, cấu hình hoặc lý do thiếu tài nguyên |
| `download` | Tải JSON kết quả suy luận |
| `exp-run`, `exp-metric`, `exp-method` | Chọn run, chỉ số và phương pháp phân tích |
| `exp-tabs`, `exp-chart`, `exp-data-table` | Chọn loại biểu đồ, SVG và bảng số liệu |
| `exp-source`, `exp-meta` | Nguồn, model, split và phạm vi đối chiếu |
| `exp-download-json/csv/svg` | Các nút xuất dữ liệu và biểu đồ (ID đầy đủ có tiền tố `exp-download-`) |
| `history` | Các job có thể mở lại |
| `connection`, `connection-open` | Dialog thiết lập kết nối Kaggle |
| `gpu-state`, `gpu-detail` | Trạng thái heartbeat worker |
| `show-token`, `connection-token`, `copy-token` | Lấy và sao chép mã phiên; không gửi token trong tài liệu/Git |

Điều hướng dùng hash, ví dụ `/#view-setup` và `/#view-experiments`; mỗi lần chỉ hiển thị một vùng. Chuyển trang không tự gửi lại job.

## 5. Cấu hình trình bày đã chốt

- Đầu vào chính: **chỉ ảnh** và **ảnh + tiêu đề**. Cấu hình chỉ tiêu đề không hiển thị ở phần chính; nghiên cứu gốc vẫn giữ nguyên.
- Zero-shot chính: CLIP ảnh, CLIP fusion (ảnh + tiêu đề), DCLIP và WaffleCLIP.
- Thích nghi chính: StatA ảnh, cần nhóm tham chiếu. Không dùng kết quả một ảnh baseline để thay cho thích nghi theo nhóm.
- Few-shot nhị phân: ảnh và ảnh + tiêu đề; encoder đóng băng, dùng head đã học.
- Few-shot sáu lớp: Flat/Cascade ảnh + tiêu đề theo tài nguyên đã có. Chưa có đối chứng few-shot chỉ ảnh tương ứng cho phần này.
- Majority, EM-Dirichlet, CLIP–StatA, legacy, unit, CALIP và các biến thể liên quan nằm trong mục đối chứng bổ sung thu gọn.

Các điểm sửa chính trong `app.js`: `familyOf()` phân nhóm; `applicable()` lọc theo task/pipeline/modality; `selectedMethods` là lựa chọn mặc định; `renderMethods()` tạo checkbox. `METHODS` trong `demo_service.py` là catalog tên/modality/pipeline; không đổi mã phương pháp chỉ để đổi chữ hiển thị.

## 6. Luồng API

| API | Công dụng |
|---|---|
| `GET /api/catalog` | Backbone và phương pháp có thể chọn |
| `GET /api/health` | Trạng thái local/worker |
| `POST /api/jobs` | Gửi ảnh base64, tiêu đề, task, pipeline, methods, backbones |
| `GET /api/jobs` | Lịch sử phiên |
| `GET /api/jobs/{id}` | Kết quả/progress của job |
| `POST /api/jobs/{id}/cancel` | Hủy yêu cầu |
| `POST /api/connection` | Lấy token phiên từ UI local |
| `GET /api/experiments` | Danh mục snapshot |
| `GET /api/experiments/{id}` | Dữ liệu thống kê của run |

Gateway worker tách ở cổng 8766: `POST /worker/heartbeat`, `/worker/claim`, `/worker/update`, dùng Bearer token. Không tunnel cổng giao diện 8765. Chi tiết schema/validation nằm trong `demo_service.py`.

## 7. Chạy suy luận bằng Kaggle

1. Chạy `docker compose up -d --build` và `docker compose logs -f tunnel` để lấy HTTPS URL.
2. Mở web → Thiết lập kết nối → Lấy mã kết nối phiên này.
3. Trên Kaggle Notebook, bật GPU/Internet; thêm và cho phép hai Secrets `DEMO_BROKER_URL` và `DEMO_WORKER_TOKEN`.
4. Tạo notebook bằng `scripts/demo.py prepare`; hướng dẫn đầy đủ trong `docs/LOCAL_DEMO.md`. Không dùng owner tài khoản của người khác; thay owner bằng tài khoản người chạy.
5. Few-shot cần assets head đã xuất đúng backbone. Repo giao diện không tự cung cấp trọng số/head riêng; thiếu assets phải lấy từ người quản lý thực nghiệm hoặc xuất theo hướng dẫn. StatA cần thêm nhóm tham chiếu hợp lệ.
6. Chạy notebook, đợi worker sẵn sàng; nhập ảnh/tiêu đề trên local và gửi yêu cầu.
7. Kết thúc: dừng notebook GPU riêng trên Kaggle, rồi `docker compose down`.

Để chỉ sửa CSS/HTML hoặc xem biểu đồ, bỏ qua toàn bộ phần Kaggle. Không cam kết thời gian inference trước khi đo trên môi trường thực tế.

## 8. Quy ước kết quả và kiểm tra

- Điểm mô hình của một upload không phải Accuracy/F1, cũng chưa mặc nhiên là xác suất hiệu chuẩn.
- Các kết quả Fakeddit min5 đang trình bày là dev; không gọi là test độc lập.
- ± là sample standard deviation giữa seed, không phải khoảng tin cậy.
- Nhãn Real/Fake thuộc taxonomy dữ liệu; không thay thế xác minh sự thật.
- Không thêm train loss nếu không có log huấn luyện tương ứng.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python -B -m unittest discover -s tests -p 'test_demo*.py'
node --check src/vlm_project/demo_web/app.js
node --check src/vlm_project/demo_web/analytics.js
node tests/demo_analytics_dom.mjs
```

Node chỉ cần cho kiểm tra JavaScript; không cần để chạy web. Một số test inference/export cần thư viện khoa học của môi trường nghiên cứu; môi trường nhẹ chỉ phục vụ UI không bảo đảm chạy toàn bộ test. Không chạy train/inference lớn để kiểm tra giao diện.

Kiểm tra thủ công: mở đủ 5 trang; đổi task/pipeline; xem nhóm thu gọn; upload/thay ảnh; chuyển trang vẫn giữ form; xem biểu đồ; khi có worker kiểm tra submit, cancel, chi tiết và tải JSON. DOM test không thay thế kiểm tra responsive/trình duyệt.

## 9. Khi bàn giao và push

Gửi file này hoặc đường dẫn GitHub `docs/DEMO_HANDOFF.md`. Commit phần demo phải kèm toàn bộ `demo_web/` (bao gồm snapshot), service, worker, inference, CLI, Docker, kiểm thử và các tài liệu liên quan. Không gửi dataset, venv, checkpoint, credentials, token phiên hoặc bundle notebook chứa tài nguyên riêng. Slide không thuộc phạm vi bàn giao này.
