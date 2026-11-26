# Demo localhost và GPU Kaggle

## Mô phỏng phương pháp

Mục **Mô phỏng phương pháp** (`/#view-simulations`) nằm ngay dưới **Bài toán**.
Bộ chọn hiển thị 26 phương pháp trong catalog demo, chia Zero-shot, Thích nghi không nhãn,
Few-shot và Đối chứng bổ sung; tiêu đề ghi rõ phương pháp đang mô phỏng.
Mỗi mục giữ tiến trình riêng, có nút
Trước/Tiếp/Xem lại từ đầu, đầu vào minh họa, biểu diễn, bước quyết định và đầu ra.
Có Chạy/Tạm dừng tự động, timeline chọn bước, sơ đồ nối đường xử lý và phần giải thích
mở theo nhu cầu. Đổi phương pháp hoặc rời trang sẽ dừng phát để tránh chạy ngầm.
Few-shot có bốn tình huống giả lập để thấy cả dự đoán đúng, bỏ sót và báo nhầm.
Tab này chạy ngay trên trình duyệt, không cần worker/GPU và không gửi job.
Ảnh mẫu là train ID `d89ibd`, được chọn trong tối đa năm dòng đầu TSV train min5;
asset local `simulations-sample.jpg` là bản sao ảnh gốc, tiêu đề gốc được giữ trong
tình huống mặc định. Các tình huống khác chỉ thay tiêu đề/điểm giả lập, không sửa ảnh nguồn.
Không đọc dữ liệu thô, test hoặc chạy encoder để tạo mẫu này.
Embedding 2D và điểm được dựng để giảng giải, không thay kết quả inference
hoặc thực nghiệm. Khối phép tính trình bày biểu thức, số thay vào và kết quả mỗi bước.
StatA giữ cố định vị trí chấm ảnh; điểm và thống kê lớp mới thay đổi.
R87 minh họa cách chọn ngưỡng từ 100 Fake selection giả lập; ngưỡng 0,36 không phải
ngưỡng trong artifact thật. Nhãn và điểm giả lập không được dùng làm số liệu báo cáo.

Khi dùng Docker, cập nhật giao diện bằng `docker compose up -d --build` rồi tải lại trang.

Tham khảo cách trình bày tương tác của [TensorFlow Playground](https://github.com/tensorflow/playground),
[CNN Explainer](https://github.com/poloclub/cnn-explainer) và
[EM, Step by Step](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/figures/interactive.html).
Chỉ tham khảo cách thể hiện luồng, màu gán mềm và bước cập nhật thống kê; không sao chép
code hay đồng nhất StatA với EM thuần túy. Nhãn minh họa vẫn được giữ gọn ở đầu trang.

## Luồng suy luận thật

Web nhận ảnh và tiêu đề, gửi yêu cầu tới hàng đợi trong RAM local. Notebook Kaggle
chủ động lấy yêu cầu qua HTTPS tunnel, chạy backbone trên CUDA và gửi dần kết quả.
Local không import Torch, không tải trọng số backbone, không train và không inference.
Ảnh được truyền tới Kaggle khi worker nhận yêu cầu; không tự tạo Kaggle Dataset cho mỗi ảnh.

Giao diện báo cáo gồm năm vùng: **Bài toán**, **Cấu hình suy luận**, **Kết quả suy luận**,
**Kết quả thực nghiệm**, **Lịch sử**. Dùng điều hướng để trình bày từng vùng độc lập. Các nhóm zero-shot,
thích nghi không nhãn và few-shot có thể đối chiếu trên cùng bảng kết quả.
Quy ước nội dung và nguồn tham khảo thiết kế: [DEMO_INTERFACE.md](DEMO_INTERFACE.md).
Các biểu đồ thực nghiệm xem được khi worker Kaggle chưa chạy: [danh mục biểu đồ](DEMO_EXPERIMENT_CHARTS.md).

## Khởi động local

### Cách khuyên dùng: Docker Compose

Mở Docker Desktop trước. Trong thư mục gốc dự án:

```sh
docker compose up -d --build
docker compose ps
docker compose logs -f tunnel
```

Mở http://127.0.0.1:8765. Lấy URL HTTPS từ log tunnel để cập nhật
`DEMO_BROKER_URL`. Lấy token mới trong giao diện → Thiết lập kết nối để cập nhật
`DEMO_WORKER_TOKEN`, rồi chạy notebook Kaggle. Không chạy thêm `scripts/demo.py serve`
hoặc cloudflared trên máy khi đã dùng Compose. Ctrl+C khi xem log chỉ thoát trình xem,
không tắt container đang chạy nền.

```sh
docker compose down       # Tắt và gỡ cả web lẫn tunnel
docker compose up -d      # Bật lại, dùng image đã build
docker compose logs -f    # Xem log cả hai container
```

Chạy lại `up -d --build` sau khi sửa code giao diện/API. Restart server sẽ mất lịch sử
phiên và đổi token; restart tunnel có thể đổi URL, cần cập nhật Secrets tương ứng.
Compose không dừng session GPU Kaggle: dừng riêng trên Kaggle sau demo.
Web chỉ publish cổng 8765 lên loopback của Mac. Tunnel chia sẻ network namespace với web,
truy cập gateway 8766 qua loopback nội bộ; không publish gateway lên host hoặc LAN.
Image web không chứa Torch, Kaggle CLI, datasets, head hay trọng số model.
RAM giới hạn web 512 MiB, tunnel 256 MiB; Docker Desktop có chi phí RAM riêng.
Không tự khởi động các container khi mở lại Docker Desktop (`restart: no`).

Nếu báo cổng 8765 đã được dùng, kiểm tra server Python cũ trước:

```sh
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

Chỉ dừng PID đã xác nhận là `scripts/demo.py serve` bằng `kill <PID>` rồi bật Compose.
Không kill mọi tiến trình Python. Nếu còn cloudflared cũ, Ctrl+C tại Terminal của nó.

### Chạy trực tiếp bằng Python

Dùng môi trường dự án đã có:

```sh
venv/bin/python scripts/demo.py doctor
venv/bin/python scripts/demo.py serve
```

Mở http://127.0.0.1:8765. Worker gateway ở `127.0.0.1:8766`; hai cổng chỉ bind loopback.
Máy mới chỉ cần môi trường nhẹ: `python3 -m pip install -r requirements-demo.txt`.
Không cần cài `requirements.txt` chứa model runtime trên máy chỉ phục vụ giao diện.

Nếu dùng `--port 8800`, giao diện ở 8800 và worker gateway ở 8801.

## Kết nối notebook với máy local

Terminal thứ hai:

```sh
cloudflared tunnel --url http://127.0.0.1:8766
```

Chỉ tunnel cổng **8766**, không tunnel cổng giao diện 8765.
Quick Tunnel cung cấp URL HTTPS tạm thời, đổi khi khởi động lại và không bảo đảm uptime.
Xem [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
Worker chỉ gọi HTTP thông thường nên không cần WebSocket/SSE.

Trong giao diện chọn **Thiết lập kết nối → Lấy mã kết nối phiên này**.
Trong Kaggle Notebook, thêm và bật quyền truy cập hai Kaggle Secrets:

| Secret | Giá trị |
|---|---|
| `DEMO_BROKER_URL` | HTTPS origin của tunnel, ví dụ `https://example.trycloudflare.com` |
| `DEMO_WORKER_TOKEN` | Mã phiên lấy từ giao diện local |

Không đặt token trong notebook, metadata, Git hoặc ảnh chụp màn hình. Mã thay đổi khi
khởi động lại server local; cần cập nhật Secret. Đóng tunnel khi kết thúc demo.
Notebook cần **Internet** và **GPU CUDA**. Worker không fallback về CPU local.

## Chuẩn bị và gửi notebook qua Kaggle CLI

CLI trong môi trường dự án: `venv/bin/python -m kaggle`. Kiểm tra đăng nhập bằng:

```sh
venv/bin/python -m kaggle kernels list --mine --page-size 3
```

Tạo bundle offline vào thư mục mới:

```sh
venv/bin/python scripts/demo.py prepare \
  --output bundles/demo \
  --owner nguyenhung3105 \
  --backbones metaclip2 openai_clip
```

Bundle chứa notebook và metadata private/GPU/Internet. Mã nguồn và một danh sách
config cho phép được nhúng vào notebook; không có dữ liệu ảnh, checkpoint hay credentials.
Lệnh không ghi đè bundle đã tồn tại; dùng tên thư mục mới khi cập nhật source.

Lần đầu, tải `bundles/demo/demo.ipynb` vào Kaggle Notebook để gán hai Secrets, bật
GPU/Internet rồi chạy. Sau khi notebook có quyền Secrets, có thể gửi phiên tiếp theo:

```sh
venv/bin/python -m kaggle kernels push -p bundles/demo
venv/bin/python -m kaggle kernels status nguyenhung3105/vlm-local-demo-worker
```

Hoặc thêm `--push` vào lệnh `prepare` để gửi ngay bundle mới. Lệnh push tạo/cập nhật
notebook và khởi chạy GPU; prepare không có `--push` hoàn toàn offline.
Xem [Kaggle Kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
Quyền Secrets trong phiên CLI/headless cần được xác nhận trên tài khoản thực tế;
nếu không có quyền, chạy notebook tương tác sau khi bật từng Secret.

Worker tự tải checkpoint về môi trường Kaggle. Không cần upload trọng số backbone từ local.
Lượt đầu cần cài thư viện, tải model và chuẩn bị prompt; đợi “GPU sẵn sàng” trước trình bày.
Chỉ một backbone nằm trong GPU tại một thời điểm để giới hạn VRAM. Chọn cả hai backbone
sẽ có thời gian đổi model; ưu tiên một backbone cho demo nhanh.

## Head few-shot và nhóm tham chiếu

Không fit head mỗi khi upload ảnh. Xuất các head đã học từ artifact full dev hoàn tất:

```sh
venv/bin/python scripts/demo.py export-assets \
  --output bundles/demo-assets \
  --binary-run results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run \
  --cascade-run results/final/kaggle/fakeddit_fewshot_cascade_min5_b7c54cfcca4a/run
```

Lệnh chỉ đọc metadata completed run và head `linear_models.npz` được chỉ định,
kiểm checksum run; không mở dataset/ảnh, không tải backbone, không fit lại.
Có thể nhúng các head nhỏ này trực tiếp vào notebook, không cần tạo Dataset riêng:

```sh
venv/bin/python scripts/demo.py prepare \
  --output bundles/demo-ready --owner nguyenhung3105 \
  --backbones metaclip2 openai_clip --assets-dir bundles/demo-assets
```

Bundle này có thêm head đã export và manifest checksum, vẫn không có ảnh, trọng số
backbone hoặc credentials. Một lựa chọn khác là upload thư mục assets thành
**private Kaggle Dataset**, rồi chuẩn bị notebook mới với:

```sh
venv/bin/python scripts/demo.py prepare \
  --output bundles/demo-with-assets --owner nguyenhung3105 \
  --dataset nguyenhung3105/vlm-demo-assets \
  --assets-path /kaggle/input/vlm-demo-assets
```

Đường dẫn asset phải chứa `manifest.json`. Nếu thiếu hoặc sai checksum/head dimension,
giao diện hiển thị lý do chưa chạy. Export hiện hỗ trợ head MetaCLIP2 từ run nguồn;
head MetaCLIP2 không được dùng cho OpenAI CLIP.

StatA/EM/Majority cần nhóm query. Mặc định các dòng này báo thiếu nhóm tham chiếu,
không gọi baseline một ảnh là kết quả thích nghi. Tùy chọn `export-assets --reference-run`
trích tối đa 74 feature **train**, gồm ảnh và tiêu đề, từ `features.npz` của run được chỉ định;
không xuất feature/nhãn dev hoặc test. Chỉ dùng khi được phép đọc feature cache nguồn.
Thêm cả ba nguồn trong cùng lần export để có một manifest thống nhất. Đây là chế độ
demo trên nhóm tham chiếu cố định, **khác protocol dev báo cáo**, được ghi chú trong kết quả.

## Những gì được so sánh

- Bài toán Real/Fake: CLIP ảnh/fusion, DCLIP/Waffle, StatA và few-shot ảnh/ảnh + tiêu đề.
  Các đối chứng khác nằm trong mục bổ sung; chỉ tiêu đề không nằm trong phần chính.
- Bài toán sáu lớp: baseline, DCLIP/Waffle/CALIP, EM/StatA/Majority phẳng,
  hierarchy CLIP/EM/StatA/Majority và few-shot phẳng/cascade/default/R87.
- Prototype Fakeddit dùng config hiện có. Prompt tác giả DCLIP/Waffle sinh bằng
  module hiện có từ commit cố định, Waffle seed demo 0 và 15 cặp prompt.
- Waffle–semantic–StatA DTD/EuroSAT và CIFAR dùng taxonomy khác, không đưa vào bảng
  Fakeddit. Demo OpenAI dùng adapter OpenCLIP, không phải tái hiện paper runtime gốc.
- Few-shot dùng toàn bộ hai policy × năm seed nếu có; dòng chính hiển thị B seed 0,
  hoặc L seed 0 nếu chỉ có legacy. Chi tiết/JSON chứa mọi dự đoán; không chọn seed theo ảnh.
- Cascade dùng hard gate gốc, tie mặc định chọn Real. Phân phối joint chỉ để xem điểm;
  nhãn cuối tuân theo gate, không lấy argmax joint để thay quyết định cascade.
- Điểm softmax chưa được hiệu chuẩn. EM/Majority/hierarchy chỉ trả nhãn thì không tạo
  confidence giả. Không hiển thị Accuracy/F1 cho một upload không có ground truth.

## Vận hành và kiểm tra

Ảnh JPEG/PNG/WebP tối đa 8 MB/20 megapixel. Tiêu đề tối đa 2.000 ký tự; tokenizer
có thể cắt tiêu đề theo context model. Bộ nhớ phiên tối đa 20 lượt, cache tối đa 8 ảnh
trên backbone đang nạp. Không ghi upload xuống đĩa. Khởi động lại server mất lịch sử phiên.
Yêu cầu đã nhận có lease 120 giây, worker gia hạn mỗi 10 giây. Mất worker thì lượt
tiếp theo có thể nhận lại sau khi lease hết; kết quả từ worker cũ bị từ chối.
Hủy loại kết quả đến muộn; kernel GPU đang tính có thể cần hoàn tất bước hiện tại.

Không cam kết thời gian suy luận khi chưa đo GPU thật. JSON ghi tổng thời gian worker,
thời gian mã hóa theo backbone và từng phương pháp. Tổng thời gian chưa bao gồm thời gian
chờ hàng đợi/truyền về local. Đổi backbone có thể phải nạp lại model và prototype.
Các nhánh có thể dùng chung kết quả thích nghi đã tính trong lượt chạy; thời gian từng dòng
là chi phí thực thi thêm của dòng đó, không phải benchmark độc lập giữa phương pháp.

```sh
venv/bin/python -B -m unittest discover -s tests -p 'test_demo*.py' -v
node --check src/vlm_project/demo_web/app.js
```

Kiểm thử dùng ảnh/feature/head tổng hợp, không truy cập dữ liệu thật hay nạp model.
`tests/demo_browser_smoke.py` kiểm giao diện bằng Playwright và Chrome có sẵn, độc lập
với model/GPU. Cài Playwright trong môi trường QA riêng nếu cần; không phải dependency local demo.
