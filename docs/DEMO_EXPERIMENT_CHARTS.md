# Biểu đồ kết quả thực nghiệm trên giao diện demo

Mục **Kết quả thực nghiệm** độc lập với **Kết quả suy luận** của một ảnh. Biểu đồ
đọc snapshot thống kê được đóng gói sẵn, không gọi GPU, không fit head và không mở dataset.

## Nội dung có thể xem

| Biểu đồ | Dữ liệu và điều kiện |
|---|---|
| So sánh phương pháp | Mean/sample std của Accuracy, Macro-F1, BA; Top-5 khi artifact có |
| Đóng góp ảnh–tiêu đề | Các nhánh ảnh, tiêu đề, kết hợp và support policy trong cùng run |
| Ma trận nhầm lẫn | Tính lại từ prediction, đảo đúng mapping nhãn prompt/few-shot; chọn seed hoặc mean |
| Cặp lớp nhầm lẫn | Với taxonomy lớn: tối đa 30 cặp phổ biến thay ma trận khó đọc |
| Chỉ số từng lớp | Precision/Recall/F1 và std, tìm kiếm và phân trang 20 lớp |
| Phân bố lớp | Support của nhãn thật trong cohort đánh giá, không phải phân bố dự đoán |
| Biến thiên seed | Điểm theo seed/tổ hợp seed trên cùng cohort; không phải epoch |
| Gate và head con | Ba nhóm lỗi cascade, recall gate, accuracy child toàn Fake và có điều kiện |
| Chênh lệch ghép cặp | Delta và std từ bảng comparisons; giữ cả mức tăng và mức giảm |
| Precision–Recall selection | Các điểm ngưỡng đã lưu trên selection dev; không gọi audit là test độc lập |
| ROC/PR đánh giá | Chỉ phương pháp lưu p(Fake); AUC/AP tính từ mọi score với xử lý ties đúng |
| Chẩn đoán xác suất | NLL, entropy chuẩn hóa, mean confidence khi có branch_diagnostics |
| Recall subtype | Phân biệt binary detection/gate recall và recall sáu lớp đầu cuối |
| Ngân sách support | Đếm labels_6way/quotas trong support artifact; không phải learning curve |
| Oracle gate | Phân tích dùng nhãn gate thật, tách khỏi so sánh phương pháp triển khai |

Từng run chỉ hiện các mục có đủ dữ liệu. Có thể thay chỉ số, phương pháp, seed, đơn vị
ma trận hoặc policy theo biểu đồ. Bảng số liệu, CSV, JSON và SVG chính được cung cấp
để kiểm tra; không tạo số liệu khi thiếu artifact.

Không có train loss theo epoch vì các full run được kiểm tra không lưu lịch sử đó.
NLL là chẩn đoán đầu ra đã đánh giá, không phải loss huấn luyện backbone. Các run support
có cùng tổng ngân sách không được biến thành đường accuracy theo số shot.

## Phạm vi dữ liệu

- Nguồn: `results/published/manifest.json` và audit được manifest tham chiếu.
- Có 27 completed full run đã passed audit; loại smoke, pilot và diagnostic khỏi bảng chính.
- Một bộ đối chiếu Fakeddit nhị phân bổ sung chỉ gộp zero-shot/thích nghi và few-shot
  khi **cohort hash, model và số mẫu khớp nhau**. Tổng cộng 28 lựa chọn trên giao diện.
- Các run khác cohort, khác split hoặc khác backbone được chọn và hiển thị riêng.
- Bản cũ và min5 không bị gộp theo tên phương pháp. Split `dev` luôn được ghi là đánh giá
  phát triển; tên split `test` của benchmark không tự tạo kết luận xác nhận độc lập.

Exporter kiểm tra SHA256 của audit, published summary, run completion và các artifact
được dùng. Nó stream `predictions.tsv` để tính lại accuracy/Macro-F1/BA và ma trận,
đối chiếu mean/std với summary, rồi chỉ xuất thống kê tổng hợp. Không xuất sample ID,
ảnh, tiêu đề, feature, posterior theo từng mẫu hoặc hệ số model.

Ma trận mean là số đếm trung bình/seed, không cộng seed thành nhiều mẫu độc lập. F1 theo
lớp được tính từng seed rồi tổng hợp, không tính F1 từ ma trận trung bình để thay mean F1.
Thanh sai số là **sample std, ddof=1**, không phải khoảng tin cậy.

AUC ROC được tính từ toàn bộ nhóm ngưỡng có score bằng nhau; AP dùng tổng
`ΔRecall × Precision`. Đường ROC/PR được lấy mẫu tối đa 201 điểm để hiển thị; AUC/AP
không được tính từ đường rút gọn. Quy ước lớp dương là **Fake** và phụ thuộc đúng
mapping nhãn của run nguồn. PR selection và PR trên cohort đánh giá được tách thành
hai mục, ghi rõ nguồn và split.

## Cập nhật snapshot và chạy

Snapshot có sẵn trong `src/vlm_project/demo_web/experiment-results.json` và `experiments/`.
Máy chỉ xem giao diện không cần có artifact đầy đủ hoặc thư viện ML.
Khi có thay đổi kết quả nguồn và đã có artifact local đầy đủ:

```sh
venv/bin/python -B scripts/export_demo_results.py
docker compose up -d --build
```

Exporter chỉ đọc các đường dẫn full run được manifest liệt kê; không quét data/raw,
không đọc feature cache/checkpoint và không chạy chương trình inference. Sau đó mở
http://127.0.0.1:8765/#view-experiments. Không cần bật Kaggle để xem biểu đồ.

Snapshot được tách thành từng run và tải khi chọn để không nạp toàn bộ thống kê vào
trình duyệt ngay khi mở trang. Cổng worker và cơ chế token giữ nguyên; thay đổi này
không yêu cầu tạo lại notebook Kaggle. Nếu recreate server/tunnel thì Secrets phiên
cần được cập nhật theo quy trình hiện tại.

## Kiểm tra

- RED: bốn test exporter thất bại trước khi có module xuất dữ liệu.
- Kiểm fixture: checksum sai bị từ chối; không gộp khác cohort/model/mẫu; mapping binary
  prompt/few-shot đúng; tổng ma trận đúng số mẫu; smoke bị loại.
- Kiểm snapshot: SHA256 từng file, phạm vi metric [0,1], tổng ma trận/sample và giá trị
  Accuracy support cân bằng được đối chiếu bảng đã lưu.
- ROC/AP với score trùng nhau được đối chiếu độc lập với scikit-learn trên dữ liệu tổng hợp.
- `node tests/demo_analytics_dom.mjs`: 28 bộ kết quả, 171 biểu đồ, bảng và đổi đơn vị
  ma trận PASS; không có NaN/Infinity/undefined trong SVG. Đây là kiểm thử logic DOM,
  không thay thế kiểm tra bố cục bằng trình duyệt.
- `tests/demo_analytics_browser.py` có kiểm tra toàn bộ chart, điều khiển metric/seed,
  download CSV/SVG/JSON và ba viewport. Chưa tính là PASS khi chưa có kết quả Chrome.
- Không commit/push hoặc gửi job Kaggle trong lượt bổ sung biểu đồ.

Kiểm tra bàn giao: container đã được build một lần, nhưng đối chiếu SHA256 sau lần bổ
sung ROC/PR xác nhận bản đang chạy chưa trùng bản mã/snapshot cuối. Không tuyên bố
deployment cuối hoặc Chrome regression PASS khi hai tác vụ đó chưa trả kết quả.

Tự đánh giá: accuracy 4/5 (số liệu và ties AUC/AP có kiểm độc lập); completeness 3/5
(đã có mã, snapshot và 171 cấu hình chart qua DOM; còn chốt Docker/Chrome);
clarity 4/5 (tách inference, dev, selection và oracle); actionability 4/5 (có exporter,
snapshot và Compose); conciseness 4/5 (tải từng run, chia panel). Trung bình 3,8/5.
Ưu tiên: hoàn tất phê duyệt thực thi Docker/Chrome, đối chiếu checksum bản chạy và ghi
kết quả QA cuối. Không hạ tiêu chuẩn bằng cách coi DOM test là kiểm tra bố cục.
