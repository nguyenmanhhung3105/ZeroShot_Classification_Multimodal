# Bản đồ hệ thống zero-shot

**Bắt đầu tại [01_tong_quan.md](01_tong_quan.md).** Mỗi trang có sơ đồ input → xử lý → output, bảng giải thích các khối và đường dẫn tới code thực thi.

Đối chiếu code/cấu hình ngày **12-10-2026**. Bộ sơ đồ ban đầu chỉ dựa trên code. Phần quan hệ ảnh–text bổ sung sau đó có thử thật trên năm dòng đầu Fakeddit; xem báo cáo riêng. Thông số chưa đo từ runtime không được tự suy ra từ tên checkpoint.

## Đọc theo nhu cầu

| Bạn muốn hiểu | Trang |
|---|---|
| Toàn bộ hệ thống và các nhánh nghiên cứu | [01 — Tổng quan](01_tong_quan.md) |
| Từ ảnh + text đến nhãn dự đoán trong code của bạn | [02 — Pipeline chính](02_pipeline_chinh.md) |
| Backbone MetaCLIP2 đang bật | [03 — MetaCLIP2](03_metaclip2.md) |
| Backbone DFN5B đã tích hợp | [04 — DFN5B](04_dfn5b.md) |
| Backbone SigLIP2 đã tích hợp | [05 — SigLIP2](05_siglip2.md) |
| Model CLIP và checkpoint của repo OpenAI đã clone | [06 — OpenAI CLIP](06_openai_clip.md) |
| Tại sao WaffleCLIP sinh nhiều prompt ngẫu nhiên | [07 — WaffleCLIP](07_waffleclip.md) |
| DCLIP dùng mô tả lớp như thế nào | [08 — DCLIP](08_dclip.md) |
| Runner dùng hàm tác giả trên dữ liệu của bạn | [09 — Nghiên cứu upstream](09_nghien_cuu_upstream.md) |
| Runner thích nghi phương pháp trên backbone của bạn | [10 — Nghiên cứu adapted](10_nghien_cuu_adapted.md) |
| Dataset, taxonomy, chunking, metric và file kết quả | [11 — Dữ liệu và output](11_du_lieu_output.md) |
| Danh sách đầy đủ backbone/method, trạng thái bật và phạm vi repo clone | [12 — Đối chiếu cấu hình](12_doi_chieu_cau_hinh.md) |
| Quan hệ ảnh–text và công thức match/mismatch/neutral | [13 — Quan hệ ảnh–text](13_quan_he_anh_text.md) |

## Bố cục và ký hiệu thống nhất

Sơ đồ dùng **SVG được dựng sẵn**, xem trực tiếp trong Markdown Preview hoặc GitHub. Nếu trình đọc không hiện ảnh, mở liên kết SVG cạnh sơ đồ. Các sơ đồ đọc **từ trên xuống dưới**; những nhánh song song được đặt cùng hàng. Không cần plugin Mermaid.

| Màu khối | Ý nghĩa |
|---|---|
| Xanh dương | Input: ảnh, text, prompt, nhãn thật |
| Cam | Cấu hình và điều kiện thí nghiệm |
| Xám | Tiền xử lý, gộp, điều khiển luồng |
| Tím | Encoder/backbone đã huấn luyện sẵn |
| Vàng | Similarity, hiệu chỉnh, fusion |
| Xanh lá | Nhãn dự đoán, metric, file kết quả |

Mũi tên liền là dòng dữ liệu. Mũi tên đứt là cấu hình hoặc dữ liệu dành cho đánh giá. Hai ô text encoder trong một sơ đồ sử dụng **cùng model và trọng số**, không phải hai model được huấn luyện riêng.

`N`: số mẫu hợp lệ; `C`: số lớp; `D`: chiều embedding của backbone; `K_c`: số prompt của lớp `c`; `L`: context tokenizer. `D`, kích thước ảnh và một số chi tiết encoder được lấy từ model khi chạy, không gán chung cho tất cả backbone.

## Giữ bố cục khi cập nhật

Nguồn sơ đồ là [so_do/tao_so_do.py](so_do/tao_so_do.py). Script chỉ dùng thư viện chuẩn Python và Graphviz, không import pipeline hay truy cập dataset. Mỗi sơ đồ có thêm nguồn `.dot` và ảnh `.svg` trong `so_do/`.

```bash
python3 docs/so_do_he_thong/so_do/tao_so_do.py
```

Quy ước chỉnh sửa: giữ hướng trên xuống, giới hạn mỗi khối vào một trách nhiệm, dùng xuống dòng chủ động, xếp nhánh ảnh/text/prompt cùng hàng, tách hình khi quy tắc quyết định khác nhau. Sửa nguồn rồi dựng lại SVG; không kéo vị trí hoặc sửa tay SVG. Hình ảnh cùng nguồn trong Git giúp bố cục ổn định giữa các lần xem.

## Phạm vi chứng cứ

Tài liệu dựa vào `src/`, `configs/`, README và code các checkout `third_party/`. Các đường dẫn dữ liệu được lấy từ cấu hình; bộ tài liệu không xác minh nội dung hoặc thống kê của các file dữ liệu. Các checkpoint không được tải/nạp để lấy thông số kiến trúc. Phương pháp trong repo nghiên cứu và phương pháp đang được runner dự án gọi được ghi rõ ở từng trang.

Tài liệu trạng thái thí nghiệm: [research.md](../research.md), [báo cáo chạy thử](../zeroshot_smoke_20261011.md), [bàn giao](../ban_giao_du_an.md). Flag quan hệ mới mặc định tắt trong cả ba YAML; kiểm thử hiện có 110 test, gồm mở rộng flag upstream trên cả năm task và invariance original_image.
