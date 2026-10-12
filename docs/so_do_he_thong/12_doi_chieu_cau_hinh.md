# 12 — Danh sách đầy đủ và cấu hình hiệu lực

**Trang này phân biệt phần đã tích hợp, phần đang bật và khả năng riêng trong repo clone.** Các giá trị bên dưới đối chiếu ngày 12-10-2026; khi YAML/code đổi cần cập nhật cùng tài liệu.

[Mục lục](README.md) · [Tổng quan](01_tong_quan.md)

## Backbone và trạng thái

| Model/backbone | Code/registry | Chính | Adapted | Upstream |
|---|---|---|---|---|
| MetaCLIP2 `metaclip2_vith14` | VLMWrapper/OpenCLIP | Đang bật | Theo base config, đang chọn | Không dùng |
| DFN5B `dfn5b_vith14` | VLMWrapper/OpenCLIP | Đã hỗ trợ; YAML comment | Có thể chọn qua base config | Không dùng |
| SigLIP2 `siglip2_so400m` | VLMWrapper/OpenCLIP | Đã hỗ trợ; YAML comment | Có thể chọn qua base config | Không dùng |
| OpenAI CLIP ViT-B/32 | Checkout OpenAI + OpenAIClipBackend | Không nằm registry chính | Không nằm registry chính | Đang chọn |
| OpenAI CLIP RN50, ViT-L/14 | Cùng backend CLIP gốc | Không | Không | Có thể chọn; cũng thuộc CLI Waffle gốc |
| OpenAI CLIP RN101, RN50x4/x16/x64, ViT-B/16, ViT-L/14@336px | `clip.available_models()` trong checkout | Không | Không | Backend có khai báo; chưa kiểm chứng mọi checkpoint |

WaffleCLIP/DCLIP là phương pháp prompt/scoring trên backbone; xem [WaffleCLIP](07_waffleclip.md), [DCLIP](08_dclip.md). Chọn backbone và chọn method là hai trục thí nghiệm riêng.

## YAML chính hiện tại

| Dataset | Task đang bật | Limit | Prompt aggregation | w ảnh | Scale text | Quyết định |
|---|---|---:|---|---:|---|---|
| Fakeddit | 6way | 1000 | mean_sim, variant current | 0.75 | 25 | Softmax từng nhánh → fusion → argmax |
| CrisisMMD | informativeness | 1000 | mean_embed | 0.5 | null → scale model | Softmax từng nhánh → fusion → argmax |
| MM-IMDb | genres | 1000 | mean_sim | 0.5 | Không áp dụng khi relative_z | Relative_z > 1.0; fallback_top1 mặc định true |

Batch inference chính 16. MM-IMDb chunking bật, `chunk_aggregation=mean_embed`. `threshold=0.22` có sẵn cho fixed_threshold nhưng không có hiệu lực ở relative_z hiện tại.

Các task đã hỗ trợ còn lại: Fakeddit 2way và CrisisMMD humanitarian. Đổi task đổi taxonomy/cột nhãn tương ứng, không đổi tự động backbone.

## Method tích hợp trong hai runner nghiên cứu

Mọi runner hiện có flag quan hệ mặc định **false**. Pipeline chính dùng `inference.image_text_relation.enabled`; adapted/upstream dùng `image_text_relation.enabled` ở cấp gốc YAML riêng. Bật true và mode auto: Fakeddit 6way class_aware (weight 0.25, 0=match/2=mismatch), task khác diagnostic. Upstream original_image luôn chỉ diagnostic. Xem [công thức và phạm vi](13_quan_he_anh_text.md).

| Runner | Method hợp lệ | Đang bật | Có lặp seed? |
|---|---|---|---|
| Upstream | clip | Có | Không |
| Upstream | waffle | Có | Có, hiện `[0]` |
| Upstream | dclip_labels | Có | Không |
| Upstream | dclip | Có | Không |
| Adapted | baseline | Có | Không |
| Adapted | current_mean_sim | Có | Không |
| Adapted | class_name | Có | Không |
| Adapted | waffle | Có | Có, hiện `[0,1,2,3,4]` |
| Adapted | dclip | Đã hỗ trợ, chưa bật | Không |

Upstream bật CrisisMMD humanitarian 5 mẫu, batch 16, outputs original_image/multimodal, 15 cặp Waffle. Adapted chọn CrisisMMD humanitarian 1000 mẫu, 15 cặp Waffle; backbone/scale/fusion/prompt text lấy base config. Hai runner không kế thừa số mẫu của nhau.

## Toàn bộ mode khai báo trong checkout WaffleCLIP

Phần này là bản đồ khả năng của repo đã clone. **Runner dự án chưa chuyển tiếp các ablation này**, ngoài đường clip/waffle và các hàm DCLIP được ghi phía trên. Các mode có chung phần ảnh → encoder → similarity → aggregate → nhãn; khác nhau ở nguồn/tạo prompt trước text encoder.

| Mode trong `METHODS` | Input của khối tạo prompt → xử lý → output prompt |
|---|---|
| `clip` | Tên lớp → template mặc định → một prompt/lớp |
| `gpt_descriptions` | Mô tả riêng lớp → ghép tên lớp/template → nhiều prompt mô tả |
| `random_descriptions` | Pool mô tả mọi lớp → lấy ngẫu nhiên riêng từng lớp → prompt mô tả phân bổ lại |
| `shared_random_descriptions` | Pool mô tả → một bộ ngẫu nhiên chung → ghép tên từng lớp |
| `swapped_descriptions` | Các bộ mô tả lớp → hoán đổi bộ giữa lớp → prompt với mô tả của lớp khác |
| `scrambled_descriptions` | Từ trong mô tả của mỗi lớp → lấy ngẫu nhiên có lặp theo độ dài câu → prompt trộn từ |
| `random_description_words` | Pool từ descriptor → sinh câu ngẫu nhiên → phân bổ riêng mỗi lớp |
| `shared_random_description_words` | Pool từ descriptor → sinh chuỗi từ → chia sẻ một bộ giữa lớp |
| `adaptive_random_characters` | Alphabet từ mô tả + độ dài tên lớp → chuỗi ký tự riêng mỗi lớp |
| `waffle_characters` | Chuỗi ký tự → chọn một bộ chia sẻ → thay tên lớp trong câu |
| `adaptive_randomness` | Từ điển + alphabet → cặp chuỗi từ/ký tự riêng từng lớp |
| `waffle` | Từ điển + alphabet → cặp từ/ký tự → chia sẻ bộ suffix giữa lớp |
| `adaptive_randomness_and_gpt_descriptions` | Random riêng lớp + mô tả đúng lớp → trộn và lấy mẫu theo budget |
| `waffle_and_gpt` | Random chia sẻ + mô tả đúng lớp → trộn/lấy mẫu; bộ cuối có thể khác giữa lớp |
| `prompt_ensemble` | Bộ template → lấy ngẫu nhiên 2×waffle_count có lặp → thay tên lớp |
| `gpt_descriptions_and_prompt_ensemble` | Template lấy mẫu + descriptor đúng lớp lấy mẫu → ghép prompt |
| `waffle_and_prompt_ensemble` | Template lấy mẫu + cặp từ/ký tự → chọn bộ chia sẻ giữa lớp |
| `context_prompt_ensemble` | Template lấy mẫu + concept đọc từ prefix + tên lớp → prompt có concept |
| `shared_random_characters` | Có tên trong METHODS nhưng không có nhánh xử lý riêng trong `load_gpt_descriptions` của commit hiện tại; không coi là mode đã được xác minh |

High-level concept guidance còn dùng `label_before_text`, ví dụ `A photo of a food: a `. Repo có bước sinh concept trước thí nghiệm; runner dự án không tự chạy bước này. Với `shared_random_characters`, nhận tên từ CLI không đủ chứng minh thuật toán đã triển khai; source hiện rơi qua các nhánh khác và trả dictionary còn lại.

## Các lựa chọn scoring trong repo tác giả

| Nơi | Lựa chọn tồn tại | Dự án đang dùng |
|---|---|---|
| Waffle `aggregate_similarity` | mean, max, sum; CLI aggregate liệt kê mean/max | mean |
| Waffle `merge_predictions` | Mean embedding, L2 lại, rồi cosine | Không chuyển tiếp cờ này trong upstream |
| DCLIP `aggregate_similarity` | mean, max, sum | mean |
| DCLIP template | prepend/append, prefix/separator/suffix, chỉnh descriptor | Upstream prepend, không prefix/hậu tố, separator `, ` |
| OpenAI CLIP API | Forward logits hoặc encode riêng | Runner encode riêng rồi tính điểm |

Các helper giải thích từng descriptor tồn tại trong DCLIP; logger của dự án chưa xuất lời giải thích tương ứng. Các ablation được liệt kê để hiểu checkout, không phải xác nhận đã chạy chúng trên dataset của bạn.

## Repo tác giả và commit khóa

| Checkout | URL nguồn từ cấu hình | Commit |
|---|---|---|
| OpenAI_CLIP | `https://github.com/openai/CLIP.git` | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| WaffleCLIP | `https://github.com/ExplainableML/WaffleCLIP.git` | `7a1b8ee48e31285f62ecd839fecb6b89cbef81f1` |
| DCLIP | `https://github.com/sachit-menon/classify_by_description_release.git` | `d1a3d8920af89bd0d6a692ed3afeb277df46e082` |

Checkout `third_party/` bị Git dự án bỏ qua. Liên kết code clone trong tài liệu dùng được khi checkout còn trên máy; sau khi chỉ clone repo chính cần chuẩn bị checkout đúng phiên bản để mở các file nguồn đó. Sơ đồ SVG và phần diễn giải vẫn nằm trong tài liệu của repo chính.

## Điểm kiểm khi đổi cấu hình

Giữ cùng model/backbone, task, taxonomy, cohort, weight, text scale, decision và nhánh text khi mục tiêu là đối chứng prompt ảnh. Đổi backbone đồng thời với đổi method không cho phép quy chênh lệch metric riêng cho thuật toán. Descriptor hiện tại là mô tả taxonomy cố định, không phải descriptor GPT-3 của benchmark tác giả.

## Nguồn đối chiếu

[Registry](../../src/models/model_registry.py) · [YAML chính](../../configs/experiment_config.yaml) · [YAML upstream](../../configs/research/upstream.yaml) · [YAML adapted](../../configs/research/adapted.yaml) · [sources.json](../../configs/research/sources.json).

[Waffle METHODS và nhánh tạo prompt](../../third_party/WaffleCLIP/waffle_tools.py) · [DCLIP loader](../../third_party/DCLIP/loading_helpers.py) · [OpenAI available_models](../../third_party/OpenAI_CLIP/clip/clip.py).
