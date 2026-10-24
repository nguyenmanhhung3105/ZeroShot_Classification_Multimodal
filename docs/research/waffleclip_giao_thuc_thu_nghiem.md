# WaffleCLIP — rà soát trước triển khai và giao thức thử nghiệm đầu tiên

Ngày: **22/10/2026**. Tài liệu này chốt phép thử đối chứng trước khi phát triển biến thể riêng. Cơ sở nghiên cứu nằm trong [phân tích WaffleCLIP](waffleclip_phan_tich_va_huong_thich_nghi.md).

**Trạng thái bản rà soát ban đầu:** đã đối chiếu code và kiểm tra phần sinh prompt/tokenizer; ở bước đó chưa thêm profile chạy mới, chưa thay scoring, chưa chạy model thật hoặc gửi Kaggle. Kết quả dưới đây là kiểm tra kỹ thuật, không phải Accuracy của model trên CrisisMMD.

**Cập nhật triển khai:** đã tạo [Waffle lab độc lập](../../experiments/waffle_lab/README.md), gồm mốc CLIP/Waffle và các đối chứng cải tiến. Lab có profile riêng, khóa cohort và tổng hợp Accuracy qua seed. Mục 3 dưới đây mô tả runner upstream cũ; lab xử lý các điểm đó trong đường chạy riêng, không sửa phương pháp gốc. Chưa chạy inference thật cho các biến thể mới.

## 1. Nguồn tác giả và thiết lập làm mốc

Checkout `third_party/WaffleCLIP` sạch, ở commit **`7a1b8ee48e31285f62ecd839fecb6b89cbef81f1`**, trùng cấu hình nguồn dự án. Bộ test upstream cũng xác minh checkout DCLIP và OpenAI CLIP theo commit đã ghim.

| Thành phần | Thiết lập đã đối chiếu | Nguồn |
|---|---|---|
| Phương pháp | Tạo nhiều prompt quanh tên lớp, không sinh từ ảnh | `load_gpt_descriptions`, mode `waffle` |
| Backbone làm mốc | OpenAI CLIP ViT-B/32, frozen | `base_main.py`; không đổi sang MetaCLIP2 trong phép thử đầu |
| Tiền xử lý ảnh | Resize, center crop, RGB, chuẩn hóa CLIP | `_transform()` và preprocess checkpoint trong wrapper |
| Ngân sách | 15 cặp, tức 30 prompt/lớp | `waffle_count=15` |
| Nguồn từ | `word_list.pkl` của tác giả | Code gốc; file hash đã ghi trong audit |
| Nguồn ký tự | Alphabet rút từ descriptor JSON | Code gốc, không thay bằng ASCII ở phép thử làm mốc |
| Hậu tố | Cùng một bộ hậu tố cho các lớp trong cùng seed | Phần cuối nhánh `waffle` |
| Seed | `0,1,2,3,4,5,6` | Vòng `rep in range(reps)` gọi `seed_everything(rep)` |
| Mã hóa | Chuẩn hóa từng embedding ảnh và prompt | Code gốc và adapter upstream |
| Gộp điểm | Trung bình cosine theo lớp | `aggregate_similarity(..., 'mean')`; không bật merge |
| Quyết định | Argmax điểm ảnh | Bài toán một nhãn; tweet không tham gia |
| Nhánh mở rộng | Concept miền, descriptor và Waffle + descriptor là cấu hình riêng | README và các mode tác giả; không ghép vào mốc đầu |

Nguồn trực tuyến: [repo tác giả](https://github.com/ExplainableML/WaffleCLIP), [paper, mục 3–4 và Appendix](https://arxiv.org/html/2306.07282v2). Các trang GitHub ở URL commit trả cache miss trong lần truy cập web này; chi tiết thực thi được đối chiếu với chính checkout local đã ghim.

## 2. Kiểm tra trực tiếp đã thực hiện

Đã nạp riêng các định nghĩa hàm sinh prompt của tác giả, không thực thi loader benchmark ở top-level. Từ điển được đọc bằng bộ đọc giới hạn kiểu chuỗi của dự án. Tokenizer dùng BPE OpenAI CLIP local; không tải checkpoint. Chỉ dùng tên lớp và JSON mô tả, **không mở TSV/ảnh dataset thực**.

| Kiểm tra | Informativeness | Humanitarian |
|---|---:|---:|
| Số lớp | 2 | 8 |
| Số seed | 7 | 7 |
| Prompt Waffle/lớp/seed | 30 | 30 |
| Tổng prompt Waffle đã kiểm | 420 | 1.680 |
| Độ dài gồm token đầu/cuối | 13–20 | 14–27 |
| Có prompt vượt 77 token | Không | Không |
| Cùng seed tái tạo cùng prompt | Đạt | Đạt |
| Bảy seed tạo bảy bộ prompt khác nhau | Đạt | Đạt |
| Đúng thứ tự lớp và hậu tố chia sẻ | Đạt | Đạt |
| Khôi phục trạng thái RNG sau lời gọi wrapper | Đạt | Đạt |

Baseline `clip` có một prompt/lớp, dài tối đa 9 token ở informativeness và 12 token ở humanitarian. Các độ dài này chỉ áp dụng cho **bộ tên lớp và template hiện tại**, không bảo đảm cho alias/concept mới.

Đã so với bộ sinh adapted cùng số seed: **0/7 bộ prompt trùng nguyên văn ở mỗi task**. Khác biệt này phù hợp với việc adapted đổi alphabet/quy trình RNG; không phải bằng chứng adapted tốt hơn hoặc kém hơn. Vì vậy phép thử đầu dùng **hàm tác giả qua upstream**.

Chứng cứ máy đọc: [waffleclip_prompt_audit_20261022.json](waffleclip_prompt_audit_20261022.json), gồm hash nguồn, từ điển, descriptor và từng bộ prompt.

Bộ test hiện có đã chạy:

```sh
PYTHONPATH=tests venv/bin/python -B -m unittest -q tests/test_upstream_projects.py
```

**9/9 test đạt.** Test dùng tokenizer thật và embedding/ảnh/TSV tổng hợp trong thư mục tạm; kiểm tra hàm tác giả, mean similarity, prepare-only, log và các đường quyết định. Không tải trọng số hay đánh giá dữ liệu thật. Code gốc có `ResourceWarning` do mở `word_list.pkl` không dùng context manager; không làm test thất bại, chưa sửa checkout tác giả.

## 3. Những điểm phải xử lý hoặc ghi rõ trước khi chạy chính

### 3.1. Profile hiện tại chưa thể hiện đúng nghiên cứu mới

`configs/launcher_3000.yaml` hiện có baseline fusion và DCLIP adapted, chưa có job upstream Waffle chỉ ảnh. `configs/launcher.yaml` có hai job `upstream_crisismmd_*_image_only` nhưng mặc định 1.000 mẫu và seed 0. Khi triển khai cần một profile Waffle rõ ràng trỏ về cohort mới, tránh vô tình chạy giới hạn cũ hoặc chọn adapted thay upstream.

Giữ quy ước vận hành hiện tại: **một task, một model, một method, một output, một seed mỗi lệnh Kaggle**. Không gộp mọi backbone/phương pháp vào một job dài. Cờ số cặp phải được chốt ở research YAML; không giả định launcher hiện truyền tiếp mọi trường tùy ý từ job.

### 3.2. Cần xác minh cohort qua các lượt, không chỉ bật sample lock ở base config

`evaluate_trial()` của upstream hiện đặt `sample_lock_mode='off'` trong bản config dùng audit. Vì vậy sample lock ở cấu hình chung **không tự bảo đảm** các lượt upstream trùng bộ mẫu. Khi triển khai cần kiểm soát cohort ở bước chuẩn bị/chạy hoặc đối chiếu fingerprint độc lập, gồm thứ tự ID, nhãn task, ảnh và nguồn dữ liệu, tách khỏi hash prompt.

Trong lần rà soát này chưa kiểm tra lại 3.000 bản ghi hoặc ảnh thật. Tài liệu cohort và config hiện ghi nguồn CrisisMMD là phần train chung hai task, dùng nhãn ảnh. Đây là thăm dò trên nguồn train với model frozen; không gọi là test benchmark chính thức. Không ghép với kết quả v1 chỉ vì cùng số mẫu hoặc cùng tên task.

### 3.3. Cần có Accuracy mean/std qua seed

Hàm `seed_summary()` upstream hiện tổng hợp các metric như Macro-F1 và balanced accuracy nhưng danh sách chưa gồm `accuracy`. Bảng tổng hợp nghiên cứu mới phải thêm Accuracy mean/std và giữ cả từng seed, để nối được với giao thức báo cáo của tác giả. Đây là thay đổi báo cáo đầu ra cần thực hiện; chưa sửa trong bước rà soát.

### 3.4. Margin của output gốc khác pipeline fusion

`original_image` một nhãn hiện log softmax của cosine với **scale 1**; pipeline fusion dùng scale model trước softmax. Argmax của ảnh giữ nguyên nhưng confidence/margin có thể rất khác. Không áp ngưỡng margin `0,10` đã khảo sát ở fusion vào output này. Phép thử đầu chỉ so phân loại, chưa dùng cổng phân vân.

### 3.5. Các thành phần phải đóng băng

Đóng băng thứ tự/tên lớp, JSON cấp alphabet, từ điển, template, số cặp, checkpoint, preprocess, cách aggregate và cohort. Mọi prompt sinh phải có trong `frozen_plan.json` trước inference. Không bỏ prompt trùng một cách tự động vì code tác giả có thể lấy lặp; khử trùng sẽ đổi phân bố lấy mẫu.

## 4. Giao thức thử nghiệm giai đoạn đầu

| Mục | Quyết định |
|---|---|
| Dataset | CrisisMMD chính hiện tại; cùng 3.000 cặp cho hai task |
| Target | Nhãn ảnh, đúng cột riêng của informativeness/humanitarian |
| Model | OpenAI CLIP ViT-B/32 qua backend upstream gốc |
| Đầu vào quyết định | Chỉ ảnh; không fusion tweet hoặc quan hệ ảnh–tweet |
| Baseline | `clip`: một template tên lớp theo tác giả |
| Phương pháp | `waffle`: hàm tác giả, 15 cặp/lớp |
| Tên lớp | Giữ bộ tên hiện tại; chưa thêm alias/mô tả mới |
| Concept/descriptor có nghĩa | Chưa thêm vào prompt ở giai đoạn làm mốc |
| Tổng hợp | Mean similarity; không normalize lại prototype trung bình |
| Seed | Baseline xác định chạy một lần/task; Waffle chạy từng seed 0–6/task |
| Metric chính cho dự án | Macro-F1, kèm Accuracy và F1/support từng lớp |
| Đối chiếu | Cùng cohort và bảng sai→đúng/đúng→sai, không chọn seed cao nhất |

Ma trận đầy đủ là **16 lượt**: hai baseline và 14 Waffle. Có thể chạy một lượt GPU 5–10 mẫu để xác nhận cài đặt/checkpoint/log trước; lượt đó chỉ là smoke test và không đưa vào bảng chính. Sau đó chạy từng lệnh riêng. Nếu mới có seed 0, báo đúng là kết quả một seed, chưa coi đó là kết luận về độ ổn định của Waffle.

Kết quả theo cấu trúc đã thống nhất:

```text
results/
  image_only_upstream_clip/crisismmd/<task>/samples_3000/openai_clip_vit_b_32/<run>/
  image_only_upstream_waffle/crisismmd/<task>/samples_3000/openai_clip_vit_b_32/<run>/
```

Mỗi run lưu seed/phương pháp/cấu hình nguồn trong manifest và frozen plan. Tổng hợp bảy seed là phân tích cùng phương pháp, không phải bảy model backbone hoặc bảy tập dữ liệu khác nhau.

## 5. Chỉ sau khi có mốc này mới mở rộng

1. So tên lớp hiện tại với diễn đạt thị giác có nghĩa; có mốc diễn đạt đó chưa thêm noise.
2. So ensemble prompt có nghĩa và Waffle ở cùng ngân sách, tách đối chứng chuẩn hóa prototype.
3. Nếu có lợi ích lặp lại, mở sang MetaCLIP2 và đưa tweet trở lại với fusion cố định.
4. Sau đó mới thử quy tắc giảm nhạy prompt hoặc xử lý phân vân; không ghép các thay đổi vào cùng phép thử đầu.

Tình trạng sau rà soát: **đủ cơ sở để triển khai phép thử làm mốc**, với các việc về profile, cohort và báo cáo seed đã nêu rõ. Điều đó chưa bảo đảm Waffle sẽ tăng chất lượng trên CrisisMMD; đánh giá model thật là bước kế tiếp để trả lời câu hỏi ấy.
