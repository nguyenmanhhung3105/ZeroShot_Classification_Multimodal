# CLIP hai tầng với text hỗ trợ: đối chứng dev đã khóa

Đầu vào ảnh + tweet, nhãn đích vẫn là label_image humanitarian. Model MetaCLIP2
duy nhất, cùng cohort dev 2.219 ảnh, seed 0 và prompt ảnh semantic/semantic_waffle
đã chọn. Không đổi prompt/luồng mặc định của các lượt chỉ ảnh đã có.

## Thiết kế

- Gate tầng 1 giữ nguyên quyết định dựa trên ảnh. Tweet không được tự biến ảnh
  không informative thành ảnh informative trong hai phương án hai tầng mới.
- Nhánh text dùng đúng khối tweet của baseline, tách bằng get_branch_prompt_sets;
  cùng thứ tự lớp. Prompt mean_embed, text sample encode một lần cho hai task.
- Tầng 2 fixed: 0,75 softmax(scale * image cosine) + 0,25 softmax(scale * text cosine).
  Scale lấy từ checkpoint, không điều chỉnh theo nhãn dev. Quyết định ở bảy nhóm
  positive nếu gate ảnh positive; giữ not_humanitarian nếu gate ảnh negative.
- Tầng 2 conditional: chỉ dùng 25% text nếu gate ảnh và gate text đều informative
  và text không rỗng. Còn lại giữ điểm ảnh. Đồng thuận relevance không chứng minh
  tweet và ảnh nói về cùng sự kiện hoặc đúng cùng nội dung.
- Text rỗng: fusion dùng ảnh; đối chứng text-only ghi __missing_text__ và tính
  sai, không bỏ mẫu. Không dùng label_text, label_image hoặc y_true để đặt weight.
- Tweet quá context được model cắt theo cơ chế cũ, có log token trước cắt và số
  mẫu bị cắt. Prompt lớp quá context bị từ chối. Chưa thêm chunking vì sẽ đổi thêm
  một yếu tố trong cùng phép thử. Softmax chưa được hiệu chuẩn xác suất.

Trong một lượt xuất bảy hàng: single_stage, two_stage, margin_pool (ba đối chứng
ảnh đã có), text_only, single_stage_fixed (ảnh–text một tầng), two_stage_fixed,
two_stage_conditional. Không cần chạy bảy job riêng. Text-only cũng được chấm
theo nhãn ảnh để đo sự hữu ích cho đúng target hiện tại, không coi là đánh giá
theo nhãn text gốc của CrisisMMD.

## Kiểm tra đã thực hiện

Kiểm thử tổng hợp cho thấy text có thể đổi nhãn chi tiết khi được phép; text
không informative bị cổng conditional từ chối; text rỗng fallback ảnh; reader
đọc đúng prefix và từ chối sai ID/thứ tự. Launcher giữ cờ multimodal và đường
dẫn riêng. Analyzer tái dựng các phương án từ score đã lưu, kiểm tra fingerprint
text plan/cohort và trọng số. CI mặc định của analyzer vẫn chỉ so image one-stage
với image two-stage; không dùng CI đó để kết luận lợi ích multimodal.

Đã chạy model OpenAI CLIP ViT-B/32 thật trên CPU với năm ảnh/tweet đầu dev:
image two-stage, fixed multimodal và conditional đều đúng 4/5; chỉ text đúng 2/5.
Đây là kiểm tra luồng chạy, không chứng minh tăng chất lượng. Log:
`results/clip_two_stage_mm_heldout/crisismmd/humanitarian/samples_5/openai_clip_vit_b_32/humanitarian_7bad9cc9b0a9/`.
Không cắt token ở năm tweet này. Chưa chạy MetaCLIP2 multimodal trên dev lớn.

## Chạy toàn dev trên Kaggle

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/clip_two_stage/launcher_dev_multimodal.yaml --job two_stage_mm_dev --seed 0 --use-kaggle true
```

Không thêm --max-samples 3000. Mặc định chính xác 2.219 mẫu. Có thể dùng
--max-samples 5 cho smoke T4. Đầu ra nằm trong
`results/clip_two_stage_mm_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/`.
Đây vẫn là DEV phục vụ phát triển, chưa gọi là test cuối, dù tên thư mục heldout
đánh dấu tách khỏi train. Runner giữ seed, prompt và weight cố định.

```sh
venv/bin/python -B experiments/clip_two_stage/analyze.py --run-dir PATH_KET_QUA_MM
```

Cần so cùng seed/cohort giữa image two-stage, fixed và conditional, báo Macro-F1,
Accuracy, fixed/broken khi thêm text, số mẫu được dùng text và từng lớp. Nếu text
không giúp thì không nhận là đã cải thiện đa phương thức. Sau khi chốt trên dev,
chuẩn bị test tách khỏi cả train/dev để xác nhận, không chỉnh tiếp trên test.
