# Đối chứng tách prompt ảnh/text của baseline CrisisMMD

Đã triển khai `prompt_variant: modality_split`. Không viết lại câu, thêm mô tả,
thêm Waffle hoặc thay cách gộp điểm. Baseline `current` tiếp tục dùng toàn bộ
prompt chung; biến thể mới dùng đúng khối ảnh cho nhánh ảnh và khối tweet cho
nhánh văn bản. Thứ tự lớp, nhãn ảnh, checkpoint, scale và fusion giữ theo cấu hình.

Ranh giới prompt là khai báo rõ theo bộ v1, không tìm keyword để tự phân loại:

| Task/lớp | Prompt text | Prompt ảnh |
|---|---:|---:|
| Informativeness / informative | 12 | 15 |
| Informativeness / not_informative | 11 | 13 |
| Humanitarian / mỗi lớp | 6 | 6 |

Phép kiểm tra xác nhận nối `text + image` thu lại chính xác danh sách gốc, không
đổi câu hoặc bỏ câu. Runner mã hóa hai bộ riêng bằng cùng `mean_embed`, truyền
vector ảnh qua `image_class_embeds` hiện có. Audit lưu cả hai bộ prompt và số
lượng. Không cho ghép research image override với modality_split trong cùng
phép thử. Các dataset khác và cấu hình mặc định không tự bật biến thể này.

## Đã chạy

OpenAI CLIP ViT-B/32 thật trên CPU; cùng năm mẫu đầu mỗi task, dùng cả ảnh và
tweet, nhãn đích là nhãn ảnh. Fusion 0,5/0,5, quan hệ tắt, mean_embed:

| Task | Current đúng / 5 | Modality split đúng / 5 |
|---|---:|---:|
| Informativeness | 4 | 4 |
| Humanitarian | 2 | 1 |

Log, audit, điểm từng nhánh, manifest và summary nằm trong
`results/modality_split_smoke/8992cf1b6464/`.
Đây là smoke test kỹ thuật, không phải bằng chứng cải thiện. Mẫu nhỏ không có đủ
tám lớp, không dùng Macro-F1 ở đây để chọn phương pháp. Không tải MetaCLIP2 hoặc
gửi job Kaggle trong lượt triển khai. 90 kiểm thử liên quan đạt.

## Đánh giá MetaCLIP2 trên 3.000 mẫu

Launcher riêng `configs/crisismmd_modality_split.yaml`; không thay YAML chính.
Các job mặc định năm mẫu và phải gọi tường minh. Chạy lần lượt:

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/crisismmd_modality_split.yaml --job split_info --max-samples 3000 --use-kaggle true
venv/bin/python -B scripts/launch_runs.py --launcher-config configs/crisismmd_modality_split.yaml --job split_humanitarian --max-samples 3000 --use-kaggle true
```

Kết quả ở `results/modality_split/crisismmd/<task>/samples_3000/metaclip2_vith14/`.
Hai job `baseline_info` và `baseline_humanitarian` trong cùng YAML dùng current,
cùng cách gộp điểm và fusion. Dùng nếu baseline đã lưu khác đầu vào hoặc giao
thức; không chạy lại nếu đối chiếu được checkpoint, cohort, label source,
mean_embed, scale, quan hệ tắt và fusion 0,5/0,5 hoàn toàn giống nhau.

So Accuracy, Macro-F1 fusion và từng nhánh, số sai→đúng/đúng→sai theo cùng ID;
không so với Waffle lab chỉ ảnh như thể cùng đầu vào. Không bảo đảm tách modality
sẽ tăng điểm. Nếu không cải thiện, giữ baseline và ghi kết quả âm thay vì chỉnh
câu liên tục theo nhãn của cohort train đã khảo sát.
