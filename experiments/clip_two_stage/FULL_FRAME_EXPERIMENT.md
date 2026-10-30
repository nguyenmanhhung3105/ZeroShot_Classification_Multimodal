# Kiểm tra còn khả năng cải thiện: giữ thông tin toàn khung ảnh

## Sàng lọc trước khi dùng thêm GPU

Đã dùng các score/norm prototype lưu sẵn trên train 3.000 mẫu để thử các đối chứng
không đổi prompt. Không đọc dataset hoặc chạy model mới cho các phép dưới đây.
Giữ cùng ảnh, thứ tự nhãn và gate semantic informativeness:

| Seed 0, hai tầng | Accuracy | Macro-F1 |
|---|---:|---:|
| Mean cosine gốc | 68,03 | 49,88 |
| Chỉ chuẩn hóa lại prototype tầng 1 | 69,40 | 49,36 |
| Chỉ chuẩn hóa lại prototype tầng 2 | 67,33 | 47,76 |
| Chuẩn hóa lại cả hai tầng | 68,73 | 47,67 |

Cả ba seed đều giảm Macro-F1 khi đổi chuẩn hóa theo các cách trên. Trung bình
score tầng 2 qua seed 0/1/2, rồi chạy gate gốc, đạt Accuracy 68,40%, Macro-F1
50,01%; trung bình metric ba seed riêng là 50,05%. Không có lợi ích rõ để mặc định
thêm chi phí ensemble nhiều seed. Đây là kết quả phát triển, không phải test.

Đã rà Frolic (NeurIPS 2024), hướng học phân phối prompt/điều chỉnh bias không
nhãn: https://arxiv.org/abs/2410.19294 và https://github.com/zhuhsingyuu/Frolic .
Chưa tái lập Frolic hoặc nhận các phép chuẩn hóa trên là thuật toán Frolic.
Không đưa số tăng trên benchmark của paper thành kỳ vọng CrisisMMD.

## Hướng mới được triển khai để kiểm tra

CLIP preprocess có thể cắt phần ngoài tâm ảnh. Với ảnh cảnh, bản đồ hoặc thông
báo có chữ, thông tin ngoại vi có thể hữu ích. Cần kiểm tra bằng đối chứng thay
vì mặc định mọi ảnh đều bị mất phần quan trọng. Giữ nguyên prompt/gate/seed:

1. Preprocess gốc (baseline).
2. Toàn khung: pad ảnh RGB thành vuông bằng nền đen, giữ nguyên pixel và tỷ lệ,
   rồi đi qua chính preprocess của checkpoint, không kéo giãn ảnh thành vuông.
3. Trung bình cosine của hai view ở từng tầng, rồi áp cùng quy tắc hai tầng.

Nguồn liên quan đến hướng nhiều góc nhìn cho CLIP: MTA (CVPR 2024),
https://arxiv.org/abs/2405.02266 . MTA dùng MeanShift/filter nhiều view;
phép thử hai view cố định này **không phải tái lập MTA**, không dùng tên MTA cho
phương pháp hiện tại, cũng không tuyên bố multi-view là ý tưởng mới.
TPT https://arxiv.org/abs/2209.07511 có tối ưu prompt qua nhiều view; runner này
không backprop hoặc cập nhật prompt/model như TPT.

Mỗi ảnh encode hai lần bằng cùng một model. Prompt encode một lần mỗi bank,
score hai view tính chung. Không chọn view theo nhãn thật. Encode từng batch
view lần lượt để giữ peak batch GPU như trước; thời gian image encoding gần
gấp đôi. Pad đen có thể tạo nhiễu và làm đối tượng nhỏ hơn, chưa bảo đảm tăng điểm.

Chưa ghép thêm text vào phép thử này để tránh thay nhiều yếu tố cùng lúc. Những
runner multimodal đã có vẫn giữ nguyên mặc định và kết quả cũ.

## Kiểm tra và chạy

77 kiểm thử liên quan đạt, gồm bảo toàn pixel/aspect ratio, hai view giống nhau
không đổi quyết định, và launcher giữ cờ/output riêng. Smoke OpenAI CLIP thật trên
CPU với năm ảnh dev: gốc đúng 4/5, full frame 3/5, mean view 4/5. Đây chưa là
bằng chứng cải thiện. Log:
`results/clip_two_stage_views_heldout/crisismmd/humanitarian/samples_5/openai_clip_vit_b_32/humanitarian_48fa528d5669/`.
Analyzer tái tạo đúng các dự đoán từ score lưu trong artifact.

Một lượt MetaCLIP2 trên toàn bộ dev 2.219 ảnh, chưa tự gửi:

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/clip_two_stage/launcher_dev_views.yaml --job two_stage_views_dev --seed 0 --use-kaggle true
```

Không thêm --max-samples 3000. Cả baseline và hai đối chứng view mới đều xuất trong
cùng summary; không cần gửi nhiều job. Output:
`results/clip_two_stage_views_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/`.

Nếu không tăng Macro-F1 đủ rõ và không có ích cho lớp khó, dừng hướng này; không
tiếp tục chọn nhiều crop/weight theo điểm dev rồi gọi đó là test độc lập. Nếu có
lợi ích, khóa quy tắc và xác nhận trên test chưa dùng để chọn phương pháp.
