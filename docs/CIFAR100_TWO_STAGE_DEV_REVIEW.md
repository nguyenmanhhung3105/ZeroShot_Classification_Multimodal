# Kiểm tra full dev CIFAR-100 — 10/11/2026

Run: `results/final/kaggle/cifar100_two_stage_dev_4cef907deba3/run/`.
Kaggle: `nguyenhung3105/vlm-cifar100-two-stage-dev-v1/1`.

## Kết quả

| Phương pháp | Accuracy | Macro-F1 | Balanced accuracy | Top-5 |
|---|---:|---:|---:|---:|
| Flat | 86,75% | 86,44% | 86,88% | 97,04% |
| Hai tầng cứng | 58,10% | 56,77% | 58,45% | 63,44% |
| Hai tầng mềm | 67,19% | 66,24% | 67,66% | 89,40% |

Hai tầng cứng giảm 28,65 điểm phần trăm accuracy; sửa đúng 145 mẫu nhưng làm sai
3.010 mẫu so với flat. Hai tầng mềm giảm 19,56 điểm phần trăm; sửa đúng 134 mẫu
nhưng làm sai 2.090 mẫu. Mềm tốt hơn cứng 9,09 điểm phần trăm, vẫn kém flat.

## Nguyên nhân được dữ liệu hỗ trợ

Nhánh dự đoán trực tiếp 20 nhóm chỉ đạt 63,44% accuracy, chọn sai nhóm ở 3.656 mẫu.
Nhóm suy từ dự đoán fine của flat đạt 93,24%. Hard gate không thể dự đoán đúng
fine khi chọn sai nhóm. Soft vẫn đưa xác suất coarse yếu vào phân phối cuối.
Đây là bằng chứng nhánh coarse hiện tại yếu; không kết luận mọi kiến trúc phân
cấp đều kém. Chưa xác định riêng phần lỗi do prompt nhóm, scale hay quan hệ giữa
taxonomy và biểu diễn CLIP bằng các ablation độc lập.

Chẩn đoán offline bằng nhóm thật, chỉ xét fine logits trong nhóm thật, đạt 91,25%.
Đây là oracle dùng ground truth, không phải kết quả hệ thống triển khai. Nó cho
thấy tiềm năng nếu định tuyến tốt hơn, không chứng minh router thực tế đạt được.
Không sửa cấu hình, không chọn tham số hoặc mở test trong đợt kiểm tra này.

## Xác minh

- Kaggle status COMPLETE; completion/run checksum và toàn bộ artifact trong
  output_sha256 khớp bằng `completed_run`.
- Mỗi phương pháp đủ 10.000 dự đoán, 10.000 ID duy nhất; cùng ID và cùng nhãn.
- Tính lại flat/hard/soft từ logits trong `replay_scores.npz` (allow_pickle=False),
  xác suất khớp file lưu và accuracy khớp bảng.
- MetaCLIP2 ViT-H/14 worldwide, CUDA FP32, batch ảnh 8, batch prompt 64; seed
  chia dev 0, một lượt xác định, không fine-tune.

## Thời gian

Log kết thúc ở 1.727,75 giây, khoảng **28 phút 48 giây**, gồm setup/model và xuất
kết quả, không gồm thời gian chờ queue. Timer runner ghi 1.581,30 giây, khoảng
**26 phút 21 giây**, bắt đầu sau khi model được khởi tạo. Timeout 14.400 giây chỉ
là giới hạn job, không phải thời gian chạy thực tế.

## Bước tiếp theo

Giữ run này làm đối chứng phát triển. Ưu tiên ablation router/prompt coarse trên
dev và kiểm tra calibration trước khi đánh giá test. Không chỉ cộng xác suất
fine theo nhóm rồi nhân lại conditional và tuyên bố cải tiến: phép đó trả lại
đúng phân phối flat khi dùng cùng logits. Cần bằng chứng bổ sung hoặc cơ chế mới
có thể sửa lỗi fine, cùng các đối chứng tách lợi ích prompt và định tuyến.

Đây là kết quả dev đã được xem để phát triển, chưa phải xác nhận độc lập trên test.
