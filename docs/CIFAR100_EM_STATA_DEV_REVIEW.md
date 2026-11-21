# Kết quả EM/StatA CIFAR-100 dev — 10/11/2026

Run: `results/final/kaggle/cifar100_em_stata_dev_c3d25f6a4d9c/run/`.
Version: `nguyenhung3105/vlm-cifar100-em-stata-dev-v1/1`.

| Phương pháp | Flat accuracy | Hai tầng accuracy |
|---|---:|---:|
| CLIP | 86,75% | 58,10% |
| EM | 64,66 ± 0,18% | 46,25 ± 0,70% |
| StatA | 86,01 ± 0,03% | 52,19 ± 0,20% |
| Majority | 86,44 ± 0,02% | 54,20 ± 0,10% |

CLIP tất định một lượt; các nhánh adaptation ba grouping seed 0/1/2, sample std.
Không có nhánh EM/StatA nào vượt flat CLIP; không có hierarchy nào vượt flat
tương ứng. Flat mô tả 88,38% và equal fusion 88,87% từ các lượt dev trước vẫn
là đối chứng mạnh hơn; không gọi những con số dev này là xác nhận trên test.

## Kiểm tra artifact

- Completion/run và tất cả artifact output_sha256 khớp.
- Đúng 20 tổ hợp method/seed, mỗi tổ hợp 10.000 ID duy nhất và cùng cohort.
- Mỗi seed có 134 nhóm query (133 nhóm 75, nhóm cuối 25), cùng group IDs dùng
  cho flat 100 lớp và coarse 20 lớp. Solver metadata có 268 bản ghi/seed vì hai level.
- Timer runner 7.539,44 giây, khoảng 2 giờ 05 phút 39 giây sau khởi tạo model;
  không phải toàn bộ thời gian job hoặc queue.

## Chẩn đoán

Nhánh coarse CLIP đạt 63,44%. EM coarse đạt 50,61/49,99/51,41% theo ba seed;
StatA 57,21/57,50/57,57%; Majority 59,42/59,24/59,52%. Adaptation chưa sửa được
router yếu, còn làm giảm độ chính xác nhóm. Hard gate loại nhóm đúng thì tầng
hai không thể cứu, dù fine logits còn giữ đúng lớp.

Trong nhóm 75 ảnh, trung bình có 52,85–53,33 lớp thật khác nhau. EM chỉ dự đoán
37,02–37,33 lớp khác nhau/nhóm, so với CLIP 52,53–52,74 và StatA 51,69–51,97.
Điều này hỗ trợ quan sát EM gom nhiều ảnh vào ít lớp hơn; chưa tách nguyên nhân
giữa regularization, quy mô nhóm và phân phối lớp bằng ablation có kiểm soát.
Không được kết luận đơn giản rằng tăng nhóm sẽ chắc chắn sửa được chất lượng.

Đây là thích nghi Majority binary của Fakeddit sang 20 nhóm CIFAR, không phải
kiểm chứng nguyên xi cấu trúc real/fake → năm loại fake. Kết quả không chứng minh
mô hình gốc trên Fakeddit kém trong chính bài toán binary đó. CALIP chưa được chạy.

## Hướng tiếp theo

Không ghép thêm nhánh rồi coi bỏ phiếu sẽ tự cải thiện. Cần xử lý riêng router
coarse yếu và kiểm tra giả định adaptation trên nhóm nhiều lớp/ít ảnh mỗi lớp.
Feature/prototype/logits đã lưu nên phần lớn ablation tiếp theo có thể replay,
không cần mã hóa lại ảnh. Giữ test chưa mở và dữ liệu cũ làm baseline.
