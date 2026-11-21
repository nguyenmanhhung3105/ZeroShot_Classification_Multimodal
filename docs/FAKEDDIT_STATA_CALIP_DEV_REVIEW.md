# Kiểm tra StatA → CALIP Fakeddit dev

Run: `results/final/kaggle/fakeddit_stata_calip_dev_85c4c3d20e7e/run/`.
Version: `nguyenhung3105/vlm-fakeddit-stata-calip-dev-v1/1`.

## Kết quả 2way tầng đầu

| Phương pháp | Accuracy | Macro-F1 |
|---|---:|---:|
| CLIP | 62,47% | 62,46% |
| StatA (3 seed) | 65,88 ± 0,68% | 65,87 ± 0,69% |

Gate StatA khớp chính xác dự đoán lưu từ run hierarchy trước, cả ba seed.
CALIP không thay gate. Mọi ảnh gate real đều đi về True.

## Kết quả 6way

| Tầng hai sau cùng gate StatA | Accuracy | Macro-F1 |
|---|---:|---:|
| CLIP, ảnh | 50,36% | 31,34% |
| CALIP, ảnh | 50,01% | 31,08% |
| Fusion CLIP/CALIP, ảnh | 50,20% | 31,26% |
| CLIP, ảnh–tiêu đề | 50,27% | 31,57% |
| CALIP, ảnh–tiêu đề | 50,16% | **31,74%** |
| Fusion CLIP/CALIP, ảnh–tiêu đề | 50,09% | 31,41% |

CALIP image-only giảm cả accuracy và macro-F1 so với CLIP cùng gate. Với MM,
CALIP tăng macro-F1 0,17 điểm phần trăm nhưng giảm accuracy 0,11 điểm phần trăm;
mỗi seed sửa đúng 5 ảnh, làm sai 8–9 ảnh so với CLIP MM. Fusion đều không giúp.

Macro-F1 từng lớp CLIP MM → CALIP MM (trung bình seed): True 65,28→65,28;
Satire 13,52→14,45; False Connection 33,51→33,62; Imposter 3,11→3,03;
Manipulated 66,35→65,75; Misleading 7,65→8,33. Trade-off giữa lớp giải thích việc
accuracy giảm nhưng macro-F1 tăng nhẹ; không phải mọi loại fake đều tốt hơn.

## Bất định của chênh lệch

Paired bootstrap theo ID mẫu, dùng cùng bộ ID resample cho cả ba grouping seed,
5.000 lần, seed 0. Delta macro-F1 CALIP MM − CLIP MM = +0,1736 điểm phần trăm;
CI percentile 95% **[−0,2247; +0,6847] điểm phần trăm**. Ba grouping seed cố định;
không gộp thành 9.000 mẫu độc lập. Kết quả dev chưa có xác nhận trên test.
CI chứa 0, chưa có bằng chứng rõ CALIP cải thiện macro-F1 trong cấu hình này.
Artifact: `results/final/kaggle/fakeddit_stata_calip_dev_85c4c3d20e7e/paired_macro_f1_analysis.json`.

## Xác minh kỹ thuật

- Completion/run và mọi checksum hợp lệ, đủ 24 method/seed, mỗi tổ hợp 3.000
  ID duy nhất, cùng ID/nhãn. Cohort khớp run trước.
- Replay toàn bộ 18 dự đoán hierarchy từ gate/posterior lưu khớp chính xác.
- Posterior CALIP và fusion tái tính khớp. CALIP term replay từ patch của tối đa
  3 ảnh đầu khớp trong FP32; sai số tuyệt đối tối đa 4,58e−5.
- 375 batch, grid 16×16, 256 patch, chiều embedding 1024. Global/reference đầu
  tiên sai số 0; output_tokens khôi phục ở mọi batch.
- CALIP tham số tham chiếu cố định, temperature 1, fusion 0,5/0,5; chưa calibration.
- Timer runner 633,52 giây (~10 phút 34 giây), không gồm queue hoặc phần đọc
  cohort trước timer. Không chạy thêm GPU/test trong review này.

Các MM trong run này dùng image prototype unit, không giống hệt legacy raw-mean
MM. Chỉ quy đóng góp CALIP theo baseline CLIP cùng gate/modality trong run mới.
Kết luận: giữ CALIP như ablation; chưa thay CLIP bằng CALIP làm phương pháp chính
chỉ dựa trên mức tăng dev nhỏ này. Router real/fake và các lớp fake hiếm vẫn là
điểm cần xử lý, không tiếp tục chọn trọng số/temperature bằng test.
