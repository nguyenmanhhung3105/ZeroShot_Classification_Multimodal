# Kiểm tra Fakeddit dev 6way — bản version 2

Run đã tải: `results/final/kaggle/fakeddit_hierarchy_dev_85d0ba88b624/run/`.
Kaggle receipt: `nguyenhung3105/vlm-fakeddit-hierarchy-dev-v1/2`.
Đây không phải xác nhận version 3: có receipt submitted khác cho version 3,
chưa có artifact local tương ứng; truy vấn status version 3 trả 404 trong lượt review.

## Kết quả image-only (đối chứng cùng modality)

| Thành phần | Flat accuracy | Hierarchy accuracy | Flat macro-F1 | Hierarchy macro-F1 |
|---|---:|---:|---:|---:|
| CLIP | 35,20% | 48,70% | 25,56% | 30,48% |
| EM | 31,93% | 50,29% | 26,79% | 30,26% |
| StatA | 39,63% | 50,36% | 29,27% | 31,34% |
| Majority | 36,83% | 50,03% | 26,94% | 30,78% |

EM/StatA/Majority là trung bình ba grouping seed 0/1/2, không chọn seed tốt nhất.
Hai tầng tốt hơn flat tương ứng về cả accuracy và macro-F1 trong run dev này.
Khác kết quả CIFAR-100; không dùng kết luận CIFAR để phủ định cấu trúc binary gốc.

## Nhánh multimodal legacy

| Phương pháp | Accuracy | Macro-F1 |
|---|---:|---:|
| Legacy flat MM | 35,60% | 26,24% |
| Legacy hierarchy gate CLIP MM | 49,03% | 31,09% |
| Original hierarchy Majority MM | 50,33 ± 0,18% | 31,37 ± 0,14% |

Original tăng 14,73 điểm phần trăm accuracy và 5,13 điểm phần trăm macro-F1 so
với legacy flat MM. Tuy nhiên, phần thêm Majority so gate CLIP chỉ tăng 1,30
điểm accuracy và 0,28 điểm macro-F1. Không quy toàn bộ lợi ích phân cấp cho EM/StatA.

## Chất lượng tuyệt đối và chẩn đoán

Cohort có số mẫu theo ID 0..5: 1537/204/602/92/407/158. Baseline luôn dự đoán
True đạt accuracy 51,23%, macro-F1 11,29%. Vì vậy, accuracy của hệ thống hiện còn
thấp hơn baseline lớp đông nhất dù macro-F1 tốt hơn rõ; không tuyên bố hệ thống
đã mạnh chỉ vì vượt CLIP flat hiện có.

Gate original đạt binary accuracy 63,70/63,33/63,53% theo seed. Oracle gate MM
dùng nhãn binary thật đạt accuracy 79,40%, macro-F1 45,02%; đây chỉ là giới hạn
chẩn đoán, không phải phương pháp thực tế. Một số loại fake vẫn có recall thấp
ngay cả oracle, nên cần xử lý cả router và khả năng phân biệt loại fake.

## Xác minh

- Completion/run và mọi output_sha256 hợp lệ.
- Đúng 25 method/seed; mỗi tổ hợp đủ 3.000 ID duy nhất, cùng ID/nhãn thật.
- Tái dựng posterior image/MM từ feature và raw prototype. Sai số FP32 giữa
  Kaggle/local tối đa khoảng 1,22e−5; argmax khớp hoàn toàn.
- Tái dựng tất cả prediction EM/StatA/Majority/hierarchy từ phiếu đã lưu, cả ba
  seed khớp chính xác; không chạy lại solver để kiểm tra artifact.
- Timer runner 859,83 giây (~14 phút 20 giây); không coi là tổng thời gian queue.

Kết quả trên dev đã dùng phát triển; chưa có xác nhận độc lập trên test hoặc
CI paired cho macro-F1. Chưa chạy CALIP. Không chỉnh tham số, không sửa run hoặc
tự gửi GPU job trong lượt review này.
