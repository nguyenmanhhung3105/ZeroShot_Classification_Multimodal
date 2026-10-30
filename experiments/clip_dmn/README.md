# DMN-ZS single-view và kết hợp mềm hai tầng

Đợt nghiên cứu ngày 29-10-2026 đã hoàn tất trên embedding train 3.000 và
dev 2.219 ảnh đã lưu, backbone MetaCLIP2 ViT-H/14. Không chạy encoder mới,
không truy cập raw/processed, không gửi Kaggle, không mở test. Các phép thử
chỉ dùng ảnh; text embedding ở đây là embedding của prompt tên/mô tả lớp.

## Cơ sở và phạm vi triển khai

[DMN, CVPR 2024](https://github.com/YBZh/DMN) dùng bộ nhớ ảnh động để điều
chỉnh vector đại diện lớp theo ảnh đang xét. Khóa source tác giả ở commit
`c6c218c70ebbc701b31cc3d7f7b9597117daad8e`; đối chiếu các hàm
`DualMem.update_memory_bank`, `get_text_prediction`, `get_image_pred`.

Bản zero-shot ở đây không có support gán nhãn hoặc projection được học.
Mỗi ảnh được gán nhãn giả bằng text anchor; mỗi lớp giữ tối đa 50 ảnh có
entropy thấp nhất. Memory cập nhật **trước** dự đoán, kể cả ảnh hiện tại.
Đọc memory bằng trọng số `exp(-5.5 * (1 - cosine))`, kèm anchor của lớp,
rồi tạo classifier thích nghi cho ảnh. Trộn cố định 0,5 xác suất anchor
và 0,5 xác suất dynamic; không tìm trọng số bằng nhãn thật.

Đây là **DMN-ZS single-view adaptation**, không phải tái lập benchmark
DMN đầy đủ: script zero-shot tác giả dùng 32 view, còn artifact hiện có
chỉ chứa một view thật mỗi ảnh. Chọn tối thiểu một view thay vì chọn zero
khi `floor(V * 0.1) = 0`. Code nhận được `[N,V,D]` nếu sau này có augmented
feature thật, nhưng không coi những ảnh khác nhau là augmentation.
Phần tìm trọng số bằng nhãn trong chương trình đánh giá gốc không được dùng.

Memory reset riêng theo task, split và thứ tự. Báo cáo trung bình ba hoán
vị seed 0/1/2, giữ `source_order` chỉ làm stress test và loại khỏi mean.
Đây là kiểm tra thứ tự stream, không phải ba lần train model độc lập.

## Các mở rộng và đối chứng

1. **DMN text-anchor:** scale 100, thử memory ở root, leaf và cả hai tầng;
   đối chiếu root majority/InMaP và leaf DN-unit sẵn có.
2. **Visual-anchor DMN:** thay anchor bằng proxy InMaP không Sinkhorn ở
   root/leaf, scale cố định 25 = `1 / tau_i` của InMaP. Đây là mở rộng của
   dự án; không gọi là benchmark DMN gốc. InMaP fit trên toàn cohort không
   nhãn nên toàn pipeline là transductive, dù riêng memory cập nhật tuần tự.
3. **Parent product:** lấy cảm hứng từ bước reweight theo nhãn cha của
   [CHiLS, ICML 2023](https://proceedings.mlr.press/v202/novack23a.html).
   Thay cổng cứng bằng
   `score_j = softmax(100 * DN_cosine)_j * P_root(parent(j))`, rồi argmax
   trên tám lớp. Map bảy lớp liên quan về informative, not_humanitarian về
   not_informative. Đây là giả định kết hợp đầu ra của model; hai task có
   nhãn thật riêng, không suy nhãn binary GT từ humanitarian GT. Giữ đầu ra
   leaf khác CHiLS gốc; không sinh thêm tên lớp, không tái lập CHiLS đầy đủ.
   Các confidence chưa hiệu chuẩn, head tám lớp không phải phân phối điều
   kiện bảy lớp; không gọi phép nhân này là chain rule xác suất đã hiệu chuẩn.
4. **Ablation bỏ memory:** giữ đúng parent product và scale, dùng xác suất
   InMaP ban đầu thay root đã trộn DMN. Không đổi weight để săn metric.

## Kết quả

Đơn vị %. Human là Macro-F1 tám lớp. Binary dev không có GT thật nên không
báo điểm. Mean của ba shuffle với DMN; InMaP không memory là deterministic,
ba file seed lặp lại cùng dự đoán và không phải ba phép thử độc lập.

| Phương án | Binary train | Human train | Human dev | Recall cứu trợ dev |
|---|---:|---:|---:|---:|
| Hai tầng gốc | 79,58 | 49,88 | 48,06 | 76,07 |
| Majority + DN-unit | 80,83 | 49,22 | 50,39 | 60,00 |
| InMaP gate + DN-unit | 81,75 | 50,36 | 49,75 | 70,36 |
| DMN text gate + DN-unit | 80,00 | 46,76 | 47,89 | 46,67 |
| DMN text ở cả hai tầng | 80,00 | 40,87 | 43,08 | — |
| Visual DMN gate + DN-unit | 82,34 | 51,03 | 49,92 | 55,95 |
| Visual DMN + DN parent product | 78,08 | 48,41 | 50,04 | 79,29 |
| **InMaP + DN parent product, bỏ DMN** | **78,18** | **48,23** | **50,01** | **80,00** |

DMN text-anchor không cải thiện cấu hình đã thử. Kiểm tra cache sau khi
dự đoán thấy lớp missing_or_found_people chứa 50 ảnh nhưng không có ảnh
đúng trên cả train/dev. Nhãn thật chỉ dùng chẩn đoán sau dự đoán; tuyệt đối
không dùng để lọc memory. Điều này gợi ý nhãn giả tự tin sai là một vấn đề,
không chứng minh rằng bản DMN nhiều view đầy đủ sẽ thất bại.

Visual DMN cải thiện binary train nhưng làm recall cứu trợ dev giảm còn
55,95%. Parent product phục hồi recall, đồng thời làm binary/train giảm.
Chênh Human dev có/bỏ DMN chỉ khoảng **0,03 điểm**, chưa có cơ sở cho rằng
memory đem lại lợi ích thực tế. InMaP + DN parent product là ứng viên khám
phá ưu tiên recall, chưa thay thế hai ứng viên đang giữ. Các số là kết quả
khám phá trên train/dev đã dùng nhiều lần, chưa là kết luận trên test độc lập.

## Artifact và kiểm chứng

Trong `results/analysis/`:

- `clip_dmn/00a26641fb2b`: text-anchor, đủ năm nhánh và majority DN control.
- `clip_dmn_visual/e1523f5e8972`: visual-anchor, ba nhánh.
- `clip_dmn_hierarchy/92509a7e0a76`: parent product và hard-gate control.
- `clip_inmap_hierarchy/d4fd47b09f43`: ablation không memory.

Mỗi thư mục có manifest, hash đầu vào/code/đầu ra, dự đoán theo ID,
`analysis.json` và `audit.json` trạng thái passed. Audit kiểm tra lại
prediction/metric từng lớp, thứ tự/nhãn thật, reset/cache write/replace,
mean loại source_order, và tính tích confidence bằng vòng lặp độc lập.

Sáu unittest đạt, gồm nhân quả theo prefix với anchor cố định, entropy
cache, reset, multiview tổng hợp, runner tích hợp giả và parent product.
Reference verification trích đúng ba method tác giả, không import dataset
hoặc main: bốn ca tổng hợp và 24 stream task thật (12 text, 12 visual).
Posterior, argmax, final memory đều khớp; sai số posterior tối đa 0 ở các
stream thật. Việc khớp operator không đồng nghĩa tái lập benchmark nhiều view.
Xem các file `reference_verification*.json` trong thư mục này.

Không đổi pipeline mặc định hoặc frozen final plan DN-unit v1. Muốn chọn
một phương án mới cho test phải khóa protocol mới trước khi đo test.

## Lệnh tái chạy trên artifact

Các lượt trên **đã chạy xong**; không cần gửi job hoặc chạy lại để lấy kết
quả. Các lệnh này chỉ nhằm tái lập, chạy tuần tự tại thư mục gốc:

```bash
venv/bin/python -B -m unittest experiments.clip_dmn.test_dmn -b -q
venv/bin/python -B experiments/clip_dmn/run.py --config experiments/clip_dmn/config.json
venv/bin/python -B experiments/clip_dmn/run_visual.py --config experiments/clip_dmn/config_visual.json
venv/bin/python -B experiments/clip_dmn/hierarchy.py --config experiments/clip_dmn/config_hierarchy.json
venv/bin/python -B experiments/clip_dmn/ablate_memory.py \
  --visual-run results/analysis/clip_dmn_visual/e1523f5e8972 \
  --dn-run results/analysis/clip_dn_unit/d70959a0a21c
```

Các config downstream khóa artifact gốc để bảo đảm đối chứng; nếu cần nối
một lượt mới thì tạo config riêng trỏ đúng output mới, không sửa kết quả cũ.
