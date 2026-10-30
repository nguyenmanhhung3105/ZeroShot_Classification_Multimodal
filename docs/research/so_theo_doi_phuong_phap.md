# Sổ theo dõi các hướng CLIP đang thử

Ngày cập nhật: 29-10-2026. Tiếp tục nghiên cứu theo yêu cầu người dùng;
không chỉ thay prompt. Ưu tiên phép thử không nhãn dùng cached embedding,
không đọc raw/processed vượt giới hạn AGENTS.md và không tự gửi Kaggle.
Giữ few-shot ở hướng dự phòng, chưa kích hoạt support ground truth.

## Các đợt mới hoàn tất

| Hướng | Mục đích | Kết quả | Quyết định |
|---|---|---|---|
| ZLaP, CVPR 2024 | Truyền nhãn qua đồ thị ảnh và text | Cả hai tầng dev Macro-F1 41,78% | Không chọn cấu hình đã thử |
| DN-unit từ DN, NeurIPS 2023 | Hiệu chỉnh embedding bằng mean không nhãn | Majority + DN-unit dev 50,39%, train 49,22% | Giữ ứng viên theo Macro-F1 dev |
| InMaP, NeurIPS 2023, không Sinkhorn | Học proxy ảnh bằng nhãn giả, giữ CLIP frozen | Binary train 81,75%; root-only human train/dev 50,67/48,64% | Giữ cổng làm ứng viên |
| InMaP, Sinkhorn prior đều | Kiểm tra ảnh hưởng giả định cân bằng | Hai tầng dev 42,40% | Không chọn; không mặc định ép cân bằng lớp |
| InMaP gate + DN-unit leaf | Ghép cổng tốt hơn trên binary với head DN | Train/dev 50,36/49,75%; recall cứu trợ dev 70,36% | Giữ ứng viên cân nhắc cùng majority + DN |
| LAME, CVPR 2022, RBF/kNN | Hiệu chỉnh đầu ra bằng lân cận ảnh, không cập nhật proxy | kNN leaf-only dev 47,11%; các nhánh khác thấp hơn | Không chọn hai cấu hình nhóm 75 đã thử |
| DMN-ZS single-view từ CVPR 2024 | Memory nhãn giả entropy thấp, classifier theo từng ảnh | DMN root + DN dev 47,89%; DMN cả hai tầng 43,08% | Không chọn; chưa tái lập bản nhiều view |
| Visual-anchor DMN + DN-unit | Anchor InMaP thay text, memory ở root | Binary train 82,34%; human train/dev 51,03/49,92%; recall cứu trợ dev 55,95% | Chưa chọn vì giảm recall |
| Parent product từ CHiLS, ICML 2023 | Kết hợp confidence cha/con thay cổng cứng | Có DMN dev 50,04%/recall 79,29%; bỏ DMN 50,01%/80,00% | Giữ bản không memory làm hướng ưu tiên recall; binary/train giảm |
| DPE single-view, NeurIPS 2024 | Residual visual/text, entropy + alignment, cache và online averaging | Binary train 80,58%; majority + DPE leaf train/dev 48,81/49,92% | Chưa vượt DN-unit với cùng cổng |
| DPE + DN confidence matching từ Frolic | Ghép hai head và cân bằng mean confidence không nhãn | Majority train/dev 49,04/50,29%; InMaP 50,30/49,71% | Không chọn; DN-unit cùng cổng vẫn cao hơn |
| CALIP global/local, AAAI 2023 | Attention hai chiều giữa patch và prompt, giữ điểm global | Majority + CALIP train/dev 47,63/51,14%; so với DN 49,22/50,39%; binary CALIP root 80,31% train | Giữ ứng viên Macro-F1 dev; train/Accuracy dev giảm, CI chứa 0 |

Không so sánh train với dev như thể cùng dataset, không chọn seed tốt nhất,
không lấy mức tăng trên ImageNet làm phần trăm kỳ vọng cho CrisisMMD.
Các lượt trên đều hoàn tất trên artifact train/dev hiện có, chưa mở test.

## So sánh các ứng viên đang giữ

Macro-F1 (%); train 3.000, dev 2.219. InMaP không có random seed tối ưu;
majority lấy mean ba grouping seed. Binary dev không có nhãn thật.

| Phương án | Binary train | Human train | Human dev | Recall cứu trợ dev |
|---|---:|---:|---:|---:|
| Hai tầng gốc | 79,58 | 49,88 | 48,06 | 76,07 |
| Majority + DN-unit | 80,83 | 49,22 | 50,39 | 60,00 |
| InMaP gate + DN-unit | 81,75 | 50,36 | 49,75 | 70,36 |
| InMaP + DN parent product, khám phá | 78,18 | 48,23 | 50,01 | 80,00 |
| Majority + CALIP leaf | 80,83 | 47,63 | 51,14 | 61,07 |

InMaP + DN tăng human Macro-F1 trên cả train/dev so với gốc, và tăng recall
cứu trợ dev 10,36 điểm so với majority + DN. Nó vẫn có dev Macro-F1 thấp
hơn 0,65 điểm so với majority + DN và recall thấp hơn gốc 5,71 điểm. Chưa
có một ứng viên thắng mọi tiêu chí; không tự chuyển mặc định pipeline.
Đối chứng chỉ unit-normalize text có dev 50,40% nhưng train 47,63%, đã ghi
riêng, không quy mức tăng đó cho DN/InMaP.

Parent product giúp recall cứu trợ dev đạt 80,00%, nhưng binary train
giảm còn 78,18% và human train còn 48,23%; chưa thay thế hai ứng viên trên.
Mở rộng từ CHiLS giữ đầu ra tám lớp và giả định ánh xạ nhãn cha, không phải
tái lập CHiLS hoặc định nghĩa lại nhãn GT binary. Bỏ DMN chỉ mất 0,03 điểm
human dev so với có memory; chưa thấy lợi ích đủ rõ để giữ memory.

CALIP leaf có Macro-F1 dev cao hơn DN 0,75 điểm, tăng ở cả ba grouping
seed, nhưng Accuracy dev giảm 70,59→70,36% và train giảm 49,22→47,63%.
F1 phương tiện tăng, F1 cứu trợ giảm. Bootstrap theo query/event đều còn
chứa 0; giữ ứng viên, không kết luận thắng chắc chắn hoặc đổi test plan.

## Kiểm chứng và artifact

- [InMaP và tổ hợp DN](../../experiments/clip_inmap/README.md): 3 unit test,
  4 ca giả + đủ 8 fit thật (2.000 vòng) khớp hàm tác giả; audit prediction,
  metric, code/output/source hash. Artifact `clip_inmap/37d5a5db4a8a` và
  `clip_inmap_dn/1b61ddf777c1` trong `results/analysis/`.
- [LAME](../../experiments/clip_lame/README.md): 4 unit test, 4 ca giả +
  840 nhóm task khớp posterior và prediction tác giả; audit từng lớp.
  Artifact `results/analysis/clip_lame/e257fe3950a8`.
- [DMN và parent product](../../experiments/clip_dmn/README.md): 6 unittest,
  4 ca tổng hợp + 24 stream task thật khớp ba method tác giả; đủ bốn đợt
  đã audit hash, prediction, metric, memory và phép nhân confidence độc lập.
  Nhãn thật chỉ dùng đánh giá/chẩn đoán cache sau dự đoán; không chọn mẫu
  memory bằng GT. Single-view và visual-anchor là adaptation, không gọi
  benchmark DMN gốc. Không có lợi ích rõ từ memory ở cấu hình đã thử.
- [DPE và fusion](../../experiments/clip_dpe/README.md): 6 unittest, hai ca
  tổng hợp và 12 stream task thật khớp vòng DPE tác giả trên CPU FP32, max
  logit error 0. Audit toàn cache/prediction/metric; sáu input fusion khớp
  search_T Frolic gốc. Giữ đầy đủ residual và alignment, chỉ một view thật.
  Artifact `clip_dpe/6c5efeb4da87`, `clip_dpe_fusion/e1f3284c8605`.
- [Luồng final test đã chuẩn bị](../../experiments/clip_final/README.md):
  frozen plan DN-unit v1, loại trùng 5.219 ảnh train/dev, một job cùng ba đối
  chứng. 75 kiểm thử launcher/worker/runner và protocol đã đạt; bản test
  cập nhật thêm kiểm tra job plan vẫn giữ đúng flags/frozen asset.
  Chưa tạo cohort test thật hoặc gửi job. Kế hoạch này giữ ứng viên DN-unit
  v1; nếu sau nghiên cứu chọn ứng viên khác thì khóa protocol mới **trước**
  khi đo test, không ngầm gọi test DN-unit v1 là test của InMaP.

## Hướng khảo sát tiếp theo

1. **Ưu tiên CALIP/global-local:** đã đọc paper, core attention và extractor
   ViT, kiểm tra API OpenCLIP v3.3.0. Chẩn đoán majority + DN dev có trung
   bình 182,33 ảnh target-positive bị chặn, 220,33 target-negative được cho
   qua và 250 ảnh positive qua cổng chọn sai lớp. Đề xuất khai thác token
   patch trong cùng forward, giữ cổng để thử head và đo riêng root-only.
   Đã chạy cả train/dev. CALIP đã khóa coefficient trước khi đo,
   không dùng nhánh search bằng nhãn dev/test của repo tác giả.
   Xem [thiết kế và nguồn](clip_global_local_huong_moi_20261029.md), chẩn
   đoán `results/analysis/clip_bottleneck/93aaeef5ace4/`. Counterfactual dùng
   GT chỉ để phân tích, không phải model mới hoặc binary GT dev.
   Đã triển khai [runner CALIP](../../experiments/clip_calip/README.md),
   extractor cùng forward, sáu đối chứng/nhánh và audit tự động; 6 test mới
   đạt, operator khớp source commit đã ghim. Người dùng đã chạy 3.000 train
   run `2503caa2909f`; audit local và receipt đạt, đối chiếu get_logits tác
   giả trên năm ảnh thật cho prediction khớp. CALIP leaf chỉ tăng 0,0046
   điểm so unit control; attention rất phân tán (entropy chuẩn hóa ~0,9994).
   Dev 2.219 cũng đã hoàn tất run `03a1f0babaaf`: CALIP leaf đạt 51,14%,
   cao hơn DN 50,39% và unit control 50,40%, không đổi coefficient. Audit,
   receipt/code và source-reference thật đều đạt. Giữ ứng viên Macro-F1
   dev; chưa chọn thay DN vì train/Accuracy giảm và CI còn chứa 0. Review
   ở `results/analysis/clip_calip_review/a5ebf3236af7/`.
2. DPE single-view đã thử đủ residual, chưa vượt ứng viên hiện có. Muốn
   đánh giá nhánh ImageNet nhiều view cần export augmentation thật và cân
   đối chi phí; chưa tự tạo job mới hoặc suy rằng nhiều view sẽ tăng điểm.
3. Frolic đầy đủ chưa tái lập: core công khai còn thiếu utils; công thức
   covariance thực nghiệm giả định prior đều. Chỉ confidence matching đã
   thử ở trên, không gọi các số fusion là kết quả Frolic đầy đủ.

Mỗi đợt mới cần giả thuyết, cấu hình trước khi xem metric, cùng cohort/
backbone/prompt, đối chứng thích hợp, metric 2/8 nhãn khi có GT thật, và
record cả kết quả giảm. Không dùng test để chọn phương pháp hoặc tinh chỉnh.
