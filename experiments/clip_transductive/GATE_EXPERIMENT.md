# Transductive ở cổng informativeness, humanitarian giữ argmax độc lập

## Phạm vi theo yêu cầu

Áp dụng EM-Dirichlet cho task hai nhãn informative/not_informative. Sau đó
ghép cổng này vào hệ thống hai tầng, giữ cách chọn lớp humanitarian hiện tại:

```text
Ảnh → MetaCLIP2 → score semantic informativeness → transductive hai nhãn
  ├─ not_informative → not_humanitarian
  └─ informative    → argmax độc lập trên bảy lớp humanitarian dương
```

Giữ đủ tám lớp đầu ra humanitarian gốc, không kế thừa ablation bỏ lớp trước.
Không chạy lại backbone hoặc sinh lại prompt. Đây là runner thử nghiệm độc lập;
cấu hình và artifact của phương pháp hai tầng cũ được giữ nguyên.

## Dữ liệu và giao thức

Hai phép đánh giá riêng, không trộn cohort hoặc coi train là dev:

1. **Binary train:** 3.000 ảnh có nhãn informativeness thật, dùng artifact
   `results/waffle_lab_semantic/crisismmd/informativeness/samples_3000/metaclip2_vith14/informativeness_44c15f5c695f/`.
2. **Humanitarian dev:** cùng 2.219 ảnh có nhãn humanitarian đã đánh giá các
   phương pháp trước, dùng artifact
   `results/clip_two_stage_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/humanitarian_77187aa56f13/`.

Cohort dev này không có nhãn informativeness thật: metric binary trên dev
được để null. Không suy nhãn informative từ humanitarian để báo Accuracy/F1
task informativeness. Vẫn đánh giá được đầu ra tám lớp của toàn hệ thống.

Loader xác thực status, ID, score, cohort/protocol fingerprint và baseline
từ artifact gốc. Kiểm tra cùng backbone và nguyên bank prompt semantic
informativeness giữa hai nguồn. Nguồn binary là train có nhãn ảnh, nguồn
humanitarian là dev có nhãn ảnh. Không mở dataset, raw hoặc trọng số model.

Cấu hình khóa ở `config_gate.json`: temperature=30, query size=75, 20 vòng
EM ngoài, tối đa 1.000 vòng MM trong, grouping seeds [0,1,2], không support.
Theo code tác giả lambda/N=floor(K/5)=0 khi K=2; không tự tăng lambda để tune.

Mỗi lượt xáo trộn tất cả ảnh trong cohort, bao gồm cả những ảnh cổng cũ chặn,
để cổng mới có thể đổi cả hai hướng. Binary train có 40 nhóm 75 ảnh. Dev có
29 nhóm 75 và một nhóm 44 ảnh. Các nhóm không dùng nhãn thật hay sự kiện.
EM-Dirichlet + Hungarian chỉ dùng feature `softmax(30 * info_cosine_scores)`.

Tầng humanitarian giữ nguyên score semantic_waffle prompt seed 0 và argmax
trên bảy lớp dương. Grouping seed thay cách chia nhóm, không đổi prompt seed.

## Kết quả ngày 28-10-2026

| Đánh giá | Phương pháp | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|
| Binary train, 3.000 mẫu | Semantic argmax cũ | 79,93 | 79,58 |
| Binary train, 3.000 mẫu | Transductive seed 0 | 79,97 | 79,95 |
| Binary train, 3.000 mẫu | Transductive seed 1 | 80,03 | 80,02 |
| Binary train, 3.000 mẫu | Transductive seed 2 | 80,10 | 80,09 |
| Binary train, 3.000 mẫu | Transductive trung bình | 80,03 | 80,02 |
| Humanitarian dev, 2.219 mẫu | Hai tầng cũ | 66,20 | 48,06 |
| Humanitarian dev, 2.219 mẫu | Cổng transductive seed 0 | 69,00 | 48,52 |
| Humanitarian dev, 2.219 mẫu | Cổng transductive seed 1 | 69,27 | 48,34 |
| Humanitarian dev, 2.219 mẫu | Cổng transductive seed 2 | 69,18 | 49,28 |
| Humanitarian dev, 2.219 mẫu | Cổng transductive trung bình | 69,15 | 48,71 |

Trung bình binary Macro-F1 tăng 0,43 điểm phần trăm, Accuracy tăng 0,10 điểm.
Humanitarian Macro-F1 tăng **0,66 điểm phần trăm**, Accuracy tăng **2,94 điểm**.
Độ lệch chuẩn tổng thể Macro-F1 giữa các nhóm: binary 0,06 điểm, humanitarian
0,41 điểm. Không chọn seed 2 làm kết quả đại diện hoặc ensemble dự đoán.

## Đánh đổi quan sát được

Cổng mới cho ít ảnh qua hơn: 1.417 ảnh ở baseline, trung bình 1.156 ảnh với
transductive. F1 not_humanitarian tăng từ 76,17% lên trung bình 80,07%.

Tuy nhiên recall informative trên binary train giảm từ **89,94% xuống 80,01%**;
recall not_informative tăng từ 69,22% lên 80,06%. Đây là chuyển đổi giữa khả
năng phát hiện hai lớp, không phải cả hai cùng tốt hơn.

Trên humanitarian dev, recall rescue/volunteering/donation giảm từ **76,07%
xuống 58,10%**, F1 lớp này giảm từ 51,02% xuống 47,87%. F1 vehicle tăng từ
50,67% lên 55,00%; F1 missing/found giảm từ 4,76% xuống 2,90% (chỉ hai mẫu).
Không kết luận cổng mới cải thiện mọi lớp từ Accuracy/Macro-F1 trung bình.

Đây là ứng viên cải thiện nhẹ ở metric tổng thể. Cần cân nhắc mục tiêu phát
hiện thông tin hữu ích/cứu trợ khi chọn phương án cuối, và kiểm chứng độc lập
sau khi khóa cấu hình. Chưa thay mặc định của pipeline chính.

## Xác minh

- 17 kiểm thử transductive, bỏ lớp và cổng mới đạt. Kiểm tra K=2 có lambda=0,
  objective giảm, đủ ảnh, nhóm singleton, thứ tự lớp, tính lặp lại seed,
  taxonomy, tầng humanitarian giữ argmax và tách metric train/dev.
- Audit hash code, hai nguồn artifact và 20 file output; tính lại metric,
  fixed/broken, ID/thứ tự/ground truth và batch membership cả ba seed.
- Đối chiếu toàn bộ **210 nhóm** ở hai cohort/ba seed với EM_DIRICHLET và
  Hungarian gốc, commit `9066ad371757c794e4b8255e852f837ac9a1d3b1`: mọi nhãn
  binary lưu lại đều khớp, sai số assignment float64 tối đa dưới 4,5e-14.
- Tính lại đầu ra humanitarian từ nhãn binary tham chiếu và score tầng 2:
  khớp toàn bộ dự đoán đã lưu. Khi nhãn cổng không đổi, đầu ra humanitarian
  không đổi. Không fabricate nhãn informativeness dev.

Artifact: `results/analysis/clip_transductive_gate/c3fa825bc57b/`.
Có `analysis.json`, `summary.tsv`, `run_manifest.json`, `audit.json` và
`{binary_train,humanitarian_dev}/grouping_seed_N/` chứa dự đoán, metric,
objective và ánh xạ cụm.

## Giới hạn

Binary được đo trên train đã dùng thăm dò; dev humanitarian đã được dùng
trong nhiều lần chọn hướng. Chưa có binary dev metric hoặc test độc lập.
Ba seed đo độ nhạy chia nhóm, không chứng minh ý nghĩa thống kê. Tính
transductive khiến dự đoán một ảnh phụ thuộc những ảnh cùng nhóm; không
diễn giải như suy luận độc lập từng ảnh của baseline.

## Tái lập

Lượt trên đã chạy xong bằng CPU từ điểm CLIP có sẵn. Chạy lại sẽ tạo thư mục
kết quả mới, không cần job Kaggle:

```sh
venv/bin/python -B experiments/clip_transductive/gate.py
```

Kiểm thử:

```sh
venv/bin/python -B -m unittest \
  experiments.clip_transductive.test_gate \
  experiments.clip_transductive.test_transductive \
  experiments.clip_transductive.test_drop_class -b -q
```
