# CALIP cho CrisisMMD hai tầng

Đã hoàn tất MetaCLIP2 thật trên **3.000 ảnh train**, run `2503caa2909f`,
và **2.219 ảnh dev**, run `03a1f0babaaf`. Cả hai audit local đạt;
**chưa có CALIP test**. CALIP leaf tăng Macro-F1 dev nhưng giảm train.
Giai đoạn triển khai ban đầu kiểm thử bằng ảnh giả và OpenCLIP ViT nhỏ;
lượt thật do người dùng chạy trên Kaggle. Lượt kiểm tra kết quả chỉ đọc
artifact, không encode lại ảnh hoặc gửi thêm job.

## Kết quả train đã xác minh

Macro-F1 (%), cùng cohort, mean ba grouping seed:

| Phương án | Binary train | Humanitarian train |
|---|---:|---:|
| Hai tầng gốc | 79,58 | 49,88 |
| Majority + DN-unit | 80,83 | 49,22 |
| Majority + unit prototype | 80,83 | 47,6258 |
| Majority + CALIP leaf | 80,83 | 47,6304 |
| CALIP gate + DN-unit | 80,31 | 49,22 |

CALIP leaf giảm 1,59 điểm so với DN-unit, gần như giữ nguyên unit control
(chỉ +0,0046 điểm). Mỗi seed đổi 33–34 nhãn so với unit control, sửa đúng
11 và làm sai 8 ảnh. Không quy toàn bộ mức giảm so với DN cho attention vì
hai nhánh cũng khác cách hiệu chỉnh embedding.

Cổng CALIP tăng binary train so với gốc nhưng thấp hơn majority và InMaP.
Recall cứu trợ train CALIP gate + DN đạt 65,18% so với majority + DN
57,50%, vẫn thấp hơn gốc 73,04%. Không có một ứng viên cải thiện mọi mặt.

Attention humanitarian lưu FP16 có entropy chuẩn hóa trung bình khoảng
0,99943 trên 256 patch (đều hoàn toàn là 1). Đây là tín hiệu attention rất
phân tán ở cấu hình hiện tại, chưa là bằng chứng định vị được vùng hữu ích
hoặc nguyên nhân duy nhất làm phương pháp kém. Không đổi hệ số theo kết quả.

Đã đối chiếu code/output hash, receipt với staging, ID/GT, score và metric
từng lớp. Global reference error=0; cosine so với artifact gốc sai khác tối
đa 5,63e-6, mọi argmax gốc khớp. Trích get_logits tác giả chạy trên feature
thật của năm ảnh đầu, hai task: mọi prediction khớp, max logit error 4,20e-5
trong tolerance FP32. Báo cáo:
`results/analysis/clip_calip_review/33ce66d89318/{audit,author_reference,comparison}.json`.

## Kết quả dev đã xác minh

Cùng 2.219 ảnh dev, mean ba grouping seed; binary GT dev không có, giữ null:

| Phương án | Human Macro-F1 | Accuracy | Recall cứu trợ |
|---|---:|---:|---:|
| Hai tầng gốc | 48,06 | 66,20 | 76,07 |
| Majority + mean cosine | 49,24 | 69,84 | 60,48 |
| Majority + unit prototype | 50,40 | 69,96 | 61,07 |
| Majority + DN-unit | 50,39 | 70,59 | 60,00 |
| **Majority + CALIP leaf** | **51,14** | 70,36 | 61,07 |
| CALIP gate + DN-unit | 49,78 | 69,09 | 68,57 |

CALIP leaf tăng 0,7462 điểm Macro-F1 so với DN và 0,7440 điểm so với unit
control. Cả ba seed đều tăng: CALIP 51,0465/50,5983/51,7779%, DN tương ứng
50,2230/50,1366/50,8245%. Không chọn seed cao nhất hoặc đổi coefficient.

So với unit control, mỗi seed sửa 13 và làm sai 4 ảnh. So với DN, mỗi seed
sửa 15 và làm sai 20: Accuracy giảm 0,2253 điểm, dù Macro-F1 tăng do cân
bằng precision/recall giữa các lớp. F1 phương tiện tăng 52,36→59,74%, F1
cứu trợ giảm 51,96→48,81%. Recall cứu trợ chỉ tăng 60,00→61,07%. Lớp
missing/found góp khoảng 0,116 trong tổng 0,746 điểm tăng; mức tăng không
chỉ đến từ lớp hai mẫu nhưng vẫn nhạy với các lớp ít mẫu.

Bootstrap ghép cặp 2.000 lần, giữ prediction cố định, theo 30 nhóm query
và riêng theo 7 sự kiện: mọi khoảng 95% của chênh lệch CALIP−DN/CALIP−unit
vẫn chứa 0. Chỉ xem đây là tín hiệu thăm dò; nhóm query có thể cùng tweet/
event, event bootstrap có ít cụm, chưa hiệu chỉnh nhiều phép so sánh và
dev đã dùng nhiều lần. Không coi ba seed là ba dataset độc lập.

Dev đã kiểm tra receipt/staging, code/output hash, ID/GT và toàn bộ metric.
Feature năm ảnh dev đầu khớp get_logits tác giả cho cả hai task. Chi tiết:
`results/analysis/clip_calip_review/a5ebf3236af7/{audit,author_reference,comparison}.json`.

**Quyết định:** giữ majority + CALIP làm ứng viên theo Macro-F1 dev, chưa
thay pipeline hoặc chốt model cuối vì train và Accuracy dev so với DN giảm,
độ bất định còn lớn. Frozen final plan DN-unit v1 vẫn nguyên vẹn; muốn test
CALIP phải khóa protocol phù hợp trước khi đo, không dùng test chọn hệ số.

## Cơ chế đã khóa

[CALIP, AAAI 2023](https://arxiv.org/abs/2209.14169),
[repo tác giả](https://github.com/ZiyuGuo99/CALIP), commit
`83e21b6b985934dce6e8414c26300671b4825841`.

- Global và patch lấy từ cùng forward của OpenCLIP ViT CLS pooling. Lấy
  token sau đúng LayerNorm/projection hiện có, chuẩn hóa unit. Giữ điểm
  toàn ảnh; forward tham chiếu bổ sung ở batch đầu phải khớp embedding.
- Attention nhân 2 như `get_logits` tác giả; hai chiều spatial/text,
  pooling mean + max. Không chuẩn hóa lại feature sau attention.
- Ba term logits: global CLIP, global ảnh với text đã tương tác, visual
  đã tương tác với text. Hệ số cố định `[1, 1.12, 0.02]` từ bộ ImageNet
  trong paper. Scale=100 của checkpoint hiện tại. Không chạy nhánh search
  dùng labels của repo tác giả hoặc dò coefficient trên dev.
- CALIP giữ toàn bộ bank tám lớp khi tính attention; lúc qua cổng mới chọn
  trong bảy lớp positive. Không sinh lại prompt với taxonomy bảy lớp.
- Prompt/prototype semantic tầng 1 và semantic Waffle seed 0 tầng 2 giữ
  đúng artifact cũ. Text ở đây là prompt, không sử dụng tweet.

Đây là adaptation cho MetaCLIP2, prompt CrisisMMD và hệ số cố định; không
nhận đã tái lập điểm benchmark tác giả hoặc đã chứng minh tính mới toàn cầu.

## Các phương án trong một job

| Tên trong summary | Cổng | Tầng chi tiết |
|---|---|---|
| `two_stage_original` | CLIP semantic gốc | Mean cosine gốc |
| `majority_raw_leaf` | Majority hiện có | Mean cosine gốc |
| `majority_unit_leaf` | Majority hiện có | Prototype unit, đối chứng normalization |
| `majority_dn_unit` | Majority hiện có | DN-unit đang giữ |
| `majority_calip_leaf` | Majority hiện có | **CALIP tám lớp mới** |
| `calip_gate_dn_leaf` | **CALIP hai lớp mới** | DN-unit đang giữ |

Hai nhánh mới tách root-only và leaf-only để xác định tác động từng phần.
Chưa gộp cả hai tầng CALIP. Không fit EM/StatA lại trong lượt GPU này.

`configs/prompts/crisismmd_calip_reference.json` được chuẩn bị chỉ từ result
train 3.000/dev 2.219 đã kiểm chứng: prompt banks/prototypes, ID/hash ảnh,
score gốc, DN leaf và majority gate của ba grouping seed 0/1/2. Nhãn trong
reference chỉ phục vụ xác nhận cohort và metric; không đưa vào core/extractor.
Không có support ảnh có nhãn hoặc fit supervised.

Mỗi job đã tính ba cổng đối chứng và mean, không gửi ba job. CALIP root
xác định, không coi các bản ghi lặp lại là ba lần model độc lập. Smoke 5
dùng prefix của cổng full-cohort đã lưu, chỉ kiểm tra kỹ thuật. Runner từ
chối số mẫu khác 5 hoặc đủ cohort đã khóa, từ chối test và flags thí nghiệm
khác. Nếu ảnh, thứ tự, nhãn, prompt embedding hoặc argmax baseline khác
đối chứng, dừng trước khi nhận kết quả CALIP.

## Chạy thử lần lượt

Tại thư mục gốc. Chạy smoke 5 ảnh trước:

```bash
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_calip/launcher.yaml \
  --job calip_train --max-samples 5 --use-kaggle true
```

Chỉ sau khi smoke thành công, chạy đủ 3.000 ảnh train:

```bash
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_calip/launcher.yaml \
  --job calip_train --max-samples 3000 --use-kaggle true
```

Nếu hai bước trên đạt, đánh giá toàn bộ 2.219 dev:

```bash
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_calip/launcher_dev.yaml \
  --job calip_dev --use-kaggle true
```

Dev dùng đúng 2.219 ảnh, không thêm `--max-samples 3000`. Không gửi các job
đồng thời vào cùng kernel. Nếu đã gửi rồi mất kết nối, dùng `--resume-stage`
launcher in ra; không gửi lại lượt đang chạy. Để chạy local đổi
`--use-kaggle false`, nhưng MetaCLIP2 lớn và có thể tải checkpoint.

## Kết quả và audit

Output riêng:

```text
results/clip_calip/crisismmd/humanitarian/samples_N/metaclip2_vith14/<run>/
results/clip_calip_heldout/crisismmd/humanitarian/samples_2219/metaclip2_vith14/<run>/
```

Runner tự audit trước khi hoàn tất: hash, ID/GT, score baseline, tổng ba
term CALIP, tái tính operator trên feature năm ảnh đầu (hoặc batch đầu nếu
nhỏ hơn 5), từng prediction và metric đủ các lớp. Audit lỗi làm job failed.
Mỗi run có `analysis.json`, `audit.json`, `summary.tsv`, `scores.npz`,
`verification_features.npz`, `extraction.json`, ba TSV seed và frozen
reference/cohort. Attention lưu FP16 để giảm dung lượng; score term FP32.
Không lưu toàn bộ patch feature full dataset, chỉ tối đa năm ảnh xác minh.

Phân tích sau khi tải Kaggle, nhận cả folder giải nén hoặc ZIP worker:

```bash
venv/bin/python -B experiments/clip_calip/analyze.py --run-dir PATH_KET_QUA
```

Analyzer không đọc ảnh/model, từ chối backend giả mặc định. Macro-F1 binary
dev là null khi thiếu annotation thật; không suy GT binary từ humanitarian.
So `majority_calip_leaf` với cả DN-unit và unit control cùng cổng; so cổng
CALIP bằng binary train và đầu ra tám lớp. Attention chỉ là tín hiệu giải
thích, không có mask/bbox GT để khẳng định định vị đúng.

## Kiểm chứng triển khai

- `reference_verification.json`: bốn ca 2/8 lớp, chiều 16/1024, trích đúng
  hàm `get_logits` từ AST tác giả. Hai term attention, term global và
  prediction khớp trong tolerance FP32. Không import dataset hoặc search.
- Sáu kiểm thử mới: bằng chứng vùng thay quyết định khi global mơ hồ;
  invariance hoán vị patch/lớp và input invalid; extractor với hai thứ tự
  LayerNorm/pooling, global bit-identical và restore flag khi lỗi; launcher
  và code bundle; end-to-end train/dev, ZIP và phát hiện sửa output; prepare
  không đọc dữ liệu/nạp model và từ chối cohort lệch trước model.
- Đã kiểm tra runner chuẩn bị train/dev thật với cả reader/model bị chặn;
  chỉ đọc cấu hình/reference, không đọc processed. Bằng chứng trong `checks/`.
- Thay đổi launcher chỉ thêm cờ `calip` cho job `clip_two_stage`, entrypoint
  và output riêng. Không sửa runner hai tầng gốc, worker hoặc các file
  CODE trong frozen final plan; DN-unit v1 vẫn xác thực nguyên vẹn.

Kiểm thử và reference:

```bash
venv/bin/python -B -m unittest experiments.clip_calip.test_calip -b -q
venv/bin/python -B experiments/clip_calip/verify_reference.py \
  --reference-dir /private/tmp/vlm-calip-reference \
  --output /private/tmp/calip_reference_check.json
```

Có thể thêm `--features PATH/verification_features.npz` để đối chiếu operator
tác giả trên các feature xuất từ lượt thật. Kiểm tra CPU/tensor giả chưa
xác nhận memory/runtime trên T4; smoke thật ở trên vẫn cần thực hiện.
