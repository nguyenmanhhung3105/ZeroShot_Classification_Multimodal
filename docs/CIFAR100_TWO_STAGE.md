# Thí nghiệm CIFAR-100: phẳng và phân cấp hai tầng

Nhánh độc lập, không đổi các job benchmark/Fakeddit hiện có. Model giữ nguyên
MetaCLIP2 ViT-H/14 worldwide, FP32 và preprocessing của `Backbone` hiện tại.
Không fine-tune, không StatA/EM, không tải dataset lại và không dùng train để fit.

## Giao thức cố định ban đầu

`configs/cifar100_two_stage.json`: seed 0, dev 10.000 mẫu, batch ảnh 8, logit
scale 100 cho cả fine/coarse, một prompt `a photo of {label}.` cho mỗi lớp.
Tên lớp lấy đúng ID từ manifest CIFAR, chỉ đổi dấu gạch dưới thành khoảng trắng.
Prototype và feature ảnh được chuẩn hóa unit. Không thử grid hoặc tự chọn winner.

Dev = 10.000 ID từ `default_rng(0).permutation(50000)[:10000]`, sau đó sắp ID.
Chọn ID trước khi đọc nhãn/ảnh, không stratify. 40.000 train còn lại chưa dùng.
Test giữ nguyên 10.000 mẫu chính thức; ID luôn có tiền tố train/test để phân biệt.
Smoke lấy 5 ID đầu của dev đã cố định; không chọn mẫu theo dự đoán và không smoke test.

Ảnh chỉ được đọc/giải mã theo batch bằng seek bản ghi binary; không chuyển cả
dataset thành PNG hoặc nạp toàn bộ ảnh trước khi chọn subset.

## Ba phương pháp

Gọi `z_f` là logits 100 lớp fine, `z_g` là logits trực tiếp từ 20 prompt nhóm.
`q = softmax(z_g)`, `r(y|g) = softmax(z_f trên 5 lớp con của g)`.

- `flat`: `softmax(z_f)` trên đủ 100 lớp.
- `hierarchy_hard`: chọn `g=argmax(q)`, chỉ dự đoán trong 5 lớp con nhóm đó.
- `hierarchy_soft`: `p(y)=q(parent(y))*r(y|parent(y))`, argmax trên đủ 100 lớp.

Cả ba dùng cùng feature và logits fine. Hai phương pháp phân cấp sử dụng thêm
20 prompt coarse, vì vậy cải thiện nếu có chưa tự chứng minh chỉ riêng việc chia
tầng có lợi. Cần ablation tiếp theo để tách lợi ích prompt nhóm và định tuyến.
Chỉ lọc/re-normalize chính logits cũ trong một nhóm không tạo thêm bằng chứng ảnh.
Nếu q được cộng từ xác suất fine thay vì prompt nhóm riêng, p mềm bằng p phẳng;
test số học kiểm tra tính chất này. Không tuyên bố đây là phương pháp mới hay
tái lập nguyên HD-CNN/CascadeVLM/TLAC.

## 1. Chuẩn bị và gửi smoke inference 5 ảnh

### Cách khuyến nghị: dùng launcher như các job tầng 1

Full dev 10.000 mẫu, một lệnh gồm kiểm tra quota, gửi, theo dõi 120 giây/lần,
tải ZIP kết quả, kiểm tra checksum và in bảng metric:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job cifar100_two_stage_dev --use-kaggle true --wait
```

Smoke inference 5 ảnh dùng cùng lệnh, thêm `--max-samples 5` (hoặc chọn job
`cifar100_two_stage_smoke`). Thêm `--prepare-only` để chỉ xem giới hạn và đóng gói,
không gọi API, quota, tải model hay gửi GPU job.

Launcher chặn job queued/running, lưu version sau push và theo dõi đúng version đó.
Mất kết nối sau push: dùng `--resume-stage <Stage đã in>` để tiếp tục chờ/tải,
không gửi lại. Kết quả ở `results/final/kaggle/cifar100_two_stage_dev_<id>/run/`.
Code CIFAR nằm trong notebook whitelist, nên không upload thêm code Dataset và
không upload lại Dataset CIFAR. `code_dataset: null` trong receipt là bình thường.

### Cách thủ công bằng Kaggle CLI

```sh
venv/bin/python -B scripts/run_cifar100_two_stage.py --mode smoke --prepare-only
venv/bin/python -B scripts/prepare_cifar100_two_stage_job.py \
  --mode smoke --output /tmp/vlm-cifar100-two-stage-smoke
```

Lệnh chuẩn bị không truy cập dataset, không tải model và không gửi job. Notebook
độc lập chứa whitelist code/config/requirements, không kèm ảnh, token hoặc repo tác giả.

Trước khi push, kiểm tra quota và trạng thái để tránh chạy trùng:

```sh
venv/bin/kaggle quota
venv/bin/kaggle kernels status nguyenhung3105/vlm-cifar100-two-stage-smoke-v1
venv/bin/kaggle kernels push -p /tmp/vlm-cifar100-two-stage-smoke
```

Job dùng GPU T4, Internet để cài dependencies và tải MetaCLIP2/tokenizer; chỉ
gắn `nguyenhung3105/vlm-cifar100-sources-v1`. Giữ precision/backbone hiện có.
Không chạy full ngầm sau smoke. Theo dõi mỗi hai phút:

```sh
while true; do
  date
  venv/bin/kaggle kernels status nguyenhung3105/vlm-cifar100-two-stage-smoke-v1
  sleep 120
done
```

Khi Complete, tải Output nhỏ:

```sh
venv/bin/kaggle kernels output nguyenhung3105/vlm-cifar100-two-stage-smoke-v1 \
  -p results/final/cifar100_two_stage_smoke
```

Đạt khi có ba dòng metric, 5 mẫu/phương pháp, `completed.json` và
`run.json` status complete. Accuracy/macro-F1 của 5 ảnh không dùng báo cáo chất
lượng; macro-F1 luôn tính đủ 100 lớp nên đặc biệt thấp nếu support chỉ 5 ảnh.

## 2. Dev đầy đủ, chỉ chạy sau khi smoke hợp lệ

```sh
venv/bin/python -B scripts/prepare_cifar100_two_stage_job.py \
  --mode dev --output /tmp/vlm-cifar100-two-stage-dev
venv/bin/kaggle kernels push -p /tmp/vlm-cifar100-two-stage-dev
```

Job riêng `nguyenhung3105/vlm-cifar100-two-stage-dev-v1`, đủ 10.000 dev.
Xem toàn bộ các đối chứng, không chọn seed tốt nhất. Đổi prompt/tham số sau khi
xem dev là phát triển; phải chạy lại dev để có artifact khóa mới.

## 3. Test độc lập

Chỉ chuẩn bị khi đã hoàn tất dev; giữ cùng code, cấu hình, requirements,
phiên bản thư viện và source manifest. Không smoke hoặc tìm tham số trên test.

```sh
venv/bin/python -B scripts/prepare_cifar100_two_stage_job.py \
  --mode test --dev-kernel nguyenhung3105/vlm-cifar100-two-stage-dev-v1 \
  --output /tmp/vlm-cifar100-two-stage-test
venv/bin/kaggle kernels push -p /tmp/vlm-cifar100-two-stage-test
```

Notebook gắn Output dev làm Input, yêu cầu đúng một run full-dev complete.
Trước khi đọc pixel test, kiểm tra completion, hash run/artifact và protocol.
Smoke không được dùng để mở test. Khóa này bảo vệ nhất quán giao thức, không phải
cơ chế chống người vận hành tự sửa dữ liệu hoặc dùng kết quả test trước đó.

## Đầu ra

- `frozen_plan.json`, `run.json`: config, prompts, taxonomy, code/requirements/library
  versions, seed, fingerprint ID và các bản ghi đã đọc, model runtime.
- `summary.tsv`, `metrics_by_seed.tsv`, `per_class.json`: accuracy, macro-F1,
  balanced accuracy, top-5. Một lượt xác định cho mỗi phương pháp, không std nhiều seed.
- `predictions.tsv`: cùng ID/nhãn thật cho ba đối chứng.
- `comparisons.tsv`: delta accuracy, số sửa đúng/làm sai so với flat.
- `routing.json`: accuracy nhóm trực tiếp và nhóm suy từ flat, lỗi hard gate.
- `replay_scores.npz`: logits fine/coarse, xác suất ba phương pháp, mapping, ID/nhãn;
  đọc với `allow_pickle=False`. Không chứa ảnh hay trọng số.
- `completed.json`: chỉ tạo sau khi hoàn tất và hash artifact.

Chưa có CI/kiểm định ý nghĩa thống kê trong runner này. Chênh lệch dev chưa phải
bằng chứng tổng quát hóa; phân tích paired trên test và ablation là bước tiếp theo.

## Kiểm tra local

```sh
venv/bin/python -m unittest discover -s tests -p test_cifar100_two_stage.py -v
```

Kiểm tra normalization/mapping không liên tiếp, tính đồng nhất khi nhóm cộng từ
flat, split cố định, chặn test từ smoke hoặc protocol sai, checksum khóa, end-to-end
5 record binary tổng hợp với backend giả và notebook whitelist/compile.
Không đọc dataset thật, không tải weights, không gửi GPU job trong kiểm tra local.
