# Few-shot min5 hai tầng: Real/Fake → năm loại Fake

Đã chạy full dev 3.000 mẫu và audit version 2: xem [kết quả hai tầng/flat cùng cohort](FAKEDDIT_FEWSHOT_CASCADE_MIN5_REVIEW.md). 18 artifact, 30 tổ hợp method–seed và 90.000 prediction hợp lệ. Chưa mở test; phần hướng dẫn dưới đây giữ để tái hiện, không gửi lại chỉ vì đã có stage cũ.

## Phương pháp và đối chứng

- Backbone MetaCLIP2 đóng băng; ảnh/tiêu đề unit, nối rồi chia sqrt(2), cùng cách tạo feature của baseline Task 1.
- Hai policy, năm seed 0–4. Mỗi classifier dùng pool support 384 ID/seed: legacy 64 Real + 320 Fake; balanced 192 Real + 192 Fake. Giữ đúng ID/thứ tự/quota đã lưu ở run support nguồn.
- Mỗi policy/seed fit ba LogisticRegression C=1, class_weight=balanced, solver=lbfgs, max_iter=2000, tol=1e-6: binary gate trên 384 support; child năm lớp trên đúng subset Fake có sẵn (320 hoặc 192); flat sáu lớp trên cùng 384 support. Không bổ sung ID có nhãn riêng cho tầng hai.
- Gate dùng default argmax (tie Real) hoặc ngưỡng Task 1 target selection 87% đã lưu. Không tìm lại ngưỡng từ kết quả cascade. Mỗi gate Fake được chuyển sang argmax child + 1, gate Real trả lớp 0. Đây là hard routing, không lấy argmax của posterior joint sáu lớp.
- Lớp 0=Real, 1=Satire/Parody, 2=False Connection, 3=Imposter, 4=Manipulated, 5=Misleading. Mã binary 0=Real/1=Fake, ngược với `2_way_label` nguồn; code dùng nhãn 6way để suy ra binary.

Tổng cộng 30 head fit (2 policy × 5 seed × 3 head), tạo 10 đối chứng flat và 20 hierarchy method–seed. Bảng tổng hợp có sáu hàng: mỗi policy gồm flat, hierarchy default và hierarchy ngưỡng cố định. Không gọi đây là phương pháp zero-shot hay một thuật toán học mới.

## Snapshot dev

`configs/fakeddit_fewshot_cascade_min5.json` được xuất từ:

- Source: `results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run`.
- Review ngưỡng: `results/final/fewshot_support_thresholds/recall_tradeoff/dev_1b4a7accc90c`.
- Snapshot SHA256 nội dung payload: `68c6661478044fa7d56862ef846c581dc8cf5a0950aa7181adc5ac211b07f275`.

Exporter tái dùng bộ kiểm protocol để xác minh checksum, support train-only/quota và tái tính ngưỡng từ selection trước khi xuất. Snapshot chỉ chứa cấu hình, support ID/nhãn, partition ID và hash/provenance; không chứa ảnh, tiêu đề gốc hoặc weights. Đây là recipe **dev-only portable**, không phải test lock; runtime/code hiện tại được ghi riêng trong run và bundle manifest.

Snapshot đã tạo bằng `scripts/prepare_fakeddit_cascade_min5.py`; file được tạo exclusive, không tự ghi đè. Nếu cần recipe khác, xuất tên file mới và ghi rõ lý do. Không thay hash dataset, C, seed hoặc ngưỡng để chạy test bằng runner dev.

## Xem kế hoạch local, không GPU/dataset/Kaggle

```sh
venv/bin/python -B scripts/run_fakeddit_cascade_min5.py --prepare-only
```

Kế hoạch: 3.000 dev, hai tầng, 384 support/seed, 5 seed, 20 gate configurations và 10 flat controls; `test_opened=false`. Có thể thêm `--smoke` để kế hoạch chỉ đánh giá 5 dev.

CLI local thực thi cần `--train-root` và `--dev-root` tường minh, không tự tìm test/legacy slug. Không thực thi inference local trong bước phát triển này.

## Chuẩn bị Kaggle

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_cascade_min5.yaml \
  --job fakeddit_fewshot_cascade_min5 --max-samples 5 --prepare-only
```

Chỉ tạo bundle/notebook/stage local. Notebook gắn đúng `vlm-fakeddit-train-min5-v1` và `vlm-fakeddit-dev-min5-v1`, không gắn test, không gọi freeze legacy. Kernel riêng: `nguyenhung3105/vlm-min5-fakeddit-fewshot-cascade-min5-smoke`.

Sau khi được duyệt gửi smoke, lệnh thực thi là:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher_cascade_min5.yaml \
  --job fakeddit_fewshot_cascade_min5 --max-samples 5 --use-kaggle true --wait
```

Smoke giữ đủ 10 support set × 384 ID (hợp **1.591 ID train**), fit đủ 30 head nhưng chỉ đánh giá 5 dev. Không hiểu smoke là tổng cộng chỉ 5 ảnh. Không tự chuyển sang full. GPU cấu hình T4, timeout 7.200 giây là giới hạn tối đa, không phải thời gian chạy đã đo của cascade. Quota chưa được truy vấn trong prepare-only.

Sau khi smoke và artifact được kiểm tra, full dev dùng lệnh tương tự bỏ `--max-samples 5`. Khi gián đoạn theo dõi, dùng `--resume-stage` của job đã gửi, không tự submit trùng. Không tự gửi GPU hoặc upload dataset trong bước triển khai.

## Kiểm tra trước model

Runner chỉ có đường dev, không có option chạy test. Kiểm SHA256 manifest train/dev min5, số mẫu, fingerprint ID/hash ảnh/hash text của full cohort; chặn overlap; xác minh support đúng nguồn/quota trước model. Smoke vẫn kiểm manifest của full cohort và toàn bộ train; `data.fakeddit` kiểm schema/nguồn dữ liệu như baseline.

Model chỉ nạp một lần, encode hợp train support và dev một lượt mỗi tập; hai policy dùng chung feature. Runtime architecture/pretrained/tokenizer/context/dtype/image_size phải khớp nguồn. Tuple/list image_size được chuẩn hóa khi so metadata. Backend được unload cả khi encoding lỗi.

Hàm `infer` chỉ nhận feature query, không nhận nhãn dev. Chỉ sau khi mọi gate/child/flat posterior đã được tính xong, hàm `evaluate` mới dùng nhãn dev cho metrics, routing và oracle.

## Đọc kết quả

Run mới: `results/final/kaggle/fakeddit_fewshot_cascade_min5_<stage>/run/` khi thu qua launcher, hoặc `settings.output_root/fakeddit_fewshot_cascade_min5/dev_<id>` nếu chạy local.

| Artifact | Nội dung |
|---|---|
| `summary.tsv`, `metrics_by_seed.tsv` | Accuracy/macro-F1/BA sáu lớp end-to-end, flat và cascade, toàn dev |
| `comparisons.tsv`, `comparisons_summary.tsv` | Cascade − flat cùng policy, support/seed, ID và modality; delta điểm %, sửa sai/làm sai |
| `gate_metrics.tsv` | Binary gate, recall Real/Fake và lỗi hai chiều |
| `routing.tsv` | Real→Fake, Fake→Real, lỗi subtype sau gate Fake đúng; chất lượng child trên mọi Fake thật và trên Fake được route đúng |
| `subtype_detection.tsv`, `subtype_summary.tsv` | Gate recall và end-to-end recall theo sáu lớp; child recall trên từng subtype Fake thật |
| `audit_metrics_by_seed.tsv`, `audit_summary.tsv` | Cùng flat/cascade trên phần audit dev đã định trước (mặc định 1.500 ID); không chọn lại threshold bằng audit |
| `oracle_diagnostic.json` | Dùng nhãn binary thật để tách lỗi gate khỏi child; không nằm trong bảng phương pháp chính |
| `replay_scores.npz`, `linear_models.npz` | Gate 2 lớp, child 5 lớp, flat 6 lớp, hard prediction và hệ số/classes để replay |
| `support.json`, `cohort.json`, `frozen_plan.json`, `run.json`, `completed.json` | ID, metadata, code/specification checksum và completion |

Ba nguồn lỗi tách rời cộng đúng số dự đoán sáu lớp sai: Real bị route Fake + Fake bị route Real + sai subtype khi Fake được route đúng. Recall end-to-end của một subtype Fake không thể vượt recall Fake của gate trên subtype đó. Không dùng conditional child accuracy thay end-to-end accuracy.

Full dev có cả selection đã dùng chọn ngưỡng; bảng audit tách riêng nhưng toàn bộ dev đã phục vụ phát triển. Cả hai đều **không phải xác nhận test độc lập**. Oracle chỉ chẩn đoán; accuracy oracle là trần khi cố định dự đoán child, không khẳng định mọi metric oracle là cận trên. Mean/sample std qua 5 support seed không phải CI.

## Kiểm tra code và giới hạn hiện tại

Các kết quả test code dưới đây thuộc giai đoạn triển khai trước khi chạy GPU. Trạng thái mới: smoke version 2 và full dev version 2 đã audit PASS; số liệu và quyết định ở [review full dev](FAKEDDIT_FEWSHOT_CASCADE_MIN5_REVIEW.md).

```sh
venv/bin/python -B -m unittest discover -s tests -p test_fakeddit_fewshot_cascade.py -q
venv/bin/python -B -m unittest discover -s tests -p test_cascade_launcher.py -q
```

Các test tổng hợp kiểm ngân sách fit 384/320/192, ba taxonomy head, hard gate/ties, label independence, replay hệ số cả ba head, decomposition lỗi, subtype recall, audit 90.000 prediction, oracle tách khỏi ranking, guard trước model, unload khi lỗi, smoke giữ support và notebook chỉ gắn train/dev.

Kết quả kiểm tra: 71 tests liên quan passed (7 cascade, 1 launcher cascade, 8 Task 1, 3 support, 9 protocol, 8 runner Task 1, 8 threshold và 27 nền tảng/launcher). Lượt đo stdlib trace/dis trên 6 test cascade trước khi thêm case decomposition bổ sung đạt 228/241 dòng (94,6%) và passed; không phải branch coverage hoặc coverage subprocess. Audit trên 90.000 prediction tổng hợp passed; replay sigmoid/softmax từ hệ số cả ba head khớp. Syntax/whitespace và biên dịch notebook smoke/full passed. Không chạy type checker/linter riêng vì môi trường chưa có công cụ tương ứng; không cài thêm dependency hoặc chạy model thật.

Smoke ban đầu được chuẩn bị offline tại `results/_system/launches/final/fakeddit_fewshot_cascade_min5_794869940ff3`. Kernel `nguyenhung3105/vlm-min5-fakeddit-fewshot-cascade-min5-smoke`, code dataset ban đầu `nguyenhung3105/vlm-final-code-4f659f4c30a5`. Đã kiểm checksum ZIP/snapshot, compile notebook, xác minh chỉ hai Input train/dev và không chứa dataset/weights. Sau đó smoke version 1 được hủy, smoke version 2 và full dev version 2 đã hoàn tất ở các stage mới. Bản tài liệu trong bundle phản ánh thời điểm đóng gói; trạng thái hiện tại nằm trong review riêng, không sửa artifact nguồn.

TDD đã ghi nhận RED do thiếu module/launcher, thiếu output subtype và thông báo launcher sai; GREEN sau triển khai. Không commit checkpoint vì quy tắc dự án chỉ cho commit khi người dùng yêu cầu.

Full dev cho thấy cascade cải thiện accuracy/macro-F1 so flat cùng support legacy, nhưng chưa vượt flat balanced mạnh nhất về macro-F1 trung bình. Bước tiếp theo là chốt bộ ứng viên và tạo protocol **hai tầng**, rồi mới xác nhận test end-to-end. Protocol Task 1 một tầng trước đây không thay thế protocol đó và đã stale do code fingerprint đổi; không niêm phong nó để bỏ qua bước này.

Tự rà soát tại thời điểm trước khi chạy GPU theo `agent-self-evaluation`: chính xác 4/5 (có replay/audit tổng hợp); đầy đủ 3/5 (khi đó chưa có kết quả GPU dev); rõ ràng 4/5; khả thi 4/5; súc tích 4/5. Trung bình lịch sử 3,8/5. Đánh giá cập nhật với bằng chứng GPU và giới hạn test nằm trong review full dev; không dùng test tổng hợp thay số đo thực tế.
