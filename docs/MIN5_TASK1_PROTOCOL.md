# Protocol khóa few-shot Task 1 min5

Phạm vi hiện tại đã chuyển sang [hoàn thiện cascade hai tầng trên dev](FAKEDDIT_FEWSHOT_CASCADE_MIN5.md). Các draft Task 1 v1/v2/v3 dưới đây là lịch sử; đã stale sau thay đổi code cascade, không dùng để mở test. Khi đã có full dev hai tầng, cần protocol end-to-end mới, không niêm phong lock Task 1 để thay thế.

Phạm vi: thành phần Real/Fake một tầng, hai policy support và các quyết định default/tuned đã phân tích trên dev. Đã có runner riêng `scripts/run_min5_test.py` tích hợp guard; chưa niêm phong hoặc chạy test thật và không chứng minh hoàn tất cascade.

Không sửa `protocol.py` legacy, cấu hình thí nghiệm hoặc run nguồn. Các lệnh chuẩn bị/validate/prepare-only dưới đây chỉ đọc artifact và source/config, không đọc dataset/test, feature cache hoặc trọng số; không gọi Kaggle. Runner chỉ đọc train/test và nạp backbone khi chạy thực tế với protocol đã niêm phong.

## 1. Chuẩn bị draft

```sh
venv/bin/python -B scripts/prepare_min5_protocol.py prepare \
  --source-run results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run \
  --threshold-run results/final/fewshot_support_thresholds/recall_tradeoff/dev_1b4a7accc90c \
  --output results/final/protocols/min5_task1_draft_20261116_v3.json
```

Đã chạy thành công lệnh này trong bước triển khai. Không chạy lại cùng output: script dùng exclusive create, từ chối ghi đè. Khi cần chuẩn bị bản mới, đặt tên output mới.

```sh
venv/bin/python -B scripts/prepare_min5_protocol.py validate \
  --protocol results/final/protocols/min5_task1_draft_20261116_v3.json \
  --allow-draft
```

Bỏ `--allow-draft` sẽ bị từ chối vì draft không phải test lock. Bản hiện tại giữ 20 cấu hình: hai policy × năm seed × (default và target selection 87%), chỉ modality ảnh–tiêu đề. Chưa chọn winner theo audit.

## 2. Nội dung và xác minh

- Receipt/run và checksum các artifact thực sự sử dụng; không xác minh lại feature/model cache không được đọc.
- Đúng source full dev min5 3.000 mẫu, label mapping 0=Real/1=Fake; review phải trỏ đúng SHA256 source run.
- Support nguyên bản: đủ hai policy/năm seed, 384 ID duy nhất/seed, đều thuộc train; kiểm nhãn/quota sáu lớp và support fingerprint.
- Danh sách loại trừ của toàn bộ 3.000 train và 3.000 dev: ID, hash ảnh, hash tiêu đề; kiểm train/dev không trùng trên các khóa này.
- Chia selection/audit được tái tạo từ ID/subtype/seed/fraction. Ngưỡng được tính lại chỉ từ posterior selection, không chỉ tin file threshold đã lưu.
- Khóa specification nguồn, feature convention, metadata backbone nguồn và support ID; ghi rõ code lịch sử của source khác với code được khóa hiện tại.
- Fingerprint code hiện tại, script chuẩn bị, requirements, cấu hình few-shot/support, nội dung file config và phiên bản thư viện. Không nạp backbone. Nếu thiếu thư viện, fingerprint ghi `null`; đây không phải xác nhận môi trường đủ điều kiện inference.
- Metric chính macro-F1; báo thêm accuracy, BA, recall Real/Fake. Bộ ứng viên giữ nguyên cả hai policy/default/tuned, không hứa target recall sẽ đạt trên test.

File `project_min5.yaml` hiện vẫn có slug test v2 cũ. Runner mới **không sử dụng slug đó để mở test**: yêu cầu train/test root tường minh, xác minh hash manifest trước loader và không dùng `locate_data` của pipeline legacy. Script runner cũng nằm trong fingerprint.

## 3. Niêm phong riêng Task 1 khi đã chốt phạm vi

Sau khi chốt Task 1 và biết SHA256 của manifest test min5 từ quy trình chuẩn bị đã kiểm tra, dùng CLI `seal` với `--draft`, `--test-manifest-sha256` và `--output` mới. Xem cú pháp:

```sh
venv/bin/python -B scripts/prepare_min5_protocol.py seal --help
```

Chưa thực hiện seal và không tự đọc manifest test để điền tham số. Seal không mở test; nó kiểm tra fingerprint và chuyển trạng thái từ `draft` thành `sealed_task1`, ghi hash test được cung cấp. Hash đúng định dạng chưa chứng minh provenance; runner vẫn phải kiểm split nguồn và dataset.

Nếu đồ án bắt buộc hai tầng, giữ protocol này như thành phần tầng một; phải phát triển tầng hai trên dev rồi chuẩn bị protocol end-to-end khác trước đợt xác nhận test của toàn hệ thống. Không dùng `sealed_task1` thay bằng chứng cascade đã khóa.

## 4. Runner test đã tích hợp guard

Xem kế hoạch mà không đọc dataset/nạp model:

```sh
venv/bin/python -B scripts/run_min5_test.py \
  --protocol results/final/protocols/min5_task1_draft_20261116_v3.json \
  --prepare-only
```

Kết quả hiện tại: `protocol_state=draft`, `candidate_seed_groups=20`, `test_opened=false`, `executable=false`. Draft được dùng xem kế hoạch, không được thực thi inference. Tham số `--train-root` và `--test-root` bị bỏ qua trong prepare-only; không có smoke test.

Khi đã có protocol sealed và được yêu cầu chạy test, CLI cần `--protocol`, `--train-root` và `--test-root` tường minh; `--config` là tùy chọn cho file cùng nội dung đã khóa. Không có lựa chọn tự tìm dataset hoặc gửi job. Xem `venv/bin/python -B scripts/run_min5_test.py --help`.

Thứ tự runner thực hiện:

1. Gọi `min5_protocol.validate` trước khi đọc dataset hoặc nạp model, kiểm trạng thái sealed và code/config/dependency hiện tại. Config đổi nội dung dù chỉ đường dẫn cũng làm hash đổi; cần draft mới. Có thể cung cấp file config ở vị trí khác với cùng nội dung qua `settings_file` cho cả `validate` và `check_test_cohort`.
2. Xác minh manifest test đúng SHA256 đã khóa trước xử lý dữ liệu; gọi loader với split `test`, đúng 3.000 mẫu, không smoke.
3. Gọi `check_test_cohort(lock, rows, metadata)` trước model: kiểm lại fingerprint, split `test`, nguồn `test_public`, hash manifest và chống trùng với toàn bộ train/dev theo ID/hash ảnh/hash tiêu đề. Guard nhận metadata đã load, không tự đọc dataset.
4. Dùng đúng support/classifier/feature và ứng viên trong payload; ngưỡng cố định từ selection dev. Không fit bằng nhãn test, không tune lại ngưỡng, không chọn seed tốt nhất. Ghi checksum kết quả và phân biệt mean/std với CI.

Runner kiểm tra toàn bộ train khớp ID/hash đã khóa, giữ nguyên support ID/thứ tự/quota từng seed, chỉ encode hợp support một lần và test một lần. Cả hai policy dùng chung feature; mỗi policy/modality/seed chỉ fit một classifier rồi áp dụng các ngưỡng cố định. Dùng cùng hàm `predict`/feature/LogisticRegression như dev; nhãn test chỉ dùng đánh giá sau khi mọi posterior đã tính xong. Metadata backbone architecture/pretrained/tokenizer/context/dtype/image_size phải khớp source; tuple/list image_size được so theo biểu diễn JSON chuẩn. Backend được unload kể cả lỗi mã hóa hoặc metadata.

Output mới dưới `results/final/fakeddit_fewshot_test_min5/task1/test_<id>/`: summary/metrics/predictions/per-class/class-summary, support và cohort, posterior và hệ số head để replay, frozen plan/run/completion có checksum. Run ghi rõ `split=test`, `test_opened=true`, `stages=1` và hash protocol. Chưa có output test thật trong bước triển khai này.

Đây là runner standalone, **chưa tích hợp launcher Kaggle**. Fingerprint khóa phiên bản dependency chính xác, nên không mặc định protocol tạo trên máy local sẽ hợp lệ trong môi trường Kaggle khác; cần chuẩn bị lại draft ở môi trường đích rồi seal. Lock không tự chặn script khác không gọi guard.

## 5. Kiểm tra

```sh
venv/bin/python -B -m unittest discover -s tests -p test_min5_protocol.py -q
venv/bin/python -B -m unittest discover -s tests -p test_fewshot_support_thresholds.py -q
venv/bin/python -B -m unittest discover -s tests -p test_min5_test_runner.py -q
```

Tests protocol dùng artifact tổng hợp cho prepare/seal/validate, chống ghi đè, code/config drift, support sai/trùng/ngoài train, threshold đã bị đổi dù rehash review, split/hash test sai, train/dev/test overlap và CLI từ chối draft khi yêu cầu sealed. Không mở dữ liệu test thật.

Kết quả giai đoạn protocol ban đầu (trước runner): 7 tests protocol và 8 tests phân tích ngưỡng passed. Line coverage module protocol lúc đó bằng stdlib trace/dis: 174/190 dòng (91,6%), không phải branch coverage và không tính coverage subprocess CLI.

Tự rà soát sau khi tích hợp runner theo `agent-self-evaluation`: chính xác 4/5 (replay và audit tổng hợp khớp, chưa kiểm inference thật); đầy đủ 4/5 (runner standalone hoàn tất, chưa có launcher Kaggle); rõ ràng 4/5 (CLI tách prepare-only/thực thi, cần đọc quy tắc draft/seal); khả thi 4/5 (draft v3 và prepare-only đã chạy, thực thi còn cần manifest/niêm phong); súc tích 4/5 (tái dùng predict/metric, payload dài vì giữ ID/hash chống trùng). Trung bình 4,0/5. Cải thiện tiếp theo theo tác động: chốt phạm vi một/hai tầng, niêm phong đúng môi trường sau khi biết manifest test, rồi mới xác nhận test theo yêu cầu. Đây không phải xác nhận chất lượng mô hình trên test.

## Rà soát bổ sung

Phát hiện và sửa hai lỗi: `check_test_cohort` trước đây chưa kiểm đầy đủ version/kind/test_opened/hash test như `validate`; và chưa nhận `settings_file` thay thế khi config được chuyển vị trí. Hai entry point hiện dùng chung `_validate_lock`. Test mới đã tái hiện lỗi trước khi sửa, gồm protocol sai schema nhưng digest nhất quán và config cùng nội dung ở vị trí khác.

Bản nháp hiện hành sau khi thêm runner là `min5_task1_draft_20261116_v3.json`; các bản trước được giữ để truy vết và đã stale vì code fingerprint thay đổi. Chưa seal, chưa mở test thật. Runner được kiểm bằng dữ liệu/encoder tổng hợp: ngưỡng cố định, replay hệ số head, đổi nhãn test không đổi posterior, guard trước dataset/model, unload khi lỗi và prepare-only không đụng dataset. Audit độc lập trên prediction tổng hợp tính lại metric/mean/std.

Kiểm tra sau tích hợp: 8 tests runner, 9 tests protocol, 8 tests few-shot Task 1 và 3 tests support đều passed (28 tests). Audit end-to-end tổng hợp passed trên 60.000 prediction của 20 tổ hợp method–seed; replay posterior từ hệ số head khớp. Syntax/whitespace, validate draft v3 và prepare-only thực tế passed. Không chạy type checker/linter riêng, model thật hoặc gửi job; cấu hình thí nghiệm và run nguồn giữ nguyên.

Line coverage runner đo bằng stdlib trace/dis trên bộ 8 tests đạt 136/147 dòng (92,5%); lượt đo cũng passed. Đây không phải branch coverage, không tính các dòng thực thi bên trong subprocess CLI.
