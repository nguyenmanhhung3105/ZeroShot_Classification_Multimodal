# Protocol và runner test hai tầng min5

Đã niêm phong protocol riêng cho toàn pipeline sáu lớp tại `configs/cascade_min5_test_lock.json`, không dùng lock binary Task 1. Chưa đọc test thật hoặc chạy inference test. Lock dùng runtime Kaggle đã xác minh; `runtime_matches=false` khi xem trên Mac là trạng thái dự kiến.

Protocol SHA256: `bf26da192f608e6d5be07ee74b07e64d09d24ec3857cf5c1a19c1df11cbf0547`. Manifest test SHA256 do người dùng cung cấp: `1aa002cb23a12a90c8bbe0bbaff5facca1d8764dc4005dc8d78f46bffce23ed4`. Runtime từ `results/final/cascade_runtime_capture_v1/cascade_target_runtime.json`, receipt và bundle đã kiểm checksum. Phần dưới giữ cả quy trình draft để truy vết/tái lập, không tạo lại hoặc ghi đè lock hiện hành.

## Bộ phương pháp được khóa

Nguồn dev: `results/final/kaggle/fakeddit_fewshot_cascade_min5_b7c54cfcca4a/run/` (full dev version 2 đã audit). Nguồn train/support: `results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run/`.

Giữ đúng sáu nhánh đã chạy:

1. `legacy_6way_flat`
2. `legacy_6way_hierarchy_default`
3. `legacy_6way_hierarchy_target_0.87`
4. `balanced_binary_flat`
5. `balanced_binary_hierarchy_default`
6. `balanced_binary_hierarchy_target_0.87`

Năm seed 0–4, 384 support/seed/policy, cùng ID/thứ tự/quota từ dev, frozen MetaCLIP2 ảnh–tiêu đề và các head LogisticRegression C=1 balanced. Không chọn thêm seed, ngưỡng hoặc head từ test. Target 0,87 là recall Fake trên selection dev, không phải probability threshold=0,87.

Metric chính **macro-F1 sáu lớp**; báo thêm accuracy, BA, metric theo lớp, binary gate và routing/child. Oracle chỉ chẩn đoán sau inference, không thuộc sáu nhánh hoặc xếp hạng.

## Tạo và xem draft

Draft đã tạo bằng lệnh sau; không chạy lại cùng output vì file được tạo exclusive:

```sh
venv/bin/python -B scripts/prepare_cascade_protocol.py prepare \
  --dev-run results/final/kaggle/fakeddit_fewshot_cascade_min5_b7c54cfcca4a/run \
  --train-source results/final/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run \
  --output results/final/protocols/cascade_min5_test_draft_20261116.json
```

Kiểm kế hoạch an toàn, không truy cập train/test root hoặc nạp model:

```sh
venv/bin/python -B scripts/run_cascade_test.py \
  --protocol results/final/protocols/cascade_min5_test_draft_20261116.json \
  --prepare-only
```

Kết quả hiện tại: `protocol_state=draft`, `stages=2`, `samples=3000`, `method_seed_groups=30`, `test_opened=false`, `executable=false`. Đây là trạng thái đúng trước niêm phong.

Prepare kiểm checksum các artifact dev/train được dùng, full dev 3.000 mẫu/30 method–seed, sáu nhánh đúng taxonomy, cohort/support đúng snapshot và các file inference cốt lõi vẫn khớp code đã chạy dev. Các module wrapper test mới không buộc chạy lại dev khi code inference cốt lõi giữ nguyên. Không đọc feature cache hoặc hệ số model nguồn để tạo draft.

Payload lưu recipe, source hash, support, toàn bộ 3.000 ID/hash ảnh/hash tiêu đề train và 3.000 dev, fingerprint code/config/scripts/requirements/snapshot. Recipe trong payload phải khớp cả hash lẫn file snapshot ngoài; chỉ sửa ngưỡng rồi tính lại digest protocol vẫn bị từ chối.

## Khóa runtime của môi trường thực thi

`requirements.txt` hiện có nhiều khoảng phiên bản; chỉ hash file requirements không đủ cố định môi trường. Runtime cần được ghi nhận **trong môi trường sẽ chạy test**, sau khi cài requirements và trước khi mở test. Lệnh này chỉ đọc metadata Python/package:

```sh
python -B scripts/prepare_cascade_protocol.py capture-runtime \
  --output /kaggle/working/cascade_target_runtime.json
```

Ví dụ trên dùng Python của môi trường Kaggle và giả định đang ở thư mục project đã giải nén. Nếu chạy test local, dùng Python local tương ứng. Không coi runtime Mac và Kaggle CUDA là cùng môi trường mặc định.

File runtime chứa Python và phiên bản chính xác của 15 thư viện trong requirements. Capture/seal từ chối thư viện thiếu hoặc không thỏa ràng buộc requirements (ví dụ `open_clip_torch` khác 3.3.0). File không chứa credential. Runner kiểm lại phiên bản trước mọi truy cập dataset/model.

Giới hạn lịch sử: dev cũ không ghi đầy đủ package versions. Protocol ghi rõ điều này, giữ code/recipe/metadata backbone dev và khóa runtime đích mới trước test; không khẳng định đã tái lập tuyệt đối mọi dependency của dev chỉ từ file requirements.

## Niêm phong khi đủ metadata

Cần SHA256 của **manifest test min5 đã chuẩn bị đúng nguồn**, cùng file runtime đích. Chưa tự đọc manifest/dữ liệu test để điền giá trị này trong bước triển khai. Không dùng hash train/dev hoặc cohort v2 cũ.

Cú pháp niêm phong, chỉ thực hiện khi các giá trị đã được xác nhận:

```sh
venv/bin/python -B scripts/prepare_cascade_protocol.py seal \
  --draft results/final/protocols/cascade_min5_test_draft_20261116.json \
  --test-manifest-sha256 "$CASCADE_TEST_MANIFEST_SHA256" \
  --runtime-file /duong/dan/cascade_target_runtime.json \
  --output configs/cascade_min5_test_lock.json
```

Biến `CASCADE_TEST_MANIFEST_SHA256` phải là đúng 64 ký tự hex; đường dẫn runtime là ví dụ cần thay. Seal chuyển sang `sealed_cascade`, không mở test. Không ghi đè file đã có; code/config/recipe đổi thì cần draft và seal mới. Seal hiện hành đã hoàn tất bằng hash người dùng cung cấp và runtime Kaggle, không tự đọc manifest để chọn phương án.

Đường dẫn config/snapshot bên trong repo được lưu tương đối để chuyển project sang Kaggle. Có thể dùng `--config`/`--spec` ở vị trí khác với cùng nội dung; hash vẫn phải khớp. Fingerprint không khóa đường dẫn máy cá nhân làm điều kiện inference.

## Runner thực thi

`scripts/run_cascade_test.py` là runner standalone, chưa tích hợp tự gửi job Kaggle. Khi được yêu cầu chạy test thật, cần protocol sealed cùng `--train-root` và `--test-root` tường minh. Không tự tìm slug test v2 và không có lựa chọn smoke/giới hạn test.

Thứ tự guard:

1. Schema hai tầng/sealed, code/config/recipe/runtime phải khớp trước khi đọc dataset.
2. So hash manifest train/test đã niêm phong trước loader.
3. Train khớp toàn bộ identity nguồn, support train-only đủ quota; test đúng split `test`, nguồn `test_public`, đủ 3.000 ID duy nhất và taxonomy 0–5.
4. Chặn test trùng toàn bộ train/dev theo ID, hash ảnh bytes hoặc hash tiêu đề trước model, không chỉ đối chiếu hợp support.
5. Nạp backbone một lần, metadata architecture/pretrained/tokenizer/context/dtype/image_size và device nếu nguồn khai báo phải khớp dev. Encode hợp support và test một lượt mỗi tập.
6. Gọi đúng `cascade.infer` đã chạy dev. Nhãn test chỉ vào `cascade.evaluate` sau khi mọi gate/child/flat posterior đã có; không tìm lại ngưỡng.

Output dưới `results/final/cascade_test_min5/cascade/test_<id>/`, có 30 method–seed/90.000 prediction, summary, comparisons, routing, gate, subtype, oracle tách riêng, posterior/hệ số head và checksum. Không xuất `audit_summary.tsv` của partition dev trên test. Model được unload kể cả lỗi encoding/metadata. Mean/sample std theo seed vẫn không phải CI.

## Kiểm tra

```sh
venv/bin/python -B -m unittest discover -s tests -p test_cascade_test_protocol.py -q
```

Tests dùng full artifact/encoder tổng hợp: six-method suite, no-overwrite, schema/threshold/exclusion/config/code/runtime drift, sai manifest, overlap với toàn bộ train/dev, runtime thiếu/không thỏa requirements, metadata sai, unload khi lỗi, nhãn test không ảnh hưởng score/prediction và CLI prepare-only không chạm root. Audit độc lập trên output test tổng hợp tính lại 90.000 prediction/metric/mean/std.

Không mở dataset test, chạy inference thật, upload dữ liệu, gửi job hoặc thay cấu hình thí nghiệm trong lượt triển khai. Runtime đích và manifest test đã được dùng để niêm phong lock hiện hành như ghi ở đầu tài liệu. Bước tiếp theo là xác nhận môi trường thực thi khớp lock trước lượt test được phê duyệt; kiểm tổng hợp không thay thế xác nhận trên test độc lập.

## Kết quả xác minh triển khai

- 9 tests mới và 52 regression tests passed (7 cascade dev, 9 protocol Task 1, 8 runner Task 1, 1 launcher cascade, 27 nền tảng/launcher), tổng 61.
- Line coverage bằng tracer theo file và bytecode line map: protocol 152/175 dòng (86,9%), runner 89/94 dòng (94,7%); lượt đo 9 tests passed. Không phải branch coverage và không tính subprocess CLI.
- Audit độc lập trên test tổng hợp: 30 method–seed, 90.000 prediction; metric/mean/std và checksum khớp. Đổi nhãn test không làm đổi posterior hoặc hard prediction.
- Syntax/whitespace, prepare draft thật, validate draft và prepare-only CLI passed. Không chạy type checker/linter riêng.
- Capture runtime CLI tại local đã được thử trong thư mục tạm: bị từ chối đúng vì `sentencepiece` chưa có metadata cài đặt. Không ghi runtime giả hoặc tự cài dependency. Điều này không ảnh hưởng chuẩn bị draft; runtime để seal cần được lấy trong môi trường đích đủ requirements.

TDD bắt đầu bằng RED do module chưa có, rồi thêm kiểm tra runtime sai kiểu và sửa trước GREEN. Không commit/stage/push vì quy tắc dự án chỉ cho commit khi được yêu cầu.

Tự rà soát theo `agent-self-evaluation`: chính xác 4/5 (guard/inference/audit tổng hợp khớp, chưa kiểm test thật); đầy đủ 4/5 (đủ protocol và runner standalone, chưa tích hợp launcher gửi test); rõ ràng 4/5 (tách draft/runtime/seal, cần đọc hướng dẫn môi trường đích); khả thi 4/5 (draft/prepare-only đã chạy, seal cần manifest và runtime); súc tích 4/5 (tái dùng infer/evaluate dev, giữ schema test riêng để tránh nhầm binary). Trung bình 4,0/5. Ưu tiên tiếp theo: capture runtime đích, xác nhận manifest/cohort test min5 rồi seal và duyệt lượt xác nhận. Người dùng có thể đối chiếu qua tests và artifact; đây không phải chứng nhận chất lượng trên test.
