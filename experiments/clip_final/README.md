# Kiểm chứng test cho ứng viên majority + DN-unit

## Mục tiêu đã khóa

Đo trên cùng ảnh test ba phương án: hai tầng gốc, majority normalized CLIP
+ EM + StatA, và majority + DN-unit ở tầng chi tiết. Giữ MetaCLIP2,
image-only, prompt seed 0, grouping seed 0/1/2; không chọn seed tốt nhất.
Metric chính là Macro-F1 tám nhãn; báo cả Accuracy, metric hai nhãn và từng
lớp, đặc biệt recall cứu trợ. Nhãn ảnh là ground truth, tweet không là input.

`protocol.py` tạo `configs/prompts/crisismmd_final_dn_unit.json` **từ artifact
train/dev đã có**, gồm mean ảnh train không nhãn, vector text, bank prompt,
tham số, hash code và hợp các ID/tweet/hash ảnh của hai cohort phát triển.
Không dùng vector ảnh train làm support có nhãn, không tính lại mean từ test.
DN-unit là adaptation trên vector unit, không phải tái lập DN gốc dùng
encoder output chưa chuẩn hóa.

Sau khi khóa, không thay model, prompt, coefficient, query size hoặc chọn
method/seed theo kết quả test. Báo đủ cả ba phương án và đánh đổi. Nếu kết
quả không tăng, ghi nhận giả thuyết chưa được xác nhận; không dùng test đó
để tiếp tục tối ưu rồi gọi nó là test độc lập.

## Chuẩn bị và chạy

Agent đã tạo frozen plan từ artifact và kiểm tra code bằng dữ liệu giả.
Theo AGENTS.md, agent không chạy script chuẩn bị trên raw thật; phần dưới
để người dùng tự chạy. Không có job Kaggle nào được gửi trong bước chuẩn bị.

Từ thư mục gốc, chạy lần lượt:

```sh
venv/bin/python -B experiments/clip_final/prepare_test.py

venv/bin/python -B scripts/launch_runs.py \
  --launcher-config experiments/clip_final/launcher_test.yaml \
  --job final_dn_unit_test \
  --use-kaggle true
```

Script thứ nhất đọc **đúng hai annotation test**:

- `task_informative_text_img_test.tsv`
- `task_humanitarian_text_img_test.tsv`

và các ảnh tương ứng trong `data/raw/CrisisMMD_v2.0/`. Không mở annotation
train/dev. Nó lấy giao image ID giữa hai task test để giữ cả hai nhãn thật,
không suy informative từ humanitarian. Không dùng nhãn để chọn thứ tự mẫu.
Loại trùng với cả train/dev bằng image ID, tweet ID và SHA-256 ảnh, loại
ảnh hỏng/thiếu và hash ảnh lặp trong test, ghi rõ mọi lý do.

**Tối đa 3.000 ảnh; nếu giao test hợp lệ ít hơn thì dùng toàn bộ số đó.**
TSV, manifest, cấu hình và launcher được tạo theo số mẫu thực. Không thêm
`--max-samples 3000` vào lệnh Kaggle, không lấy bù từ train/dev hoặc nhân đôi
ảnh. Không ghi đè đầu ra có sẵn. Bộ test này là giao hai task, không đại diện
cho toàn bộ test humanitarian; manifest ghi số mẫu ngoài giao và support.

Nếu frozen plan chưa có trong một bản checkout khác, tạo trước bằng:

```sh
venv/bin/python -B experiments/clip_final/protocol.py
```

Việc này cần các artifact train/dev được ghi trong code, không đọc raw.
Không tạo lại frozen plan sau khi xem metric test để thay đổi protocol.

## Luồng Kaggle và đầu ra

Một job encode mỗi ảnh một lần rồi export feature; unload backbone trước
khi chạy các phép kết hợp trên CPU. Không gửi ba job riêng cho ba đối chứng.

- Worker kiểm tra checksum TSV, protocol ID, đủ cohort, cả hai nhãn so với
  manifest trước inference. Gói bounded gồm manifest; gói code gồm frozen
  plan JSON, không yêu cầu mang các artifact train/dev cũ lên Kaggle.
- Runner kiểm tra bank prompt đã khóa, code hash và loại trùng train/dev
  trước khi load model. Nhãn test không chuyển vào hàm `predict_locked`.
- Sau export, tự gọi `evaluate.py`; mean DN vẫn từ frozen plan. Majority
  được tính lại trên query test theo đúng ba cách chia nhóm đã khóa.
- `summary.tsv` chính thêm `majority_mean` và `majority_dn_unit_mean` để
  thấy kết quả trên Kaggle. Các dòng single-stage/margin của runner gốc chỉ
  là diagnostic có sẵn, không dùng để chọn phương án sau test.
- `final_dn_unit_analysis/analysis.json`: metric 2/8 nhãn từng phương án,
  từng seed và mean. `final_dn_unit_summary.tsv` trong thư mục này chứa đủ
  ba phương án chính. TSV lưu prediction/ID/batch; manifest ghi hash output
  và fingerprint frozen plan. Binary majority và DN-unit giống nhau vì chỉ
  thay tầng chi tiết.
- Trạng thái `final_dn_unit_analysis_status` phải là `complete`; chỉ trạng
  thái export feature complete chưa chứng minh phần đánh giá mới đã xong.

Nếu launcher đã báo gửi job rồi mất kết nối, dùng đúng `--resume-stage`
được in ra; không gửi lại lệnh job. Resume chỉ chờ/tải lượt đã gửi.

Tái kiểm tra từ feature đã tải về, không chạy backbone:

```sh
venv/bin/python -B experiments/clip_final/evaluate.py --source-run PATH_TO_TEST_RUN
```

Loader mặc định của các thí nghiệm thăm dò vẫn từ chối test. Chỉ entry này
cho phép split test sau khi xác minh frozen plan. Không chạy sweep trên test.

## Kiểm chứng và giới hạn

Unit/integration test dùng fixture tạm, không đọc dataset thật. Kiểm tra
loại trùng ID/tweet/hash, giữ riêng nhãn hai task, chống đổi code/parameter,
đóng gói frozen plan/manifest, xác minh nhãn thứ hai, từ chối test ở loader
mặc định, ground truth thay metric nhưng không thay prediction, và chạy
hậu xử lý sau khi unload backbone. Các test launcher/worker và runner cũ
được chạy để bảo vệ luồng đã có.

Không nhận diện ảnh gần giống bằng perceptual hash và không tách sự kiện
thiên tai mới. Vì vậy "độc lập" ở đây là không trùng các ảnh/tweet/hash đã
ghi nhận trong hai cohort phát triển, không phải kiểm chứng sự kiện mới.
Nếu lớp hiếm không có mẫu trong giao test, vẫn giữ đủ tám nhãn và ghi rõ
support 0, không bỏ lớp để làm đẹp Macro-F1.
