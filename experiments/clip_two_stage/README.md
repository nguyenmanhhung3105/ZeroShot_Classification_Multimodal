# CLIP hai tầng: runner local và Kaggle

## Bước tiếp theo: kiểm chứng độc lập

Đã có công cụ xác thực artifact (local hoặc ZIP Kaggle), tính lại argmax và metric,
bootstrap ghép cặp 2.000 lần, McNemar thăm dò và fixed/broken theo sự kiện:

```sh
venv/bin/python -B experiments/clip_two_stage/analyze.py --run-dir results/clip_two_stage/crisismmd/humanitarian/samples_3000/metaclip2_vith14/humanitarian_a1d64a92adeb
```

Kết quả seed 0: Delta Macro-F1 +4,89 điểm %, khoảng bootstrap theo mẫu 95%
[+2,71; +6,89] điểm %. Cả bảy sự kiện đều có số sửa đúng lớn hơn số làm sai.
Đây vẫn là thăm dò, các ảnh có thể phụ thuộc theo tweet/sự kiện và cohort đã
tham gia chọn phương pháp. Không diễn giải p-value thành bằng chứng test độc lập.

Runner hỗ trợ `evaluation_scope: heldout` trong settings; job launcher có thể đặt
cùng trường. Bắt buộc cấu hình `experiment_config` riêng trỏ processed dev/test,
không đổi tên train thành test. Launcher từ chối train trước khi gửi job, runner
kiểm lại từng hàng trước tải model. Các bộ có tên source_split dev/val/validate/
validation/test được nhận; kiểm định cuối nên là test chưa dùng chọn cấu hình.

`configs/prompts/crisismmd_development_exclusion.json` lưu ID/hash ảnh và ID tweet
của 3.000 mẫu đã dùng phát triển (2.765 tweet), không chứa nhãn. Heldout từ chối
trùng bất kỳ loại nào; không tự bỏ ảnh trùng rồi báo đủ mẫu. Không phát hiện ảnh
gần trùng hoặc thông tin trùng theo sự kiện. File được đóng gói theo thư mục
configs/prompts hiện có và hash được ghi vào manifest.

Kết quả heldout tách dưới `results/clip_two_stage_heldout/`; khóa cohort tách theo
split để không xung đột với các lượt train trước đó. Không tự chạy lại 3.000 ảnh
train dưới tên heldout. Số mẫu test lấy theo tập thực tế, không nhân bản để đủ
3.000. Hiện mới tìm thấy processed train CrisisMMD; chưa có processed validation/
test để chạy. Quy tắc dự án không cho truy cập raw nên chưa tự tạo tập mới từ raw.

Khi có processed test, cần cùng schema của cohort chính: image_id, image_path,
label_informative, label_humanitarian, source_split, label_source=image; giữ nhãn
ảnh và đủ taxonomy. Phải có đường dẫn ảnh tồn tại dưới data/processed. Cấu hình
hai task trỏ cùng TSV/ảnh. Chỉ sau khi kiểm tra nguồn dữ liệu và trùng lặp mới
chạy heldout, không dùng nhãn test để sửa prompt, seed hoặc quy tắc gate.

## Sửa lỗi lượt Kaggle 120

Lượt `5a80da280a96` tải MetaCLIP2 xong nhưng dừng trước encode ảnh vì
`AttributeError: 'CLIP' object has no attribute 'dtype'`. OpenAI CLIP cung cấp
property này, OpenCLIP không bắt buộc có. Runner đã lấy dtype từ parameter
khi thiếu property; chỉ thay metadata, không đổi prompt, cohort hay phép tính.
64 kiểm thử liên quan đạt, gồm một lượt end-to-end bằng backend giả có cấu trúc
model không có `.dtype`, kiểm tra lưu summary/manifest và giải phóng model.
Đã đọc log Kaggle 120; không gửi lại job trong lượt sửa này. Lượt cũ đã kết thúc
lỗi, resume-stage không khởi động lại được; chạy lệnh launcher mới sau khi sửa.
Các cảnh báo mistune/nbconvert ở cuối log không phải nguyên nhân dừng runner.

Mục đích: so trực tiếp phân loại tám lớp và quy tắc hai tầng trên cùng ảnh,
checkpoint, prompt tầng 2 và seed. Tầng 1 dùng semantic informativeness. Nếu
argmax là not_informative, trả not_humanitarian; nếu informative, chọn argmax
trong bảy nhóm humanitarian còn lại. Không dùng nhãn thật khi quyết định.

Giữ bank prompt tám lớp khi sinh semantic_waffle, rồi giới hạn tập ứng viên lúc
quyết định. Không sinh lại bank chỉ bảy lớp, vì độ dài tên lớp ảnh hưởng hậu tố
ngẫu nhiên, làm mất đối chứng với log trước đó. Tầng 1 vẫn có thể chặn nhầm ảnh;
log riêng gate_false_negative/gate_false_positive, fixed/broken và F1 từng lớp.

Đây là **image-only**, không đưa tweet vào cả hai tầng. Một model được tải một
lần và mỗi ảnh encode một lần. Điểm tầng 2 được tính cho mọi ảnh để có baseline
một tầng trong cùng run; không tuyên bố tiết kiệm chi phí nhờ bỏ tầng 2.
Cosine trung bình, model frozen, không tune ngưỡng hoặc train bằng nhãn.

Đã prepare prompt MetaCLIP2 không đọc dataset/checkpoint; smoke thật OpenAI
CLIP ViT-B/32 CPU trên năm ảnh; 66 kiểm thử liên quan đạt. Chưa chạy runner
MetaCLIP2 trên GPU hoặc cohort độc lập. Log smoke:
`results/clip_two_stage/crisismmd/humanitarian/samples_5/openai_clip_vit_b_32/humanitarian_81e25b9b41e6/`.
Đã sửa lỗi kiểu NumPy khi lưu bộ đếm gate trong lượt smoke đầu; lượt thành công
lưu tách biệt, không ghi đè run lỗi.

## Local

Chuẩn bị, không đọc dataset hoặc tải model:

```sh
venv/bin/python -B experiments/clip_two_stage/run.py --prepare-only
```

Chạy model thật (MetaCLIP2 cần tài nguyên lớn; smoke local đã dùng ViT-B/32):

```sh
venv/bin/python -B experiments/clip_two_stage/run.py --backbone ViT-B/32 --max-samples 5
```

## Kaggle

Chạy 5 ảnh MetaCLIP2 trước:

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/clip_two_stage/launcher.yaml --job two_stage --seed 0 --max-samples 5 --use-kaggle true
```

Sau khi smoke thành công, đánh giá 3.000 ảnh nguồn train đã dùng thăm dò:

```sh
venv/bin/python -B scripts/launch_runs.py --launcher-config experiments/clip_two_stage/launcher.yaml --job two_stage --seed 0 --max-samples 3000 --use-kaggle true
```

Muốn chạy local qua cùng launcher đổi `--use-kaggle false`. Mỗi lần một job,
một seed; không chạy đồng thời cùng kernel. Nếu job đã gửi mà mất kết nối,
dùng `--resume-stage` được in, không gửi lại. Không thêm --method hoặc --model;
đổi backbone bằng --backbone nếu cần kiểm tra kỹ thuật.

Kết quả: `results/clip_two_stage/crisismmd/humanitarian/samples_N/<model>/<run>/`.
Có frozen plan từng task, cohort có SHA256 ảnh, manifest, predictions.tsv,
metrics.json và summary.tsv gồm cả single_stage/two_stage. Quy tắc runner được
đối chiếu lại trên score 3.000 mẫu ba seed trước đó, tái tạo đúng các Macro-F1
đã báo; điều này chưa thay thế kiểm chứng inference MetaCLIP2 thật mới.

Kỳ vọng từ log cũ: humanitarian hai tầng trung bình Macro-F1 50,05%, Accuracy
68,32% qua ba seed, so với một tầng 45,10%/57,70%. Đây là kết quả phát triển,
không bảo đảm trên tập mới. Cần khóa cấu hình rồi đánh giá tập processed độc
lập chưa dùng chọn phương pháp. Không nhận hai tầng là ý tưởng mới toàn cầu.
