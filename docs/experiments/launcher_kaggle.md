# Launcher chung cho các phép chạy local và Kaggle

Dùng `scripts/launch_runs.py` và chỉnh `configs/launcher.yaml`. Cờ `use_kaggle: false` chạy trên máy; `true` gửi job lên Kaggle GPU, chờ hoàn tất và tải kết quả. Có thể ghi đè bằng `--use-kaggle true/false`. Không lưu token trong YAML; đăng nhập một lần bằng `venv/bin/kaggle auth login`.

## Job trong YAML

Mỗi job xác định `kind`, `dataset`, `task`, `max_samples`; job không phải upstream còn chọn `models`. Upstream dùng `backbones` của `configs/research/upstream.yaml` để nạp OpenAI CLIP. Các cặp dữ liệu có sẵn:

| Dataset | Task | Kiểu nhãn |
|---|---|---|
| Fakeddit | `2way`, `6way` | Một nhãn |
| CrisisMMD | `informativeness`, `humanitarian` | Một nhãn |
| MM-IMDb | `genres` | Nhiều nhãn |

`kind: standard` dùng `src/run_experiment.py`; `prompt_variants` và `prompt_aggregations` chọn các thử nghiệm prompt của runner gốc. `kind: demo` chạy `src/demo.py` và xuất hình minh họa. `kind: adapted` lấy phương pháp từ `methods` của job và cấu hình còn lại từ `configs/research/adapted.yaml`. `kind: upstream` làm tương tự với `configs/research/upstream.yaml`; backbone OpenAI CLIP chọn trong YAML nghiên cứu. `kind: relation_ablation` chỉ hỗ trợ Fakeddit 6way, tối đa 5 mẫu. Mỗi job có thể chạy local hoặc Kaggle. Job có `enabled: true` được chọn khi không truyền `--job`; `--job` chọn theo tên dù đang `enabled: false`.

Job `upstream` có thể đặt `outputs: [original_image]` để chỉ đánh giá ảnh theo phương pháp gốc, hoặc `outputs: [multimodal]` để đánh giá phần mở rộng ảnh + text. Khi bỏ qua trường này, launcher dùng danh sách `outputs` từ `configs/research/upstream.yaml`.

Các job CrisisMMD có `result_protocol: image_only` đặt kết quả vào `results/image_only_<phương_pháp>/...`. Standard/adapted phải khai báo `image_weight: 1.0`; upstream phải chọn `outputs: [original_image]`; cả hai đều tắt `image_text_relation`. Launcher từ chối cấu hình sai trước khi gửi Kaggle. Xem [lệnh đối chứng ảnh-only](crisismmd_image_only_comparison.md).

Khi gửi Kaggle, launcher yêu cầu đúng một job và một lựa chọn cho mỗi model, phương pháp, backbone, output và seed Waffle. Nếu YAML có nhiều lựa chọn, dùng `--model`, `--method`, `--backbone`, `--output`, `--seed` để chọn riêng; launcher từ chối trước khi đóng gói hoặc gửi notebook. Chỉ `--ensemble true` là ngoại lệ có chủ ý để chạy nhiều model trong cùng một job.

Ensemble là tùy chọn trong job `kind: standard`:

```yaml
ensemble:
  enabled: true
  methods: [equal_mean, margin_gate]
  margin_threshold: 0.10
```

Cần ít nhất hai model và đúng một biến thể prompt/cách tổng hợp prompt để so sánh cùng điều kiện. `equal_mean` lấy trung bình điểm cuối cùng theo lớp. `margin_gate` chỉ đổi sang trung bình khi khoảng cách top-1 trừ top-2 của model đầu tiên nhỏ hơn hoặc bằng ngưỡng cố định; áp dụng cho bài toán một nhãn. MM-IMDb chỉ nhận `equal_mean`: trung bình điểm từng nhãn rồi dùng lại `decision_mode`, ngưỡng và `fallback_top1` trong `configs/experiment_config.yaml`; điểm `relative_z` không được coi là xác suất. Các ngưỡng đều cấu hình trước, không fit trên nhãn đánh giá. `--ensemble true/false` chỉ ghi đè flag của job được chọn.

`max_samples` là số mẫu hợp lệ tối đa mà runner xử lý. Với Fakeddit, launcher quét tuần tự và dừng ngay khi có đủ mẫu đạt `min_text_length`; `max_source_rows` tùy chọn chặn số dòng phải quét nếu file nhiều dòng ngắn. `kaggle.data_mode: full_dataset` đóng gói file dữ liệu đã xử lý cùng toàn bộ ảnh của mỗi dataset vào một Kaggle Dataset riêng (`vlm-runs-data-crisismmd`, `-fakeddit`, `-mmimdb`). Gói được cache trong `results/_system/launches/full_data_cache/`; launcher đối chiếu SHA nội dung file local và chỉ upload khi dữ liệu thay đổi. Đổi job, task hoặc `max_samples` sẽ dùng lại dataset đã upload. Lần đầu phải upload toàn bộ ảnh và mỗi phiên Kaggle vẫn mount/copy dataset đã chọn. `data_mode: bounded` giữ cách cũ: mỗi lượt chỉ upload các dòng đã quét và ảnh tương ứng vào dataset chung `vlm-runs-data`, nên đổi job thường tạo phiên bản dữ liệu mới. Cả hai chế độ chỉ đọc `data/processed/`, không đọc `dataraw/`.

Muốn upload trước cả ba dataset mà chưa chạy GPU, dùng `venv/bin/python -B scripts/launch_runs.py --sync-data all`. Có thể thay `all` bằng `fakeddit`, `crisismmd` hoặc `mmimdb`. Lệnh bulk sync đối chiếu danh sách tên/kích thước tệp, tránh tải lại ZIP lớn còn trỏ đến version cũ trong lúc Kaggle xử lý bản mới. Đây là kiểm tra `file_manifest`; notebook audit CPU đối chiếu SHA cây file mount, và worker cũng kiểm tra SHA ngay trước suy luận. Các job sau dùng lại dataset khi dữ liệu không đổi.

Với cohort chính mới, dùng `--launcher-config configs/launcher_3000.yaml`.
`--sync-data` luôn đối chiếu lại từ xa, không bỏ qua chỉ vì state local trùng.
`--sync-code` đồng bộ riêng gói mã mà không tạo kernel. Biên nhận kiểm tra nằm
ở `results/_system/launches/dataset_verification.json`; `sha256` và
`file_manifest` là hai mức xác nhận khác nhau. Gói full dataset có thêm
manifest cohort; worker kiểm tra ID, nhãn thật, split và số mẫu trước khi nạp
model, đồng thời lưu cohort ID trong receipt kết quả. Xem
[quy trình chạy 3.000 mẫu](chay_kaggle_3000.md).

## Chạy

```bash
# Kiểm tra local, không tải model.
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info --dry-run

# Xem trước gói Kaggle cho cả năm cặp dataset/task, không gọi Kaggle CLI.
venv/bin/python -B scripts/launch_runs.py --job fakeddit_2way --job fakeddit_6way --job crisismmd_info --job crisismmd_humanitarian --job mmimdb_genres --max-samples 5 --use-kaggle true --dry-run

# Chạy thật một job local, dùng YAML hiện tại.
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info

# Chạy ensemble trên Kaggle. Bật trước trong YAML hoặc ghi đè bằng flag.
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info --ensemble true --use-kaggle true
```

Muốn chạy nhiều job, lặp `--job`; launcher chạy tuần tự. `--max-samples N` ghi đè giới hạn của các job đã chọn. Mọi tham số phương pháp, model, task và số mẫu có thể sửa trực tiếp trong `configs/launcher.yaml`. Tùy chọn suy luận chi tiết và đường dẫn dữ liệu giữ ở `configs/experiment_config.yaml`; launcher tạo bản cấu hình cho từng job mà không sửa file gốc.

Để chạy mỗi model/phương pháp riêng mà không sửa YAML, dùng `--model`, `--method`, `--backbone`, `--output` và `--seed` theo loại job. Lệnh `venv/bin/python -B scripts/list_kaggle_commands.py` in toàn bộ các tổ hợp thành từng dòng có thể sao chép; xem [danh sách lệnh Kaggle](kaggle_commands.md).

## Thư mục

```text
experiments/ensemble/       mã thí nghiệm ensemble
configs/launcher.yaml       danh sách job và cờ local/Kaggle
scripts/launch_runs.py      điều phối và đóng gói
scripts/kaggle_worker.py     thực thi trên Kaggle
results/<phương_pháp>/<dataset>/<task>/samples_N/<model>/<task>_<id>/  kết quả model
results/_system/launches/staging/       gói trung gian, bỏ qua bởi Git
results/_system/launches/dataset_state.json  SHA gói đã gửi, bỏ qua bởi Git
results/_system/migration_map.json  đường dẫn cũ → mới của lượt lịch sử
```

Trên Kaggle, worker lưu bảng `summary.tsv`, `run_artifacts.zip` và `run_receipt.json` trong thư mục kết quả tương ứng. Dataset được dùng lại khi SHA không đổi; launcher đối chiếu nội dung bằng tải về thay cho `datasets status` (có thể trả 403). Worker kiểm tra SHA của cây file sau khi Kaggle tự giải nén và tìm mount cả dưới `/kaggle/input/<slug>` lẫn `/kaggle/input/datasets/...`.

Mỗi lượt mới đi thẳng vào thư mục phương pháp/dataset/task/số mẫu/model. Xem [quy tắc đặt thư mục](results_layout.md) để biết cách phân loại baseline, DCLIP, Waffle, upstream, ensemble và các báo cáo phân tích. Nếu muốn in vài lượt từ terminal mà không tạo file chỉ mục:

```bash
python -B scripts/list_results.py --limit 20
python -B scripts/list_results.py --dataset crisismmd --task informativeness --model metaclip2 --method adapted_dclip --samples 3000
```

Lệnh chỉ đọc các `summary.tsv` đã tải về và không ghi file. So sánh phương pháp phải chọn cùng dataset, task, tập mẫu, model và cấu hình liên quan; hai lượt cùng số mẫu chưa chắc có cùng mẫu, nên kiểm tra `sample_set_id` hoặc ID dự đoán khi cần kết luận chính thức.

Nếu CLI mất quyền đọc kernel hoặc mất mạng **sau khi đã gửi job**, đừng chạy lại lệnh tạo job vì như vậy sẽ tạo lượt GPU mới. Nếu `kaggle auth login` báo đã đăng nhập, lệnh đó chưa làm mới phiên: chạy `kaggle auth login --force`, hoàn tất đăng nhập, rồi tiếp tục chờ và tải kết quả từ đường dẫn `stage=` đã in:

```sh
python -B scripts/launch_runs.py --resume-stage results/_system/launches/staging/<id>_<job>_<suffix>
```

`--resume-stage` không upload dữ liệu, không đẩy kernel và không chạy GPU mới. Nó kiểm tra `run_receipt.json` và mã băm của đúng lượt đã gửi trước khi nhận kết quả. Hãy tiếp tục trước khi gửi job khác dùng chung kernel `vlm-runs`; Kaggle CLI đang trỏ đến phiên bản mới nhất khi không chỉ định phiên bản.

Nếu sau khi đăng nhập lại vẫn gặp `kernels.get denied`, mở notebook tại `https://www.kaggle.com/code/<username>/<kernel-slug>` để kiểm tra quyền xem và trạng thái phiên bản vừa chạy. Có thể tải output từ mục phiên bản trên web; lỗi quyền của CLI không chứng minh job GPU đã thất bại. Đừng dán token hoặc mã đăng nhập vào terminal chat.

Kết quả lịch sử trong `results/` đã được chuyển vào cây mới; `results1/` giữ nguyên. Lệnh cũ `scripts/launch_crisismmd_ensemble.py` vẫn hoạt động qua launcher chung. Các job nghiên cứu upstream/adapted trên Kaggle sẽ cài `requirements/research.txt` và clone ba repo tác giả theo đúng commit trong `configs/research/sources.json`; cần internet của Kaggle. Đã kiểm tra launcher chung bằng một lượt Kaggle GPU Tesla T4 với 5 mẫu CrisisMMD informativeness và ensemble; kết quả nằm ở `results/ensemble/crisismmd/informativeness/samples_5/multi_model/informativeness_d4f7cbd6281e/`.
