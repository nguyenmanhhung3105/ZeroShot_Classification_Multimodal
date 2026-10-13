# Launcher chung cho các phép chạy local và Kaggle

Dùng `scripts/launch_runs.py` và chỉnh `configs/launcher.yaml`. Cờ `use_kaggle: false` chạy trên máy; `true` đóng gói đúng số mẫu đã chọn, gửi lên Kaggle GPU, chờ hoàn tất và tải kết quả. Có thể ghi đè bằng `--use-kaggle true/false`. Không lưu token trong YAML; đăng nhập một lần bằng `venv/bin/kaggle auth login`.

## Job trong YAML

Mỗi job xác định `kind`, `dataset`, `task`, `max_samples`, `models`. Các cặp dữ liệu có sẵn:

| Dataset | Task | Kiểu nhãn |
|---|---|---|
| Fakeddit | `2way`, `6way` | Một nhãn |
| CrisisMMD | `informativeness`, `humanitarian` | Một nhãn |
| MM-IMDb | `genres` | Nhiều nhãn |

`kind: standard` dùng `src/run_experiment.py`; `prompt_variants` và `prompt_aggregations` chọn các thử nghiệm prompt của runner gốc. `kind: demo` chạy `src/demo.py` và xuất hình minh họa. `kind: adapted` lấy phương pháp từ `methods` của job và cấu hình còn lại từ `configs/research/adapted.yaml`. `kind: upstream` làm tương tự với `configs/research/upstream.yaml`; backbone OpenAI CLIP chọn trong YAML nghiên cứu. `kind: relation_ablation` chỉ hỗ trợ Fakeddit 6way, tối đa 5 mẫu. Mỗi job có thể chạy local hoặc Kaggle. Job có `enabled: true` được chọn khi không truyền `--job`; `--job` chọn theo tên dù đang `enabled: false`.

Ensemble là tùy chọn trong job `kind: standard`:

```yaml
ensemble:
  enabled: true
  methods: [equal_mean, margin_gate]
  margin_threshold: 0.10
```

Cần ít nhất hai model và đúng một biến thể prompt/cách tổng hợp prompt để so sánh cùng điều kiện. `equal_mean` lấy trung bình điểm cuối cùng theo lớp. `margin_gate` chỉ đổi sang trung bình khi khoảng cách top-1 trừ top-2 của model đầu tiên nhỏ hơn hoặc bằng ngưỡng cố định; áp dụng cho bài toán một nhãn. MM-IMDb chỉ nhận `equal_mean`: trung bình điểm từng nhãn rồi dùng lại `decision_mode`, ngưỡng và `fallback_top1` trong `configs/experiment_config.yaml`; điểm `relative_z` không được coi là xác suất. Các ngưỡng đều cấu hình trước, không fit trên nhãn đánh giá. `--ensemble true/false` chỉ ghi đè flag của job được chọn.

`max_samples` là số mẫu hợp lệ tối đa. Với Fakeddit, launcher quét tuần tự và dừng ngay khi có đủ mẫu đạt `min_text_length`; `max_source_rows` tùy chọn chặn số dòng phải quét nếu file nhiều dòng ngắn. Gói Kaggle chỉ chứa các dòng đã quét và ảnh tương ứng từ `data/processed/`; không đọc `dataraw/`.

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

## Thư mục

```text
experiments/ensemble/       mã thí nghiệm ensemble
configs/launcher.yaml       danh sách job và cờ local/Kaggle
scripts/launch_runs.py      điều phối và đóng gói
scripts/kaggle_worker.py     thực thi trên Kaggle
results/runs/<job>/<id>/         đầu ra runner chuẩn
results/experiments/<job>/<id>/  đầu ra ensemble/nghiên cứu
results/launches/staging/       gói trung gian, bỏ qua bởi Git
results/launches/dataset_state.json  SHA gói đã gửi, bỏ qua bởi Git
```

Trên Kaggle, worker lưu bảng `summary.tsv`, `run_artifacts.zip` và `run_receipt.json` trong thư mục kết quả tương ứng. Dataset được dùng lại khi SHA không đổi; launcher đối chiếu nội dung bằng tải về thay cho `datasets status` (có thể trả 403). Worker kiểm tra SHA của cây file sau khi Kaggle tự giải nén và tìm mount cả dưới `/kaggle/input/<slug>` lẫn `/kaggle/input/datasets/...`.

Nếu CLI mất quyền đọc kernel hoặc mất mạng **sau khi đã gửi job**, đừng chạy lại lệnh tạo job vì như vậy sẽ tạo lượt GPU mới. Nếu `kaggle auth login` báo đã đăng nhập, lệnh đó chưa làm mới phiên: chạy `kaggle auth login --force`, hoàn tất đăng nhập, rồi tiếp tục chờ và tải kết quả từ đường dẫn `stage=` đã in:

```sh
python -B scripts/launch_runs.py --resume-stage results/launches/staging/<id>_<job>_<suffix>
```

`--resume-stage` không upload dữ liệu, không đẩy kernel và không chạy GPU mới. Nó kiểm tra `run_receipt.json` và mã băm của đúng lượt đã gửi trước khi nhận kết quả. Hãy tiếp tục trước khi gửi job khác dùng chung kernel `vlm-runs`; Kaggle CLI đang trỏ đến phiên bản mới nhất khi không chỉ định phiên bản.

Nếu sau khi đăng nhập lại vẫn gặp `kernels.get denied`, mở notebook tại `https://www.kaggle.com/code/<username>/<kernel-slug>` để kiểm tra quyền xem và trạng thái phiên bản vừa chạy. Có thể tải output từ mục phiên bản trên web; lỗi quyền của CLI không chứng minh job GPU đã thất bại. Đừng dán token hoặc mã đăng nhập vào terminal chat.

Các thư mục lịch sử `results/`, `results1/` và lệnh cũ `scripts/launch_crisismmd_ensemble.py` vẫn được giữ để không mất kết quả hoặc phá lệnh đã dùng. Các job nghiên cứu upstream/adapted trên Kaggle sẽ cài `requirements/research.txt` và clone ba repo tác giả theo đúng commit trong `configs/research/sources.json`; cần internet của Kaggle. Đã kiểm tra launcher chung bằng một lượt Kaggle GPU Tesla T4 với 5 mẫu CrisisMMD informativeness và ensemble; kết quả được tải về `results/experiments/crisismmd_info/d4f7cbd6281e/`. Các job/dataset còn lại đã qua kiểm thử cấu hình và đóng gói mẫu nhỏ, chưa chạy inference GPU thật.
