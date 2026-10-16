# Phân loại nội dung đa phương thức zero-shot

Đầu vào của hệ thống chính là **ảnh + text**. Dự án dùng MetaCLIP2, DFN hoặc SigLIP2
đã huấn luyện sẵn để so nội dung với prompt nhãn, sau đó kết hợp điểm hai nhánh.
Không có bước huấn luyện hoặc fine-tune trong runner đánh giá.

## Bố trí mã nguồn

```text
src/
├── run_experiment.py          # chạy model × dataset
├── demo.py                    # chạy nhóm mẫu nhỏ và vẽ kết quả
├── config.py                  # đọc, kiểm tra YAML
├── run/
│   ├── common.py              # đọc dữ liệu, ánh xạ nhãn, ảnh, summary
│   ├── run_fakeddit.py         # tác vụ 2way hoặc 6way
│   ├── run_crisismmd.py        # informativeness hoặc humanitarian
│   └── run_mmimdb.py           # phân loại nhiều thể loại
├── inference.py               # chấm điểm ảnh/text và fusion
├── pair_relation.py           # cosine ảnh–text, JS/entropy và policy theo nhãn
├── text_scale.py              # chọn scale riêng cho text
├── text_chunking.py           # chia plot dài và gộp các đoạn
├── evaluate.py                # tính metric
├── evaluation_audit.py        # báo cáo nhánh, độ dài và bộ mẫu
├── logging_utils.py           # dự đoán từng mẫu, lỗi và manifest
├── models/model_registry.py   # tải và mã hóa bằng model
├── prompts/                   # prompt và taxonomy dùng ở code chính
├── research/                  # runner paper và đối chứng quan hệ ảnh–text
└── preprocessing/             # script chuẩn bị/kiểm tra dữ liệu, không tự chạy

experiments/ensemble/           # mã thử nghiệm ensemble tùy chọn, 3 dataset
configs/
├── experiment_config.yaml     # cấu hình hệ thống chính
├── launcher.yaml              # job, số mẫu, phương pháp, ensemble và local/Kaggle
├── crisismmd_launcher.yaml    # cấu hình tương thích launcher CrisisMMD cũ
├── prompts/                   # prompt/descriptor cố định
└── research/
    ├── upstream.yaml          # model CLIP gốc, repo, task và chế độ đầu ra
    ├── adapted.yaml           # đối chứng trên model hiện dùng
    ├── sources.json           # URL và commit repo tác giả
    └── descriptors/           # mô tả lớp cố định cho cả năm task

tests/                         # chỉ kiểm thử tự động bằng dữ liệu giả
scripts/                       # launcher chung, worker Kaggle và lệnh tương thích cũ
requirements/kaggle_launcher.txt # CLI Kaggle tùy chọn cho máy gửi bài chạy
docs/                          # hướng dẫn chi tiết và sơ đồ hệ thống
├── experiments/               # tài liệu và báo cáo thử nghiệm
├── so_do_he_thong/             # bản đồ input → xử lý → output của model/phương pháp
│   └── so_do/                  # SVG, nguồn DOT và script dựng sơ đồ
third_party/                   # checkout repo tác giả, Git bỏ qua
results/<method>/<dataset>/<task>/samples_N/<model>/<task>_<run_id>/  kết quả từng lượt
results/_system/              # staging/cache Kaggle và bảng đường dẫn cũ → mới
results1/                      # kết quả lịch sử, được giữ nguyên
backup/, backups/              # bản sao trước sửa và file đã loại bỏ
```

Mỗi lần thêm, chuyển hoặc bỏ module phải cập nhật cây thư mục, lệnh chạy và cấu hình
trong README này cùng thay đổi mã nguồn. Comment/docstring của code do dự án quản lý
viết bằng tiếng Việt. Prompt tiếng Anh là đầu vào model và được giữ nguyên. Repo
tác giả và thông báo giấy phép giữ nguyên văn bản nguồn.

## Cài đặt và chạy hệ thống chính

Để hình dung hệ thống trước khi chạy, mở [bản đồ hệ thống](docs/so_do_he_thong/README.md).
Tài liệu có sơ đồ riêng cho MetaCLIP2, DFN5B, SigLIP2, OpenAI CLIP, WaffleCLIP,
DCLIP, hai runner nghiên cứu và luồng dữ liệu/kết quả. Sơ đồ SVG được dựng sẵn để
xem bố cục ổn định trong Markdown Preview hoặc trên GitHub.

Chạy tại thư mục gốc, trong môi trường Python của dự án:

```sh
pip install -r requirements.txt

# Các model × dataset có enabled: true
python3 src/run_experiment.py

# Chỉ một dataset
python3 src/run/run_fakeddit.py
python3 src/run/run_crisismmd.py
python3 src/run/run_mmimdb.py

# Nhóm mẫu nhỏ, cùng logic với chạy chính
python3 src/demo.py
```

Các lệnh trên nhận `--config <đường_dẫn.yaml>`. Lần tải model đầu có thể cần mạng và
dung lượng lưu trọng số. Không chạy module model_registry để tự tải toàn bộ model.

## Cấu hình cần chỉnh

Chọn model ở danh sách `models` trong `configs/experiment_config.yaml`.
Các tên hợp lệ: `metaclip2_vith14`, `dfn5b_vith14`, `siglip2_so400m`.

Với mỗi dataset, chỉnh `enabled`, `task`, `max_samples` và:
```yaml
inference:
  image_weight: 0.5       # trọng số ảnh; text có trọng số 1 - giá trị này
  text_logit_scale: null  # null lấy scale model; 25/50/... là scale tuyệt đối
```

`image_weight: 1` chỉ ảnh, `0` chỉ text, `0.5` chia đều; phạm vi hợp lệ [0, 1].
Text rỗng luôn dùng ảnh dự phòng. Cấu hình riêng của dataset ưu tiên hơn mục
`inference` toàn cục. Cấu hình cũ với `image_weight_overrides` và
`text_logit_scale_overrides` vẫn đọc được nếu không có lựa chọn riêng mới.

| Dataset | task | Nhãn/cột |
|---|---|---|
| Fakeddit | 2way / 6way | label_col_2way / label_col_6way |
| CrisisMMD | informativeness / humanitarian | label_col_info / label_col_human |
| MM-IMDb | genres | label_col |

Chỉ đổi task để chọn cột nhãn; loại tác vụ được suy ra theo dataset.
Fakeddit 2way: 0 giả, 1 thật. Fakeddit 6way: 0 thật, 1 châm biếm, 2 liên kết sai,
3 mạo danh, 4 chỉnh sửa, 5 gây hiểu lầm.
[Nguồn ánh xạ Fakeddit](https://github.com/entitize/Fakeddit/issues/14#issuecomment-815355653).

`prompt_aggregation: mean_embed` lấy trung bình embedding rồi chuẩn hóa lại;
`mean_sim` lấy trung bình cosine các prompt, không chuẩn hóa lại trung bình.
Muốn quét cả hai, đặt `prompt_aggregations: [mean_embed, mean_sim]`.
Fakeddit 6way hỗ trợ `prompt_variants: [current, class_name, descriptors]`;
các task khác dùng current.

MM-IMDb có `chunk_long_text: true/false`. Khi bật, `chunk_aggregation` chọn
`mean_embed` hoặc `max_sim`. Plot được chia vừa context của tokenizer;
max_sim lấy điểm cao nhất của từng nhãn qua các đoạn.

MM-IMDb mặc định `decision_mode: relative_z`: chuẩn hóa cosine qua 23 nhãn
của **từng mẫu và từng nhánh**, fusion rồi chọn điểm lớn hơn
`relative_z_threshold: 1.0`. Điểm không phải xác suất; mode này bỏ scale/bias.
`threshold: 0.22` chỉ dùng khi chủ động chọn `fixed_threshold`.

Nhánh ảnh ↔ text có flag chung cho mọi model/dataset, **mặc định tắt**:

```yaml
inference:
  image_text_relation:
    enabled: false           # đổi true để bật
    mode: auto
    weight: 0.25
```

`auto` chọn class_aware cho Fakeddit 6way (True/False Connection); các task còn
lại chỉ diagnostic. Cấu hình `datasets.<tên>.image_text_relation` ưu tiên hơn
flag chung, nên có thể bật/tắt riêng từng dataset. Chuyển task không cần sửa
mode nếu dùng auto. Xem [công thức và sơ đồ](docs/so_do_he_thong/13_quan_he_anh_text.md).

## Luồng xử lý và giới hạn mẫu

YAML → đọc TSV/CSV/Parquet → lọc text → chọn số mẫu → kiểm tra ảnh →
mã hóa ảnh/text và prompt → chấm điểm từng nhánh → fusion → chọn nhãn →
metric → TSV/manifest.

Fakeddit mặc định bỏ text có không quá 20 ký tự sau khi bỏ dấu cách.
`max_samples` là số dòng **sau lọc text, trước kiểm tra ảnh**. Ảnh lỗi bị bỏ;
không tự bù bằng mẫu tiếp theo. Giữ ảnh dưới dạng đường dẫn và mở theo batch.
Các runner dùng chung bộ lọc, ánh xạ, công thức suy luận và metric.

## Kết quả và cách đối chiếu

Xem [giải thích các chỉ số output](docs/giai_thich_chi_so_output.md) để hiểu công thức,
ý nghĩa và cách dùng từng metric, điểm dự đoán, cột log và metadata.

Summary của lượt mới nằm trong `results/<method>/<dataset>/<task>/samples_N/<model>/<task>_<run_id>/summary.tsv`; xem [cấu trúc kết quả](docs/experiments/results_layout.md).
Demo lưu `demo_summary_...tsv` trong thư mục visualization_dir.
Dự đoán mỗi lần chạy nằm dưới raw_predictions_dir:

- `raw_predictions.tsv`: nhãn thật/dự đoán, điểm các lớp và các nhánh.
- `error_analysis.tsv`: mẫu bị đoán sai.
- `manifest.json`: cấu hình hiệu lực và thông tin lần chạy.
- `diagnostics.tsv`: metric ảnh/text/fusion, độ nhọn, số nhãn và fallback.
- `text_lengths.tsv`: độ dài text trước cắt.
- `evaluation_snapshot.json`: cấu hình, prompt, mẫu, git hash và hash mã nguồn.

Fakeddit/CrisisMMD: Macro-F1 và balanced accuracy giúp quan sát các lớp hiếm;
accuracy/micro-F1 là thông tin bổ sung. MM-IMDb dùng macro/micro/samples-F1;
không dùng Hamming accuracy làm bằng chứng chính cho chất lượng nhiều nhãn.

`fusion_macro_f1_gain` so Macro-F1 fusion với nhánh đơn tốt nhất, chỉ tính khi
hai nhánh được đánh giá trên cùng đầy đủ bộ mẫu. Không coi tăng nhỏ là cải thiện
có ý nghĩa thống kê. Giữ cùng model, bộ mẫu, filter và trọng số khi đối chứng prompt.

`evaluation.enabled` điều khiển báo cáo; `sample_lock_mode: "off"` mặc định không
tạo khóa. `create_or_verify` và `verify` là lựa chọn chủ động. Split trong YAML do
người dùng khai báo, chưa được xác minh từ nguồn. Thử nhiều lần trên val là đánh
giá khám phá; kết quả đó không chứng minh hiệu quả trên test độc lập.

## Thí nghiệm nghiên cứu

Launcher chung chạy Fakeddit 2way/6way, CrisisMMD informativeness/humanitarian,
MM-IMDb genres và các runner nghiên cứu. Cấu hình số mẫu, model, phương pháp và
flag ensemble trong `configs/launcher.yaml`; chi tiết ở
[hướng dẫn launcher](docs/experiments/launcher_kaggle.md). Mặc định chạy local;
`--use-kaggle true` gửi job lên Kaggle. Xem trước không tải model hoặc upload:

```sh
venv/bin/python -B scripts/launch_runs.py --job crisismmd_info --dry-run
venv/bin/python -B scripts/launch_runs.py --job mmimdb_genres --ensemble true --use-kaggle true --dry-run
```

Ensemble chỉ chạy khi `ensemble.enabled: true` hoặc `--ensemble true`.
Lệnh `scripts/launch_crisismmd_ensemble.py` cũ vẫn dùng được để chạy lại
phép thử CrisisMMD, nhưng job mới nên dùng launcher chung.

Đối chứng quan hệ ảnh–text trên đúng năm dòng đầu Fakeddit, giữ cùng điểm nền:

```sh
venv/bin/python -B src/research/run_relation_ablation.py --offline --device cpu
```

Runner encode một lần/model, so baseline/heuristic cũ/class_aware và lưu plan,
hash source, summary, điểm từng mẫu dưới `results/relation_ablation/fakeddit/6way/samples_N/<model>/`.
Ngày 12-10-2026 đã thử MetaCLIP2/DFN5B/SigLIP2; chưa thấy cải thiện nhãn trên
nhóm năm mẫu này. Xem [báo cáo](docs/quan_he_anh_text_20261012.md).
Hai file `configs/research/adapted.yaml` và `upstream.yaml` có flag riêng
`image_text_relation.enabled: false` ở cấp gốc; đổi true để bật trên tất cả
model/phương pháp/task của runner được chọn. Upstream giữ `original_image`
chỉ diagnostic; `multimodal` có thể class_aware theo taxonomy. Khi flag tắt,
không tính nhánh quan hệ hoặc mã hóa thêm text/ảnh dành riêng cho nó.

Có hai runner riêng trong `src/research/`. Chạy `run_upstream.py` để dùng OpenAI
CLIP cùng các hàm tạo prompt và chấm điểm lấy từ repo WaffleCLIP/DCLIP đã ghim
commit; chạy `run_adapted_methods.py` để áp dụng các prompt ảnh lên model hiện có.
Code chính không tự kích hoạt cả hai thí nghiệm.

Để chuẩn bị trước prompt và lưu cấu hình đã khóa, không mở dataset hoặc tải model:

```sh
python3 src/research/run_upstream.py --prepare-only
```

Để chạy đối chứng bằng backbone ViT-B/32 của OpenAI CLIP:

```sh
python3 src/research/run_upstream.py
```

Runner dùng `configs/research/upstream.yaml`. Hiện chỉ bật CrisisMMD humanitarian
và `max_samples: 5` để chạy thử; Fakeddit và MM-IMDb đang tắt. Sau khi xem
`frozen_plan.json`, có thể bật một dataset, chọn task và giới hạn mẫu trong YAML.
Chỉ nên tăng lên 1000 và `seeds: [0, 1, 2, 3, 4, 5, 6]` sau khi đã chạy thử nhỏ;
riêng WaffleCLIP chạy lặp cho từng seed. Lượt thực tế có thể tải trọng số CLIP.

Các phương pháp cấu hình sẵn:

| Tên | Cách tạo prompt ảnh |
|---|---|
| `clip` | Baseline kiểu WaffleCLIP: `A photo of a {class}.` |
| `waffle` | Hàm gốc WaffleCLIP, từ điển chính thức và cách sinh prompt theo seed |
| `dclip_labels` | Baseline tên lớp theo template mặc định DCLIP |
| `dclip` | Mô tả lớp cố định, ghép theo template mặc định DCLIP |

Mỗi phương pháp có thể xuất `original_image` và `multimodal`. Kết quả ảnh gốc dùng
argmax cosine như bài phân loại ảnh một nhãn trong paper. Kết quả đa phương thức
dùng ảnh và text sample so với prompt text cố định của dự án rồi fusion theo YAML;
nhánh text giống nhau giữa các phương pháp để đo tác động nhánh ảnh. MM-IMDb là
phần mở rộng nhiều nhãn: các paper ảnh gốc không định nghĩa cách chấm 23 thể loại,
nên quyết định `relative_z`/`fixed_threshold` lấy từ cấu hình dự án.

Các file `configs/research/descriptors/*.json` đủ taxonomy cho Fakeddit 2/6 lớp,
CrisisMMD hai task và MM-IMDb 23 thể loại. Chúng được viết cố định từ định nghĩa
lớp, không phải descriptor GPT-3 gốc được tác giả dùng trong paper. Đặc biệt,
ảnh đơn lẻ không luôn cho biết một bài Fakeddit thật hay giả.

Ba checkout tác giả nằm trong `third_party/` và bị Git của dự án bỏ qua. Nguồn
được ghim tại `configs/research/sources.json`; cài thêm thư viện cần thiết bằng:

```sh
pip install -r requirements/research.txt
```

Đọc [tài liệu nghiên cứu](docs/research.md) về nguồn code, chế độ đầu ra và giới hạn
so sánh. Đọc [tài liệu bàn giao dự án](docs/ban_giao_du_an.md) để biết luồng hiện
tại, trạng thái triển khai và các bước tiếp tục ở tài khoản/môi trường khác.

Ngày 11-10-2026 đã chạy thử OpenAI CLIP thật cho cả năm tác vụ, mỗi tác vụ 5 mẫu;
đã đối chiếu scoring humanitarian và sửa báo cáo metric theo seed. Xem
[báo cáo tiếp tục nghiên cứu](docs/zeroshot_smoke_20261011.md) để đọc kết quả,
các file log và phần còn thiếu. Chưa có đánh giá quy mô lớn hoặc kết luận tốt hơn
hệ thống chính. Summary/manifest upstream mới ghi thiết bị thực thi thực tế.

Đối chứng thích nghi trên model hiện có:

```sh
python3 src/research/run_adapted_methods.py --prepare-only
python3 src/research/run_adapted_methods.py
```

Runner thích nghi lấy model từ YAML chính. Cấu hình hiện giữ limit 1000 như lựa
chọn trước đó; hãy giảm limit trước khi chạy thử. Không nhầm hai runner hoặc model
OpenAI CLIP với MetaCLIP2/SigLIP2/DFN.

## Kiểm thử và dữ liệu

```sh
python3 -B -m unittest discover -s tests -b
```

Test dùng ảnh/dữ liệu giả trong thư mục tạm; không tải checkpoint hoặc chạy toàn
dataset. Công cụ thống kê nhãn đã hủy được bỏ khỏi src/tests/docs; bản sao nằm
trong backup/20261010_cleanup. Kết quả thống kê cũ vẫn được giữ.

Các script preprocessing chỉ chạy khi chủ động gọi; một số đọc dữ liệu thô và
toàn bộ processed. Khi rà soát tuân thủ AGENTS.md: không đọc dữ liệu thô, chỉ
schema/tối đa 5 bản ghi đầu processed, không chạy pipeline tiền xử lý.
Quy trình tạo CrisisMMD từ annotation thô chưa được triển khai trong script hiện có.
