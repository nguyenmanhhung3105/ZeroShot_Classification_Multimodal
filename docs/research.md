# Các thí nghiệm WaffleCLIP / DCLIP

Hai runner có flag quan hệ riêng, mặc định tắt: `image_text_relation.enabled: false`
ở cấp gốc YAML nghiên cứu. Bật true dùng cho mọi backbone/method/task được chọn;
mode auto chọn class_aware Fakeddit 6way, task khác diagnostic. Upstream
original_image chỉ diagnostic để giữ cách quyết định ảnh gốc. Chi tiết tại
[tài liệu flag và công thức](so_do_he_thong/13_quan_he_anh_text.md).

## Dùng checkpoint và hàm gốc của repo tác giả

Runner: `src/research/run_upstream.py`. Cấu hình: `configs/research/upstream.yaml`.
Repo OpenAI CLIP, WaffleCLIP và DCLIP được ghim theo commit trong
`configs/research/sources.json`. Model mặc định là OpenAI CLIP ViT-B/32 — không
phải MetaCLIP2, SigLIP2 hoặc DFN trong pipeline chính.

Từ thư mục gốc, chuẩn bị trước prompt mà chưa mở dữ liệu hay nạp trọng số:

```sh
python3 src/research/run_upstream.py --prepare-only
```

Sau khi kiểm tra `frozen_plan.json`, chạy các dataset được bật trong YAML:

```sh
python3 src/research/run_upstream.py
```

`upstream.yaml` hiện bật CrisisMMD humanitarian, 5 mẫu; hai dataset kia tắt.
Tất cả descriptor có sẵn tại `configs/research/descriptors/`. Chọn `enabled`,
`task`, `max_samples` và `backbones` trong YAML để mở rộng từng lượt.

| `method` | Cách tạo prompt ảnh |
|---|---|
| `clip` | Baseline kiểu WaffleCLIP, `A photo of a {class}.` |
| `waffle` | Hàm gốc WaffleCLIP, dùng từ điển và thuật toán tạo prompt của repo |
| `dclip_labels` | Baseline tên lớp theo template mặc định của DCLIP |
| `dclip` | Hàm gốc DCLIP ghép mô tả lớp với tên lớp |

`seeds` chỉ lặp phương pháp waffle. `[0]` là bước chạy thử; thử nghiệm seed đầy
đủ nên dùng các seed paper đề xuất (ví dụ `[0, 1, 2, 3, 4, 5, 6]`).
`waffle_pairs: 15` tạo số cặp theo tham số của repo.

Mỗi trial lưu hai loại kết quả khi `outputs` gồm cả hai:

- `original_image`: chỉ ảnh, dùng argmax cosine, phù hợp thiết lập phân loại ảnh
  một nhãn gốc của các phương pháp.
- `multimodal`: ảnh và text sample cùng được so với vector prompt nhãn; fusion,
  text scale, aggregation, chunking và quy tắc quyết định dùng code/config dự án.

Text prompt và embedding text được giữ chung giữa các phương pháp ảnh trong cùng
cohort. Như vậy hàng `multimodal` kiểm tra tác động của thay đổi prompt ảnh khi
các điều kiện nhánh text giữ nguyên. Đây là phần mở rộng, không phải phương pháp
ảnh gốc trong paper. MM-IMDb là task nhiều nhãn; code paper không định nghĩa quy
tắc 23 thể loại, do đó đầu ra của task này được đánh dấu `multi_label_extension`
và dùng `decision_mode`, ngưỡng, fallback, chunking của YAML chính.

Descriptor trong JSON được viết cố định từ định nghĩa lớp. Chúng đủ cho cả năm
task nhưng **không phải** câu do GPT-3 sinh trong thí nghiệm gốc. Không dùng lỗi
trên tập đánh giá để sửa descriptor giữa các lượt so sánh. Mô tả Fakeddit bằng ảnh
là giả thuyết nội dung ảnh, không thể tự xác minh tính thật giả của bài đăng.

Runner này dùng các phần chính của phương pháp và checkpoint gốc; nó không chạy
dataset/benchmark riêng của tác giả, không gọi GPT-3 hoặc làm lại toàn bộ ablation
của paper. Vì thế không diễn giải đây là tái lập toàn bộ kết quả công bố.

## Phần đã triển khai và phần chưa xác minh

Đã ghim ba repo, gọi hàm sinh/mã hóa/tổng hợp điểm của tác giả, nạp model OpenAI
CLIP, chuyển loader sang schema và filter của dự án, chuẩn bị prompt trước inference,
ghi log từng mẫu và metric theo code chung. Ảnh và text được mã hóa một lần cho mỗi
model/cohort; kết quả được tái sử dụng giữa phương pháp/seed. Các output được ghi
vào thư mục mới dưới `results/upstream_<method>/<dataset>/<task>/samples_N/<model>/` (hoặc `upstream_suite` khi chạy nhiều phương pháp).

Ngày 11-10-2026 đã chạy checkpoint thật trên 5 mẫu đầu cho cả năm task, kiểm tra
log/bộ mẫu và đối chiếu scoring humanitarian với CLIP trực tiếp. Xem
[báo cáo chạy thử](zeroshot_smoke_20261011.md) để biết kết quả và các artifact.
Lệnh `--prepare-only` vẫn chỉ xác minh prompt/cấu hình; kiểm thử tự động vẫn dùng
model embedding giả và dữ liệu ảnh/TSV tổng hợp. Chưa chạy cohort 1000 mẫu hoặc
nhiều seed; chỉ mở rộng khi phạm vi đánh giá đó được yêu cầu. Việc chọn MetaCLIP2/SigLIP2/DFN so với
OpenAI CLIP thay đổi cả backbone lẫn phương pháp, nên chỉ số chênh không chứng
minh riêng hiệu quả của thuật toán WaffleCLIP/DCLIP.

Tham khảo repo đã ghim: [OpenAI CLIP](https://github.com/openai/CLIP),
[WaffleCLIP](https://github.com/ExplainableML/WaffleCLIP),
[DCLIP](https://github.com/sachit-menon/classify_by_description_release).

---

# Bản thích nghi WaffleCLIP / DCLIP trên model hiện có

## Phạm vi

Runner thích nghi: `src/research/run_adapted_methods.py`.
Đầu vào cuối cùng vẫn là ảnh + text. Chỉ thay cách tạo vector lớp của **nhánh ảnh**.
Nhánh text dùng nguyên prompt, aggregation, scale và chunking hiện có. Fusion,
relative-z/threshold, fallback và metric gọi lại code chính, không viết công thức khác.

Repo tham khảo ở `third_party/WaffleCLIP` và `third_party/DCLIP`, được bỏ qua bởi Git.
Commit chính xác lưu trong `configs/research/sources.json`. Không cài environment cũ,
không thực thi code upstream, không thêm backbone. Trước thay đổi đã backup tại
`backup/20261010_paper_methods/`. Chưa commit/push.

## Chạy từ thư mục gốc

```sh
# Chỉ sinh/lưu prompt + cấu hình; không đọc dataset, không load model
python3 src/research/run_adapted_methods.py --prepare-only

# Chạy thí nghiệm nhỏ theo cấu hình riêng
python3 src/research/run_adapted_methods.py
```

Sửa `configs/research/adapted.yaml`:

```yaml
dataset: crisismmd
task: humanitarian
max_samples: 5
methods: [baseline, current_mean_sim, class_name, waffle]
seeds: [0, 1, 2, 3, 4]
waffle_pairs: 15
```

Mặc định có **8 lần chạy/model**: 3 đối chứng + 5 seed WaffleCLIP.
Model lấy từ `models` trong `configs/experiment_config.yaml`; bản config hiện tại
chọn MetaCLIP2. Tự chọn SigLIP2/DFN trong config chính nếu muốn so cùng backbone.
Không có flag train, không gọi LLM/API sinh prompt khi đánh giá.

Muốn kiểm tra nhanh một seed: `seeds: [0]` (không thể kết luận độ ổn định).
Sau smoke test, có thể tự đổi `max_samples: 1000`; số này là số mẫu sau lọc text
đối với Fakeddit, nên có thể cần đọc nhiều hơn 1000 dòng nguồn. Ảnh lỗi bị bỏ như
runner chính; không bù mẫu sau đó. Không dùng 5 mẫu để kết luận macro-F1 tốt hơn.

Đổi dataset/task:

| dataset | task |
|---|---|
| fakeddit | 2way hoặc 6way |
| crisismmd | informativeness hoặc humanitarian |
| mmimdb | genres |

Đường dẫn/cột nhãn/filter/fusion/scale/threshold/chunking được kế thừa đúng dataset
từ base config. Runner nghiên cứu bật audit, tắt khóa mẫu tự động, nhưng vẫn so
fingerprint cohort giữa các lần chạy: nếu mẫu khác sẽ dừng, giữ log trước đó.
Không hỗ trợ đồng thời nhiều `prompt_variants`/`prompt_aggregations` trong base config;
hãy để đúng một giá trị để đối chứng dễ hiểu. `enabled` của dataset được ghi đè true
chỉ cho dataset đã chọn trong runner nghiên cứu, không sửa file config chính.

## Các đối chứng kiểm chứng điều gì?

| image_method | Nhánh ảnh | Mục đích |
|---|---|---|
| baseline | Prompt và aggregation hiện tại | Mốc so với hệ thống cũ |
| current_mean_sim | Prompt hiện tại, mean cosine | Tách tác động riêng của aggregation |
| class_name | Một prompt `A photo of a {class}.` | Mốc công bằng cho WaffleCLIP |
| waffle | 2K prompt tên lớp + nhiễu ngẫu nhiên dùng chung giữa các lớp, mean cosine | Đo lợi ích ensemble ngẫu nhiên |
| dclip | Mô tả cố định theo lớp, mean cosine | Đo lợi ích descriptor có ngữ nghĩa |

Nếu baseline vốn dùng mean_sim, baseline và current_mean_sim phải trùng nhau
(ngoại trừ sai khác số học rất nhỏ nếu backend không tất định).
`prompt_aggregation` trong summary là aggregation text/baseline;
`image_prompt_aggregation` là aggregation nhánh ảnh của đối chứng.

## DCLIP: thử thêm trên CrisisMMD humanitarian

Đã cung cấp `configs/research/descriptors/crisismmd_humanitarian.json`, đủ 8 lớp, 3 mô tả/lớp.
Đây là descriptor do assistant tạo từ định nghĩa nhãn, cố định trước khi chạy,
không khai thác dòng dữ liệu hoặc lỗi val; **không phải GPT-3 descriptors gốc**.

```yaml
methods: [baseline, current_mean_sim, class_name, waffle, dclip]
descriptor_file: configs/research/descriptors/crisismmd_humanitarian.json
```

Khi đó có 9 lần/model. Các task khác cần file descriptor đúng nhãn; runner từ chối
nhãn thiếu/thừa thay vì tự đoán hoặc fallback im lặng. Schema:

```json
{
  "provenance": "nguồn mô tả, cách sinh và ngày cố định",
  "descriptors": {
    "label_id": ["mô tả đặc trưng bằng tiếng Anh, không phải toàn bộ prompt"]
  }
}
```

Fakeddit dùng khóa chuỗi ID như `"0"`, không thay bằng tên nhãn; CrisisMMD dùng
khóa taxonomy; MM-IMDb dùng tên genre đúng hoa/thường. Lưu đầy đủ mọi lớp.

## Phần theo paper và phần thích nghi

- WaffleCLIP: từ điển chính thức, cặp từ/cụm từ + chuỗi ký tự, quy tắc độ dài theo
  tên lớp, descriptor dùng chung giữa các lớp, wrapper và mean cosine theo code gốc.
- Khác upstream: dùng RNG cục bộ và sinh trực tiếp bộ suffix chung; dùng `a-z`
  cố định thay vì suy bảng ký tự từ GPT descriptors. Seed không tái lập bitwise
  upstream. Không bật broad concepts, không trộn GPT+waffle trong thử nghiệm này.
- Mean cosine = dot product với trung bình các vector prompt đã chuẩn hóa;
  **không chuẩn hóa lại trung bình**. Không cấp phát ma trận mẫu × toàn bộ prompt.
- DCLIP: scoring bằng mô tả/mean similarity; descriptor mới, backbone và dataset mới.
- SigLIP2/MetaCLIP2/DFN, text sample → prompt, fusion và quyết định multi-label là
  phần thích nghi của dự án. Không được gọi kết quả là tái lập nguyên trạng paper.
- Main/demo cũ vẫn là baseline, không đọc cấu hình paper; chỉ runner mới thực thi
  các đối chứng này. Optional `image_class_embeds=None` giữ nguyên hành vi inference.

Tham khảo: [WaffleCLIP ICCV 2023](https://arxiv.org/abs/2306.07282),
[code chính thức](https://github.com/ExplainableML/WaffleCLIP),
[DCLIP ICLR 2023](https://arxiv.org/abs/2210.07183),
[code chính thức](https://github.com/sachit-menon/classify_by_description_release).

## Đầu ra và cách đọc

Mỗi lần tạo thư mục mới `results/adapted_<method>/<dataset>/<task>/samples_N/<model>/<task>_<unique>/` (hoặc `adapted_suite` khi chạy nhiều phương pháp), không ghi đè:

- `frozen_plan.json`: tất cả prompt đã render, config, seed, commit nguồn và hash code;
  lưu **trước** khi load model/đọc data. Prepare-only cũng tạo file này; lệnh chạy thật
  tạo plan mới nên thay config giữa hai lệnh sẽ tạo thí nghiệm khác.
- `summary.tsv`: từng phương pháp/seed, metric fusion và metric ảnh/text để đối chiếu.
- `seed_summary.tsv`: mean/std theo method và model; std quần thể `ddof=0` qua seed,
  không phải khoảng tin cậy. Một lần chạy để trống std. Không chọn seed tốt nhất.
- `raw_predictions/`: các file dự đoán, lỗi, manifest, diagnostics và snapshot như cũ.
- `failure.json` nếu lỗi sau khi model đã load; giữ log và summary đã hoàn thành,
  không xuất seed_summary cho một sweep chưa chạy hết.

Kiểm tra `text_macro_f1`/điểm text phải giữ nguyên giữa các phương pháp cùng model.
So waffle với class_name để đo nhiễu ensemble; so current_mean_sim với baseline
để đo aggregation; cuối cùng so mọi phương pháp với baseline thực tế.

Đây là `frozen_no_fit_exploratory`: không học từ nhãn, nhưng val đã được dùng nhiều
lần để phát triển pipeline. Không coi kết quả này là test độc lập hoặc tự nhận
zero-shot thuần về toàn bộ lịch sử prompt. Không tự chỉnh prompt theo lỗi vừa xem.

Trong runner thích nghi, input được encode lại mỗi trial; model chỉ load một lần.
Chưa có bằng chứng phương pháp nào tốt hơn baseline trên dữ liệu thật cho tới khi
chạy đánh giá.

## Kiểm thử

`tests/test_paper_methods.py` dùng dữ liệu/ảnh giả: seed, suffix chung, bảo vệ pickle,
nhánh text không đổi, fallback text rỗng, softmax, nhãn, descriptor schema, summary,
chunk max_sim và các runner của cả 5 task. Không tải model hoặc mở dữ liệu dự án.

```sh
python3 -B -m unittest discover -s tests -p test_paper_methods.py
```

Sau khi dọn cấu trúc, lệnh unittest discover chuẩn chạy trực tiếp. Công cụ thống kê
đã hủy được loại khỏi code hoạt động và lưu bản sao trong backup.

## Clone lại trên máy khác

```sh
mkdir -p third_party
git clone https://github.com/ExplainableML/WaffleCLIP.git third_party/WaffleCLIP
git -C third_party/WaffleCLIP checkout 7a1b8ee48e31285f62ecd839fecb6b89cbef81f1
git clone https://github.com/sachit-menon/classify_by_description_release.git third_party/DCLIP
git -C third_party/DCLIP checkout d1a3d8920af89bd0d6a692ed3afeb277df46e082
git clone https://github.com/openai/CLIP.git third_party/OpenAI_CLIP
git -C third_party/OpenAI_CLIP checkout d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
```

Sau khi clone, cài phụ thuộc dự án bằng `pip install -r requirements.txt` và
`pip install -r requirements/research.txt`.

Runner upstream xác minh cả ba commit và checkout sạch. Runner thích nghi cần hai
repo WaffleCLIP/DCLIP. Không chạy pip/setup từ repo tác giả; phần dùng chung của
dự án được khai báo trong `requirements.txt` và `requirements/research.txt`.
Giữ checkout DCLIP nguyên bản; xác nhận điều khoản của repo trước khi phát tán
phần code nguồn đó. Xem `docs/waffleclip_license.txt` về thông báo code đã thích nghi.
