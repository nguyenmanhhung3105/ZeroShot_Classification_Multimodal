# CIFAR-100: chọn top-2 nhóm và chấm lại bằng mô tả

Đây là ứng viên phát triển sau khi xem kết quả dev của phân cấp cứng/mềm. Chưa
có bằng chứng vượt flat; không gọi kết quả dev là xác nhận độc lập hoặc phương
pháp mới. Không sửa hay ghi đè kết quả chạy cũ.

## Thiết kế cố định

- Giữ MetaCLIP2 ViT-H/14 worldwide FP32, batch ảnh 8, logit scale 100 và 10.000
  ID dev seed 0 như lượt trước. Không fine-tune hoặc dùng nhãn để định tuyến.
- 100 prompt tên lớp giữ nguyên `a photo of {label}.`.
- `configs/descriptors/cifar100_visual_v1.json` chứa đúng 100 tên lớp chính thức,
  mỗi lớp hai mô tả đặc điểm nhìn thấy. Mô tả soạn thủ công từ tên lớp/kiến thức
  chung, không xem ảnh hay lỗi từng mẫu. Không xem các mô tả này là annotation
  của từng ảnh. Một số thuộc tính chỉ thường gặp, không đúng mọi cá thể.
- Prompt mô tả: `a photo of {label}, {description}.`; trung bình hai embedding
  prompt unit rồi chuẩn hóa prototype unit. Cùng bank cho hai đối chứng mô tả.
- Tầng 1 cộng xác suất của 100 lớp theo 20 nhóm. Giữ đúng top-2 nhóm, tie theo
  ID nhóm nhỏ hơn. Tầng 2 chỉ chọn trong 10 lớp con của hai nhóm đó.
- Tầng 1 luôn dùng prompt tên lớp, không dùng điểm mô tả để thay gate. Mỗi ảnh
  chỉ mã hóa một lần trong job; cùng feature cho cả bốn đối chứng.

## Đối chứng

| Tên artifact | Điểm dùng dự đoán |
|---|---|
| flat_names | Tên lớp trên đủ 100 lớp, mốc flat gốc |
| flat_descriptions | Mô tả trên đủ 100 lớp |
| top2_names | Giữ 10 lớp ứng viên rồi dùng điểm tên lớp cũ |
| top2_descriptions | Giữ cùng 10 ứng viên rồi dùng điểm mô tả |

Chấm mô tả của đủ 100 lớp cùng một lần để có baseline công bằng; hạn chế ứng
viên ở bước quyết định. Không tuyên bố tiết kiệm compute encoder từ việc lọc này.
Đây là rerank bằng CLIP và prompt mới, không phải LMM suy luận lại từng ảnh.

Không cộng điểm nhóm rồi nhân conditional để tạo lại phân phối flat. Gate top-2
có thể loại cả lớp top-1 fine trong một số trường hợp: group mass khác max fine.
`top2_names` đo chính xác tác động đó thay vì mặc định gate không bao giờ làm hại.

Muốn kết luận lợi ích của chia tầng, cần xem `top2_descriptions` so với cả
`flat_names` và `flat_descriptions`, kèm số sửa đúng/làm sai. Top-2 group recall
97,42% ở chẩn đoán cũ chỉ là khả năng giữ ứng viên, không phải accuracy cuối.

## Chạy qua launcher một lệnh

Smoke inference 5 ảnh trước:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job cifar100_rerank_smoke --use-kaggle true --wait
```

Sau khi smoke thành công, chạy full dev:

```sh
venv/bin/python -B scripts/launch_runs.py \
  --launcher-config configs/kaggle_launcher.yaml \
  --job cifar100_rerank_dev --use-kaggle true --wait
```

Launcher kiểm tra trùng job, quota, nguồn CIFAR; gửi GPU, theo dõi đúng version
mỗi 30 giây, tải ZIP metric, kiểm tra checksum và in bảng. Lỗi quota dừng trước
push. `--prepare-only` không truy cập API/dataset/model. Không chạy full ngầm sau
smoke, không tự gửi test và không đụng các job/kết quả cũ. Mất kết nối dùng lệnh
`--resume-stage` launcher in ra, không gửi lại.

Kernel riêng: `nguyenhung3105/vlm-cifar100-rerank-smoke-v1` và
`nguyenhung3105/vlm-cifar100-rerank-dev-v1`.

Lượt trước chỉ lưu logits, không lưu feature ảnh, nên lần đầu thử bộ mô tả mới
vẫn cần mã hóa lại ảnh. Lượt rerank này lưu thêm feature để có thể tái sử dụng
trong phân tích sau; không tải lại Dataset nguồn.

## Artifact và kiểm chứng

- `summary.tsv`, `per_class.json`, `predictions.tsv`: đủ bốn đối chứng cùng ID.
- `ablation_comparisons.tsv`: so mô tả/phẳng, gate/tên lớp, và hệ thống đề xuất
  với cả hai baseline; ghi corrected, harmed và delta accuracy.
- `routing.json`: top-1 nhóm, top-2 group recall và số lớp ứng viên; nhãn thật
  chỉ tham gia đánh giá sau khi dự đoán.
- `replay_scores.npz`: logits tên/mô tả, mapping, nhóm chọn, feature ảnh và xác
  suất bốn phương pháp. Đọc bằng allow_pickle=False.
- `run.json`/`frozen_plan.json`: config, prompt, hash code/descriptor/requirements,
  phiên bản thư viện, fingerprint cohort và model runtime. Có completion/checksum.

Baseline `flat_names` cần khớp gần lượt flat 86,75% cũ trên cùng dev; nếu lệch,
kiểm tra source/cohort/model/precision trước khi diễn giải cải thiện.
Smoke chỉ kiểm tra vận hành, không báo chất lượng từ năm mẫu.

Chưa có CI/kiểm định trong runner; không tự chọn mô tả theo test. Test chỉ được
chuẩn bị riêng sau full dev, cùng protocol và hash descriptor; smoke/cấu hình cũ
không mở được test. Chưa thực thi GPU trong lượt triển khai này.

## Kiểm tra local

Test tên lớp/mô tả đầy đủ, gate độc lập mô tả, group mass khác max fine, tie ổn
định, đúng 10 lớp ứng viên, xác suất chuẩn hóa, corrected/harmed, replay khớp,
end-to-end 5 ảnh tổng hợp và notebook whitelist/launcher riêng. Không đọc ảnh thật.

```sh
venv/bin/python -m unittest discover -s tests -p test_cifar100_rerank.py -v
```
