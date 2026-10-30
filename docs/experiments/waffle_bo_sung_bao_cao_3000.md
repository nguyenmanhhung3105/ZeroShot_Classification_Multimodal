# Waffle adapted — chạy như DCLIP cho báo cáo giữa kỳ

Cập nhật 25/10/2026. Phần Waffle đã được đưa vào cơ sở lý thuyết, mô hình và Chương 4
của `docs/bao_cao_giua_ky.tex`. Ma trận định lượng đã đủ **36 lượt / 108.000 dòng dự đoán**. Tám lượt Waffle
3.000 mẫu đã kiểm tra biên nhận, đầu vào, nhãn, điểm tweet và fusion; kết quả đã
đưa vào báo cáo. Macro-F1 Waffle thấp hơn DCLIP ở cả tám đối chứng.

## Phạm vi đã chốt

- **Waffle adapted**, thay prompt nhánh ảnh; tweet và fusion giữ như DCLIP adapted.
- Chỉ CrisisMMD: bốn backbone × hai task = **tám lượt mới**, 3.000 mẫu/lượt, seed prompt 0.
- 15 cặp từ/ký tự = 30 prompt/lớp; nhánh ảnh mean_sim, tweet mean_embed, fusion 0,5/0,5.
- Tắt quan hệ ảnh–văn bản và ensemble; không đưa các biến thể Waffle lab vào báo cáo này.
- Dùng cùng cohort CrisisMMD và nhãn ảnh với baseline/DCLIP hiện có. Không lấy kết quả 1.000 mẫu cũ để so.

Trên CrisisMMD, Waffle và DCLIP cùng mean_sim ở nhánh ảnh; so với baseline mean_embed
vẫn thay cả prompt và aggregate, nên không quy toàn bộ chênh lệch cho riêng nội dung prompt.
Không cần chạy Waffle cho Fakeddit/MM-IMDb trong phạm vi giữa kỳ đã chốt. Các phần
baseline và hai mẫu minh họa mỗi dataset vẫn giữ nguyên cho cả ba bộ.

## Các lệnh đã hoàn thành

Không cần chạy lại tám lệnh dưới đây. Giữ các lệnh để tái tạo thí nghiệm khi cần;
mỗi lệnh chỉ một model, một task và một seed. Kết quả hợp lệ đã được đối chiếu bằng
`scripts/analyze_report_3000.py` và lưu tại `ket_qua_3000_kiem_tra.json`.

### Informativeness

```bash
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model metaclip2_vith14 --method waffle --seed 0 --max-samples 3000 --use-kaggle true
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model dfn5b_vith14 --method waffle --seed 0 --max-samples 3000 --use-kaggle true
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model siglip2_so400m --method waffle --seed 0 --max-samples 3000 --use-kaggle true
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_info --model openai_clip_vit_b_32 --method waffle --seed 0 --max-samples 3000 --use-kaggle true
```

### Humanitarian

```bash
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model metaclip2_vith14 --method waffle --seed 0 --max-samples 3000 --use-kaggle true
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model dfn5b_vith14 --method waffle --seed 0 --max-samples 3000 --use-kaggle true
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model siglip2_so400m --method waffle --seed 0 --max-samples 3000 --use-kaggle true
python -B scripts/launch_runs.py --launcher-config configs/launcher_3000.yaml --job adapted_crisismmd_humanitarian --model openai_clip_vit_b_32 --method waffle --seed 0 --max-samples 3000 --use-kaggle true
```

## Nhận kết quả và đối chiếu

Kết quả lưu theo:

```text
results/adapted_waffle/crisismmd/<task>/samples_3000/<model>/<lượt>/
```

Launcher đã tải `summary.tsv`, `run_receipt.json` và `run_artifacts.zip` về local.
Cả tám lượt đủ 3.000 ID, cùng đầu vào với baseline/DCLIP. Nhánh text khớp chính xác;
điểm fusion khớp trọng số cố định. Có 36 lượt và 108.000 dòng dự đoán được xác thực.

## Sáu mẫu minh họa

Chương 4 có sáu mẫu thực tế, hiển thị bốn backbone. Các mẫu CrisisMMD có đủ cả
baseline, DCLIP và Waffle, ghép theo cùng ID, ảnh, text và nhãn; tổng cộng 72 đầu ra.
Bảng và heatmap được tạo lại từ log, không chạy inference mới.
