# Minh họa đầu ra thực tế trong Chương 4

Cập nhật 25/10/2026. Mẫu được chọn theo MetaCLIP2 baseline; đúng/sai được xác định riêng cho từng model.
Chọn mẫu đầu tiên trong thứ tự log thỏa tiêu chí, không chọn theo confidence cao nhất. MM-IMDb dùng Exact Match.
Toàn bộ số lớp và điểm nhánh/fusion được giữ trong JSON; hình và bảng LaTeX được tạo từ cùng dữ liệu này.
Không đọc TSV nguồn, không chạy lại model. Các mẫu không dùng để ước lượng chất lượng toàn dataset.

## Fakeddit — 3psxmz

Trạng thái tham chiếu: **đúng**.
Ảnh thật: `data/processed/pro_fakeddit/images/3psxmz.jpg`.

### 2way

Nhãn annotation: **Real**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Real | Có | 0.6125 | 0.2251 | 0.9631 | — | `16ead71cd477` |
| DFN5B | baseline | Real | Có | 0.5709 | 0.1418 | 0.9854 | — | `5cbdd8e2e5cf` |
| SigLIP2 | baseline | Real | Có | 0.8456 | 0.6912 | 0.6207 | — | `0e323fb32f4f` |
| OpenAI CLIP | baseline | Fake | Không | 0.7182 | 0.4364 | 0.8579 | — | `d3d1f2f79b7c` |

### 6way

Nhãn annotation: **True**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | True | Có | 0.5140 | 0.2564 | 0.7484 | — | `c66fec810255` |
| DFN5B | baseline | True | Có | 0.2721 | 0.0657 | 0.9653 | — | `428af87d7a4a` |
| SigLIP2 | baseline | True | Có | 0.4754 | 0.1094 | 0.6828 | — | `d3347aa759bd` |
| OpenAI CLIP | baseline | Satire/Parody | Không | 0.3215 | 0.0755 | 0.9084 | — | `93b00ff28e11` |

## Fakeddit — 8tlelb

Trạng thái tham chiếu: **sai**.
Ảnh thật: `data/processed/pro_fakeddit/images/8tlelb.jpg`.

### 2way

Nhãn annotation: **Real**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Fake | Không | 0.6076 | 0.2151 | 0.9663 | — | `16ead71cd477` |
| DFN5B | baseline | Real | Có | 0.5257 | 0.0513 | 0.9981 | — | `5cbdd8e2e5cf` |
| SigLIP2 | baseline | Real | Có | 0.6259 | 0.2518 | 0.9538 | — | `0e323fb32f4f` |
| OpenAI CLIP | baseline | Fake | Không | 0.7729 | 0.5457 | 0.7730 | — | `d3d1f2f79b7c` |

### 6way

Nhãn annotation: **True**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | False Connection | Không | 0.4970 | 0.3370 | 0.8109 | — | `c66fec810255` |
| DFN5B | baseline | Imposter | Không | 0.2274 | 0.0338 | 0.9734 | — | `428af87d7a4a` |
| SigLIP2 | baseline | True | Có | 0.4411 | 0.2566 | 0.8271 | — | `d3347aa759bd` |
| OpenAI CLIP | baseline | Satire/Parody | Không | 0.3698 | 0.0846 | 0.8365 | — | `93b00ff28e11` |

## CrisisMMD — 919806240183631872_1

Trạng thái tham chiếu: **đúng**.
Ảnh thật: `data/processed/pro_crisismmd/images/919806240183631872_1.jpg`.

### informativeness

Nhãn annotation: **Informative**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Informative | Có | 0.6736 | 0.3472 | 0.9112 | — | `788bb9578b84` |
| DFN5B | baseline | Not Informative | Không | 0.5190 | 0.0380 | 0.9990 | — | `15d72b19b3b7` |
| SigLIP2 | baseline | Not Informative | Không | 0.7163 | 0.4326 | 0.8604 | — | `2b51d19b28d3` |
| OpenAI CLIP | baseline | Informative | Có | 0.8364 | 0.6729 | 0.6428 | — | `7c507a3ac2e9` |
| MetaCLIP2 | adapted_dclip | Informative | Có | 0.7924 | 0.5847 | 0.7369 | — | `44618e680cf2` |
| DFN5B | adapted_dclip | Not Informative | Không | 0.5199 | 0.0398 | 0.9989 | — | `d06d06ceb255` |
| SigLIP2 | adapted_dclip | Not Informative | Không | 0.7512 | 0.5024 | 0.8093 | — | `33e59b9ea845` |
| OpenAI CLIP | adapted_dclip | Informative | Có | 0.8424 | 0.6849 | 0.6284 | — | `6f2ffe1d181c` |
| MetaCLIP2 | adapted_waffle | Informative | Có | 0.6653 | 0.3307 | 0.9196 | — | `39f6734476ad` |
| DFN5B | adapted_waffle | Not Informative | Không | 0.5896 | 0.1791 | 0.9767 | — | `19f905e6ad58` |
| SigLIP2 | adapted_waffle | Not Informative | Không | 0.8311 | 0.6621 | 0.6553 | — | `c874a93229ad` |
| OpenAI CLIP | adapted_waffle | Informative | Có | 0.6573 | 0.3145 | 0.9274 | — | `273c7b6138cf` |

### humanitarian

Nhãn annotation: **Rescue/Volunteer/Donation**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Rescue/Volunteer/Donation | Có | 0.5080 | 0.1619 | 0.5595 | — | `520ebf75825a` |
| DFN5B | baseline | Rescue/Volunteer/Donation | Có | 0.2042 | 0.0453 | 0.9781 | — | `ab76540ca808` |
| SigLIP2 | baseline | Not Humanitarian | Không | 0.7802 | 0.6962 | 0.4147 | — | `908c56201581` |
| OpenAI CLIP | baseline | Rescue/Volunteer/Donation | Có | 0.5410 | 0.4138 | 0.7372 | — | `3388e0f5d4ff` |
| MetaCLIP2 | adapted_dclip | Rescue/Volunteer/Donation | Có | 0.6859 | 0.5864 | 0.5578 | — | `360029407f84` |
| DFN5B | adapted_dclip | Rescue/Volunteer/Donation | Có | 0.1945 | 0.0292 | 0.9821 | — | `cdfb0ec9fbed` |
| SigLIP2 | adapted_dclip | Not Humanitarian | Không | 0.6015 | 0.4240 | 0.6215 | — | `192b5b63ad16` |
| OpenAI CLIP | adapted_dclip | Rescue/Volunteer/Donation | Có | 0.5481 | 0.4169 | 0.7113 | — | `bf574f3316ba` |
| MetaCLIP2 | adapted_waffle | Rescue/Volunteer/Donation | Có | 0.6310 | 0.4427 | 0.5593 | — | `a9722c05f361` |
| DFN5B | adapted_waffle | Rescue/Volunteer/Donation | Có | 0.2050 | 0.0432 | 0.9819 | — | `c3053205ec6a` |
| SigLIP2 | adapted_waffle | Not Humanitarian | Không | 0.5017 | 0.3848 | 0.7858 | — | `36e2ce0dfbee` |
| OpenAI CLIP | adapted_waffle | Rescue/Volunteer/Donation | Có | 0.5443 | 0.4001 | 0.7205 | — | `1b1480e95e3d` |

## CrisisMMD — 910691545577345026_0

Trạng thái tham chiếu: **sai**.
Ảnh thật: `data/processed/pro_crisismmd/images/910691545577345026_0.jpg`.

### informativeness

Nhãn annotation: **Not Informative**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Informative | Không | 0.8523 | 0.7046 | 0.6040 | — | `788bb9578b84` |
| DFN5B | baseline | Informative | Không | 0.5933 | 0.1867 | 0.9747 | — | `15d72b19b3b7` |
| SigLIP2 | baseline | Informative | Không | 0.7755 | 0.5511 | 0.7682 | — | `2b51d19b28d3` |
| OpenAI CLIP | baseline | Informative | Không | 0.8359 | 0.6717 | 0.6441 | — | `7c507a3ac2e9` |
| MetaCLIP2 | adapted_dclip | Informative | Không | 0.8799 | 0.7598 | 0.5296 | — | `44618e680cf2` |
| DFN5B | adapted_dclip | Informative | Không | 0.6378 | 0.2756 | 0.9445 | — | `d06d06ceb255` |
| SigLIP2 | adapted_dclip | Informative | Không | 0.7607 | 0.5213 | 0.7939 | — | `33e59b9ea845` |
| OpenAI CLIP | adapted_dclip | Informative | Không | 0.7094 | 0.4188 | 0.8695 | — | `6f2ffe1d181c` |
| MetaCLIP2 | adapted_waffle | Informative | Không | 0.5987 | 0.1975 | 0.9717 | — | `39f6734476ad` |
| DFN5B | adapted_waffle | Informative | Không | 0.5423 | 0.0847 | 0.9948 | — | `19f905e6ad58` |
| SigLIP2 | adapted_waffle | Not Informative | Có | 0.5243 | 0.0486 | 0.9983 | — | `c874a93229ad` |
| OpenAI CLIP | adapted_waffle | Informative | Không | 0.6003 | 0.2007 | 0.9707 | — | `273c7b6138cf` |

### humanitarian

Nhãn annotation: **Not Humanitarian**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Infrastructure Damage | Không | 0.4538 | 0.0362 | 0.5471 | — | `520ebf75825a` |
| DFN5B | baseline | Infrastructure Damage | Không | 0.2525 | 0.1057 | 0.9582 | — | `ab76540ca808` |
| SigLIP2 | baseline | Infrastructure Damage | Không | 0.4538 | 0.1941 | 0.5777 | — | `908c56201581` |
| OpenAI CLIP | baseline | Infrastructure Damage | Không | 0.4147 | 0.1791 | 0.7823 | — | `3388e0f5d4ff` |
| MetaCLIP2 | adapted_dclip | Other Information | Không | 0.5299 | 0.1108 | 0.4459 | — | `360029407f84` |
| DFN5B | adapted_dclip | Infrastructure Damage | Không | 0.2440 | 0.0746 | 0.9520 | — | `cdfb0ec9fbed` |
| SigLIP2 | adapted_dclip | Infrastructure Damage | Không | 0.5063 | 0.2440 | 0.5132 | — | `192b5b63ad16` |
| OpenAI CLIP | adapted_dclip | Infrastructure Damage | Không | 0.5901 | 0.3685 | 0.5756 | — | `bf574f3316ba` |
| MetaCLIP2 | adapted_waffle | Infrastructure Damage | Không | 0.5608 | 0.1534 | 0.4066 | — | `a9722c05f361` |
| DFN5B | adapted_waffle | Infrastructure Damage | Không | 0.2692 | 0.1194 | 0.9536 | — | `c3053205ec6a` |
| SigLIP2 | adapted_waffle | Infrastructure Damage | Không | 0.5377 | 0.3012 | 0.5034 | — | `36e2ce0dfbee` |
| OpenAI CLIP | adapted_waffle | Infrastructure Damage | Không | 0.5844 | 0.3667 | 0.6060 | — | `1b1480e95e3d` |

## MM-IMDb — 0164538

Trạng thái tham chiếu: **đúng**.
Ảnh thật: `data/processed/pro_mmimdb/images/0164538.jpeg`.

### genres

Nhãn annotation: **Drama, Romance**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Drama, Romance | Có | — | — | — | 1.0000 | `3aae524797f3` |
| DFN5B | baseline | Biography, Drama, Romance | Không | — | — | — | 0.8000 | `5932aaddb554` |
| SigLIP2 | baseline | Drama, Romance, Short | Không | — | — | — | 0.8000 | `e17746c1e744` |
| OpenAI CLIP | baseline | Drama, Romance | Có | — | — | — | 1.0000 | `955d550a8310` |

## MM-IMDb — 0443335

Trạng thái tham chiếu: **sai**.
Ảnh thật: `data/processed/pro_mmimdb/images/0443335.jpeg`.

### genres

Nhãn annotation: **Documentary**.

| Model | Phương pháp | Dự đoán | Đúng / Exact | Confidence | Margin | Entropy chuẩn hóa | F1 mẫu | Run |
|---|---|---|---|---:|---:|---:|---:|---|
| MetaCLIP2 | baseline | Biography, Music, Musical | Không | — | — | — | 0.0000 | `3aae524797f3` |
| DFN5B | baseline | Biography, Music, Musical | Không | — | — | — | 0.0000 | `5932aaddb554` |
| SigLIP2 | baseline | Documentary, Music, Musical, Short | Không | — | — | — | 0.4000 | `e17746c1e744` |
| OpenAI CLIP | baseline | Music | Không | — | — | — | 0.0000 | `955d550a8310` |

## Tái tạo

```sh
venv/bin/python -B scripts/build_report_examples.py
```

Snapshot kiểm tra hash text và kích thước ảnh; không có hash từng ảnh lịch sử trong snapshot. Hash SHA-256 của ảnh hiện tại và bản sao được ghi lại để tái kiểm tra.
