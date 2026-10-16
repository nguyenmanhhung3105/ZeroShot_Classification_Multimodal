# Tra cứu kết quả

Mỗi dòng là **một model hoặc một phương pháp** trong một lượt chạy. Bấm `summary` để xem chỉ số đầy đủ; bấm `thư mục` để xem log và `run_artifacts.zip`. Bảng được cập nhật tự động sau khi launcher tải kết quả.

`runs/`: chạy chuẩn; `experiments/`: adapted, upstream, ensemble; `analysis/`: phân tích offline; `launches/`: dữ liệu trung gian của launcher. Các thư mục lịch sử khác giữ nguyên đường dẫn cũ. `current_prompts` nghĩa là prompt gốc của dự án; cột quan hệ ảnh–text ghi rõ flag fusion bổ sung.

Muốn tìm bằng lệnh: `python -B scripts/list_results.py --dataset crisismmd --task humanitarian --limit 20`. Các lượt cùng số mẫu chưa chắc dùng cùng ID mẫu; cần kiểm tra manifest trước khi so sánh chính thức.

## crisismmd / humanitarian (5 dòng)

| Ngày giờ | Job | Model | Phương pháp | Quan hệ ảnh–text | Mẫu | Accuracy | Macro-F1 | Kết quả |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-10-15 23:41 | crisismmd_humanitarian_direct_pair | metaclip2_vith14 | current_prompts | direct_pair_fusion | 1000 | 0.7360 | 0.4438 | [summary](runs/crisismmd_humanitarian_direct_pair/45948c7b7aa6/summary.tsv) · [thư mục](runs/crisismmd_humanitarian_direct_pair/45948c7b7aa6/) |
| 2026-10-14 13:24 | upstream_crisismmd_humanitarian | openai_clip_vit_b_32 | clip / original_image | — | 1000 | 0.4240 | 0.2776 | [summary](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/summary.tsv) · [thư mục](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/) |
| 2026-10-14 13:24 | upstream_crisismmd_humanitarian | openai_clip_vit_b_32 | waffle / original_image / seed 0.0 | — | 1000 | 0.4740 | 0.3007 | [summary](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/summary.tsv) · [thư mục](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/) |
| 2026-10-14 13:24 | upstream_crisismmd_humanitarian | openai_clip_vit_b_32 | dclip_labels / original_image | — | 1000 | 0.3220 | 0.2114 | [summary](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/summary.tsv) · [thư mục](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/) |
| 2026-10-14 13:24 | upstream_crisismmd_humanitarian | openai_clip_vit_b_32 | dclip / original_image | — | 1000 | 0.4580 | 0.3359 | [summary](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/summary.tsv) · [thư mục](experiments/upstream_crisismmd_humanitarian/9cebbce1f45d/) |

## crisismmd / informativeness (18 dòng)

| Ngày giờ | Job | Model | Phương pháp | Quan hệ ảnh–text | Mẫu | Accuracy | Macro-F1 | Kết quả |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-10-15 23:23 | crisismmd_info_agreement | metaclip2_vith14 | current_prompts | agreement_fusion | 1000 | 0.8060 | 0.8059 | [summary](runs/crisismmd_info_agreement/8d71e3c908af/summary.tsv) · [thư mục](runs/crisismmd_info_agreement/8d71e3c908af/) |
| 2026-10-14 21:21 | crisismmd_info | metaclip2_vith14 | current_prompts | — | 3000 | 0.8243 | 0.8243 | [summary](runs/crisismmd_info/6acc864b7851/summary.tsv) · [thư mục](runs/crisismmd_info/6acc864b7851/) |
| 2026-10-14 20:58 | adapted_crisismmd_info | metaclip2_vith14 | dclip | — | 3000 | 0.8657 | 0.8645 | [summary](experiments/adapted_crisismmd_info/03f1882357bb/summary.tsv) · [thư mục](experiments/adapted_crisismmd_info/03f1882357bb/) |
| 2026-10-14 20:36 | adapted_crisismmd_info | metaclip2_vith14 | dclip | — | 1000 | 0.8340 | 0.8325 | [summary](experiments/adapted_crisismmd_info/2ff98e4f0871/summary.tsv) · [thư mục](experiments/adapted_crisismmd_info/2ff98e4f0871/) |
| 2026-10-14 20:29 | adapted_crisismmd_info | metaclip2_vith14 | waffle / seed 0 | — | 1000 | 0.7290 | 0.7285 | [summary](experiments/adapted_crisismmd_info/9595e1f70593/summary.tsv) · [thư mục](experiments/adapted_crisismmd_info/9595e1f70593/) |
| 2026-10-14 20:21 | adapted_crisismmd_info | metaclip2_vith14 | baseline | — | 1000 | 0.8060 | 0.8059 | [summary](experiments/adapted_crisismmd_info/72686f1b262a/summary.tsv) · [thư mục](experiments/adapted_crisismmd_info/72686f1b262a/) |
| 2026-10-14 20:10 | upstream_crisismmd_info | openai_clip_vit_b_32 | dclip / original_image | — | 1000 | 0.7750 | 0.7681 | [summary](experiments/upstream_crisismmd_info/f59288d1c77e/summary.tsv) · [thư mục](experiments/upstream_crisismmd_info/f59288d1c77e/) |
| 2026-10-14 20:04 | upstream_crisismmd_info | openai_clip_vit_b_32 | dclip_labels / original_image | — | 1000 | 0.3890 | 0.3350 | [summary](experiments/upstream_crisismmd_info/ea2e52baee6c/summary.tsv) · [thư mục](experiments/upstream_crisismmd_info/ea2e52baee6c/) |
| 2026-10-14 19:59 | upstream_crisismmd_info | openai_clip_vit_b_32 | waffle / original_image / seed 0 | — | 1000 | 0.5570 | 0.5141 | [summary](experiments/upstream_crisismmd_info/61fa6fcc913c/summary.tsv) · [thư mục](experiments/upstream_crisismmd_info/61fa6fcc913c/) |
| 2026-10-14 19:49 | upstream_crisismmd_info | openai_clip_vit_b_32 | clip / original_image | — | 1000 | 0.5010 | 0.4903 | [summary](experiments/upstream_crisismmd_info/5c06beff2492/summary.tsv) · [thư mục](experiments/upstream_crisismmd_info/5c06beff2492/) |
| 2026-10-13 23:48 | crisismmd_info | siglip2_so400m | current_prompts | — | 1000 | 0.7040 | 0.6954 | [summary](runs/crisismmd_info/e595fbb7f79a/summary.tsv) · [thư mục](runs/crisismmd_info/e595fbb7f79a/) |
| 2026-10-13 23:48 | crisismmd_info | metaclip2_vith14 | current_prompts | — | 1000 | 0.8060 | 0.8059 | [summary](runs/crisismmd_info/e595fbb7f79a/summary.tsv) · [thư mục](runs/crisismmd_info/e595fbb7f79a/) |
| 2026-10-13 23:48 | crisismmd_info | dfn5b_vith14 | current_prompts | — | 1000 | 0.6310 | 0.6022 | [summary](runs/crisismmd_info/e595fbb7f79a/summary.tsv) · [thư mục](runs/crisismmd_info/e595fbb7f79a/) |
| 2026-10-13 23:27 | crisismmd_info | siglip2_so400m | base_model | — | 5 | 0.8000 | 0.7619 | [summary](experiments/crisismmd_info/d4f7cbd6281e/summary.tsv) · [thư mục](experiments/crisismmd_info/d4f7cbd6281e/) |
| 2026-10-13 23:27 | crisismmd_info | metaclip2_vith14 | base_model | — | 5 | 1.0000 | 1.0000 | [summary](experiments/crisismmd_info/d4f7cbd6281e/summary.tsv) · [thư mục](experiments/crisismmd_info/d4f7cbd6281e/) |
| 2026-10-13 23:27 | crisismmd_info | ensemble | equal_mean | — | 5 | 0.8000 | 0.7619 | [summary](experiments/crisismmd_info/d4f7cbd6281e/summary.tsv) · [thư mục](experiments/crisismmd_info/d4f7cbd6281e/) |
| 2026-10-13 23:27 | crisismmd_info | ensemble | margin_gate | — | 5 | 1.0000 | 1.0000 | [summary](experiments/crisismmd_info/d4f7cbd6281e/summary.tsv) · [thư mục](experiments/crisismmd_info/d4f7cbd6281e/) |
| 2026-10-13 23:27 | crisismmd_info | dfn5b_vith14 | base_model | — | 5 | 0.4000 | 0.4000 | [summary](experiments/crisismmd_info/d4f7cbd6281e/summary.tsv) · [thư mục](experiments/crisismmd_info/d4f7cbd6281e/) |

## fakeddit / 2way (13 dòng)

| Ngày giờ | Job | Model | Phương pháp | Quan hệ ảnh–text | Mẫu | Accuracy | Macro-F1 | Kết quả |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-10-15 13:17 | fakeddit_2way | siglip2_so400m | base_model | — | 1000 | 0.5970 | 0.4453 | [summary](experiments/fakeddit_2way/5c026d594111/summary.tsv) · [thư mục](experiments/fakeddit_2way/5c026d594111/) |
| 2026-10-15 13:17 | fakeddit_2way | metaclip2_vith14 | base_model | — | 1000 | 0.5910 | 0.5190 | [summary](experiments/fakeddit_2way/5c026d594111/summary.tsv) · [thư mục](experiments/fakeddit_2way/5c026d594111/) |
| 2026-10-15 13:17 | fakeddit_2way | ensemble | equal_mean | — | 1000 | 0.6030 | 0.4656 | [summary](experiments/fakeddit_2way/5c026d594111/summary.tsv) · [thư mục](experiments/fakeddit_2way/5c026d594111/) |
| 2026-10-15 13:17 | fakeddit_2way | ensemble | margin_gate | — | 1000 | 0.5980 | 0.4936 | [summary](experiments/fakeddit_2way/5c026d594111/summary.tsv) · [thư mục](experiments/fakeddit_2way/5c026d594111/) |
| 2026-10-15 13:17 | fakeddit_2way | dfn5b_vith14 | base_model | — | 1000 | 0.6000 | 0.4750 | [summary](experiments/fakeddit_2way/5c026d594111/summary.tsv) · [thư mục](experiments/fakeddit_2way/5c026d594111/) |
| 2026-10-15 12:58 | fakeddit_2way | siglip2_so400m | current_prompts / current | — | 1000 | 0.5970 | 0.4453 | [summary](runs/fakeddit_2way/9baaa61e42f6/summary.tsv) · [thư mục](runs/fakeddit_2way/9baaa61e42f6/) |
| 2026-10-15 12:42 | adapted_fakeddit_2way | siglip2_so400m | dclip / current | — | 1000 | 0.5500 | 0.5146 | [summary](experiments/adapted_fakeddit_2way/4f1a23ced16d/summary.tsv) · [thư mục](experiments/adapted_fakeddit_2way/4f1a23ced16d/) |
| 2026-10-15 10:40 | adapted_fakeddit_2way | metaclip2_vith14 | dclip / current | — | 1000 | 0.5230 | 0.5106 | [summary](experiments/adapted_fakeddit_2way/fbbe3654b904/summary.tsv) · [thư mục](experiments/adapted_fakeddit_2way/fbbe3654b904/) |
| 2026-10-15 10:22 | fakeddit_2way | metaclip2_vith14 | current_prompts / current | — | 1000 | 0.5910 | 0.5190 | [summary](runs/fakeddit_2way/9ed97f55cf78/summary.tsv) · [thư mục](runs/fakeddit_2way/9ed97f55cf78/) |
| 2026-10-15 10:11 | fakeddit_2way | metaclip2_vith14 | current_prompts / current | — | 1000 | 0.5910 | 0.5190 | [summary](runs/fakeddit_2way/531b142a5d92/summary.tsv) · [thư mục](runs/fakeddit_2way/531b142a5d92/) |
| 2026-10-14 00:15 | fakeddit_2way | siglip2_so400m | current_prompts / current | — | 1000 | 0.5970 | 0.4453 | [summary](runs/fakeddit_2way/d85f0353d7d3/summary.tsv) · [thư mục](runs/fakeddit_2way/d85f0353d7d3/) |
| 2026-10-14 00:15 | fakeddit_2way | metaclip2_vith14 | current_prompts / current | — | 1000 | 0.5910 | 0.5190 | [summary](runs/fakeddit_2way/d85f0353d7d3/summary.tsv) · [thư mục](runs/fakeddit_2way/d85f0353d7d3/) |
| 2026-10-14 00:15 | fakeddit_2way | dfn5b_vith14 | current_prompts / current | — | 1000 | 0.6000 | 0.4750 | [summary](runs/fakeddit_2way/d85f0353d7d3/summary.tsv) · [thư mục](runs/fakeddit_2way/d85f0353d7d3/) |

## mmimdb / genres (3 dòng)

| Ngày giờ | Job | Model | Phương pháp | Quan hệ ảnh–text | Mẫu | Accuracy | Macro-F1 | Kết quả |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-10-14 00:45 | mmimdb_genres | siglip2_so400m | current_prompts | — | 1000 | — | 0.4837 | [summary](runs/mmimdb_genres/289a8ffa6843/summary.tsv) · [thư mục](runs/mmimdb_genres/289a8ffa6843/) |
| 2026-10-14 00:45 | mmimdb_genres | metaclip2_vith14 | current_prompts | — | 1000 | — | 0.5075 | [summary](runs/mmimdb_genres/289a8ffa6843/summary.tsv) · [thư mục](runs/mmimdb_genres/289a8ffa6843/) |
| 2026-10-14 00:45 | mmimdb_genres | dfn5b_vith14 | current_prompts | — | 1000 | — | 0.4374 | [summary](runs/mmimdb_genres/289a8ffa6843/summary.tsv) · [thư mục](runs/mmimdb_genres/289a8ffa6843/) |

## Lượt chưa có dòng kết quả (46)

Các `summary.tsv` sau chỉ có tiêu đề. Chúng **không được tính là kết quả model**. Có thể mở log/receipt trong thư mục để tìm nguyên nhân; chưa tự xóa dữ liệu nào.

- `crisismmd_info` / `26e3f06a0c8d`: [thư mục](runs/crisismmd_info/26e3f06a0c8d/)
- `crisismmd_info` / `6f64594cad0f`: [thư mục](runs/crisismmd_info/6f64594cad0f/)
- `crisismmd_info` / `f8f1fb51f99b`: [thư mục](runs/crisismmd_info/f8f1fb51f99b/)
- `crisismmd_info` / `59ec4767d4d8`: [thư mục](runs/crisismmd_info/59ec4767d4d8/)
- `crisismmd_info` / `6886d3766d86`: [thư mục](runs/crisismmd_info/6886d3766d86/)
- `crisismmd_info` / `6d2cd77c37e2`: [thư mục](runs/crisismmd_info/6d2cd77c37e2/)
- `crisismmd_info` / `dc1395cff706`: [thư mục](runs/crisismmd_info/dc1395cff706/)
- `crisismmd_info` / `00c6b9bb5d5a`: [thư mục](runs/crisismmd_info/00c6b9bb5d5a/)
- `crisismmd_info` / `0bd7439085b8`: [thư mục](runs/crisismmd_info/0bd7439085b8/)
- `crisismmd_info` / `c1de1c3baa44`: [thư mục](runs/crisismmd_info/c1de1c3baa44/)
- `crisismmd_info` / `e78a0c494851`: [thư mục](runs/crisismmd_info/e78a0c494851/)
- `crisismmd_info` / `b39c69199718`: [thư mục](runs/crisismmd_info/b39c69199718/)
- `crisismmd_info` / `b2cef86f98cd`: [thư mục](runs/crisismmd_info/b2cef86f98cd/)
- `crisismmd_info` / `96dd3fffc78c`: [thư mục](runs/crisismmd_info/96dd3fffc78c/)
- `crisismmd_info` / `0ab0fc58ec2e`: [thư mục](runs/crisismmd_info/0ab0fc58ec2e/)
- `crisismmd_info` / `9e1dd5896f82`: [thư mục](runs/crisismmd_info/9e1dd5896f82/)
- `crisismmd_info` / `7a45a8b63fca`: [thư mục](runs/crisismmd_info/7a45a8b63fca/)
- `crisismmd_info` / `17cd912393e2`: [thư mục](runs/crisismmd_info/17cd912393e2/)
- `crisismmd_info` / `7e4ac646681b`: [thư mục](runs/crisismmd_info/7e4ac646681b/)
- `crisismmd_info` / `a8b67bb9dfb8`: [thư mục](runs/crisismmd_info/a8b67bb9dfb8/)
- `crisismmd_info` / `61636652556d`: [thư mục](runs/crisismmd_info/61636652556d/)
- `crisismmd_info` / `407e14739ac3`: [thư mục](runs/crisismmd_info/407e14739ac3/)
- `crisismmd_info` / `ee95d8f4a67e`: [thư mục](runs/crisismmd_info/ee95d8f4a67e/)
- `crisismmd_info` / `d0547510eb1d`: [thư mục](runs/crisismmd_info/d0547510eb1d/)
- `crisismmd_info` / `6116bba41706`: [thư mục](runs/crisismmd_info/6116bba41706/)
- `crisismmd_info` / `99cf7512016f`: [thư mục](runs/crisismmd_info/99cf7512016f/)
- `crisismmd_info` / `c8c840f3e179`: [thư mục](runs/crisismmd_info/c8c840f3e179/)
- `crisismmd_info` / `67aa66800e52`: [thư mục](runs/crisismmd_info/67aa66800e52/)
- `crisismmd_info` / `5c7d1d50f70b`: [thư mục](runs/crisismmd_info/5c7d1d50f70b/)
- `crisismmd_info` / `fc43a4d5cb34`: [thư mục](runs/crisismmd_info/fc43a4d5cb34/)
- `crisismmd_info` / `497d0f6ff0f6`: [thư mục](runs/crisismmd_info/497d0f6ff0f6/)
- `crisismmd_info` / `75cf8792674c`: [thư mục](runs/crisismmd_info/75cf8792674c/)
- `crisismmd_info` / `89a393137b84`: [thư mục](runs/crisismmd_info/89a393137b84/)
- `crisismmd_info` / `be9979cd0783`: [thư mục](runs/crisismmd_info/be9979cd0783/)
- `crisismmd_info` / `4e6ac3ca4131`: [thư mục](runs/crisismmd_info/4e6ac3ca4131/)
- `crisismmd_info` / `3f7d7eff38d3`: [thư mục](runs/crisismmd_info/3f7d7eff38d3/)
- `crisismmd_info` / `1a7c2b3c8106`: [thư mục](runs/crisismmd_info/1a7c2b3c8106/)
- `crisismmd_info` / `5ed5f373d1f3`: [thư mục](runs/crisismmd_info/5ed5f373d1f3/)
- `crisismmd_info` / `d156546fbf81`: [thư mục](runs/crisismmd_info/d156546fbf81/)
- `crisismmd_info` / `abe7fc933bd8`: [thư mục](runs/crisismmd_info/abe7fc933bd8/)
- `crisismmd_info` / `44a7e21121bc`: [thư mục](runs/crisismmd_info/44a7e21121bc/)
- `crisismmd_info` / `1177a8e0bbd3`: [thư mục](runs/crisismmd_info/1177a8e0bbd3/)
- `crisismmd_info` / `c4ce0465cf94`: [thư mục](runs/crisismmd_info/c4ce0465cf94/)
- `crisismmd_info` / `4c0bddde7d10`: [thư mục](runs/crisismmd_info/4c0bddde7d10/)
- `crisismmd_info` / `72d10d08e632`: [thư mục](runs/crisismmd_info/72d10d08e632/)
- `crisismmd_info` / `ddd9a6fc59b0`: [thư mục](runs/crisismmd_info/ddd9a6fc59b0/)

## Phân tích offline

Các báo cáo này đọc lại kết quả đã có; không phải lượt GPU mới.

- [analysis/crisismmd_direct_pair/8d71e3c908af_metaclip2_vith14.json](analysis/crisismmd_direct_pair/8d71e3c908af_metaclip2_vith14.json)
- [analysis/crisismmd_agreement/8d71e3c908af_metaclip2_vith14.json](analysis/crisismmd_agreement/8d71e3c908af_metaclip2_vith14.json)
- [analysis/fakeddit_text_exploratory.json](analysis/fakeddit_text_exploratory.json)
- [analysis/uncertainty/9ed97f55cf78_split_conformal_group5_v2_pooled_alpha100/summary.json](analysis/uncertainty/9ed97f55cf78_split_conformal_group5_v2_pooled_alpha100/summary.json)
- [analysis/uncertainty/9ed97f55cf78_split_conformal_group5_v2_classwise_alpha100/summary.json](analysis/uncertainty/9ed97f55cf78_split_conformal_group5_v2_classwise_alpha100/summary.json)
- [analysis/uncertainty/9ed97f55cf78_split_conformal_group5_v1_alpha100/summary.json](analysis/uncertainty/9ed97f55cf78_split_conformal_group5_v1_alpha100/summary.json)
- [analysis/uncertainty/03f1882357bb_fusion_weight_group_cv_v2/summary.json](analysis/uncertainty/03f1882357bb_fusion_weight_group_cv_v2/summary.json)
- [analysis/uncertainty/03f1882357bb_fusion_weight_group_cv_v1/summary.json](analysis/uncertainty/03f1882357bb_fusion_weight_group_cv_v1/summary.json)
- [analysis/fallback/03f1882357bb_6acc864b7851_metaclip2_vith14/summary.json](analysis/fallback/03f1882357bb_6acc864b7851_metaclip2_vith14/summary.json)
- [analysis/fallback/2ff98e4f0871_e595fbb7f79a_siglip2_so400m/summary.json](analysis/fallback/2ff98e4f0871_e595fbb7f79a_siglip2_so400m/summary.json)
- [analysis/fallback/2ff98e4f0871_e595fbb7f79a_dfn5b_vith14/summary.json](analysis/fallback/2ff98e4f0871_e595fbb7f79a_dfn5b_vith14/summary.json)
- [analysis/uncertainty/2ff98e4f0871_branch_disagreement_v1/summary.json](analysis/uncertainty/2ff98e4f0871_branch_disagreement_v1/summary.json)
- [analysis/uncertainty/03f1882357bb_branch_disagreement_v1/summary.json](analysis/uncertainty/03f1882357bb_branch_disagreement_v1/summary.json)
