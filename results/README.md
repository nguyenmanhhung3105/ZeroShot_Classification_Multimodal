# Danh mục kết quả phục vụ báo cáo

Kiểm tra bàn giao ngày 21/11/2026: **36/36 run passed**, 391 artifact hợp lệ, 698 tổ hợp method–seed, 4,953,696 dòng prediction. Chi tiết và summary máy đọc được: [audit](../docs/results_audit_20261121.json). Script tái kiểm tra: `venv/bin/python -B scripts/audit_report_results.py`.

GitHub chỉ chứa bản sao nguyên byte của `summary.tsv` tại `published/`; [manifest](published/manifest.json) ghi checksum và nguồn từng bảng. Dự đoán, feature, lock vận hành và run đầy đủ vẫn ở `final/` ngoài Git; cần lấy riêng để chạy lại audit. Các liên kết archive bên dưới chỉ hoạt động trên máy có archive lịch sử.

Mỗi run gốc giữ nguyên byte. Full/pilot/smoke/diagnostic được phân biệt bằng metadata, không suy từ tên thư mục. `full` nghĩa là đủ cohort của thí nghiệm; có thể là dev hoặc split đã dùng phát triển, không đồng nghĩa test độc lập.

| Run | Loại | Mode | Mẫu | Method/seed |
|---|---|---|---:|---:|
| [dev_4700bc4c6d15](published/fewshot_task1_analysis/train_cv/dev_4700bc4c6d15/summary.tsv) | fewshot_task1_train_cv | full | 3,000 | 20 |
| [benchmark_dtd_57ebfa1029a0](published/kaggle/benchmark_dtd_57ebfa1029a0/run/summary.tsv) | benchmark | full | 1,880 | 16 |
| [benchmark_imagenetv2_83c64730123e](published/kaggle/benchmark_imagenetv2_83c64730123e/run/summary.tsv) | benchmark | full | 10,000 | 16 |
| [cifar100_em_stata_dev_c3d25f6a4d9c](published/kaggle/cifar100_em_stata_dev_c3d25f6a4d9c/run/summary.tsv) | cifar100_em_stata | full | 10,000 | 20 |
| [cifar100_rerank_dev_a6781f055ae7](published/kaggle/cifar100_rerank_dev_a6781f055ae7/run/summary.tsv) | cifar100_rerank | full | 10,000 | 4 |
| [cifar100_two_stage_dev_4cef907deba3](published/kaggle/cifar100_two_stage_dev_4cef907deba3/run/summary.tsv) | cifar100_two_stage | full | 10,000 | 3 |
| [dclip_stata_dtd_993bc656c03d](published/kaggle/dclip_stata_dtd_993bc656c03d/run/summary.tsv) | dclip_stata | full | 1,880 | 9 |
| [diagnostic_dtd_fusion_5ca4c942af32](published/kaggle/diagnostic_dtd_fusion_5ca4c942af32/run/summary.tsv) | fusion_diagnostic | diagnostic | 256 | 6 |
| [diagnostic_eurosat_fusion_9149dc20f5b4](published/kaggle/diagnostic_eurosat_fusion_9149dc20f5b4/run/summary.tsv) | fusion_diagnostic | diagnostic | 256 | 6 |
| [diagnostic_fakeddit_2way_fusion_f052fbe78550](published/kaggle/diagnostic_fakeddit_2way_fusion_f052fbe78550/run/summary.tsv) | fusion_diagnostic | diagnostic | 256 | 6 |
| [diagnostic_imagenetv2_fusion_d0872294ac03](published/kaggle/diagnostic_imagenetv2_fusion_d0872294ac03/run/summary.tsv) | fusion_diagnostic | diagnostic | 256 | 6 |
| [diagnostic_pets_fusion_9239117fb479](published/kaggle/diagnostic_pets_fusion_9239117fb479/run/summary.tsv) | fusion_diagnostic | diagnostic | 256 | 6 |
| [fakeddit_fewshot_support_min5_f3fa1b84f8d6](published/kaggle/fakeddit_fewshot_support_min5_f3fa1b84f8d6/run/summary.tsv) | fakeddit_fewshot_support | full | 3,000 | 30 |
| [fakeddit_fewshot_task1_min5_3d8cadcbf035](published/kaggle/fakeddit_fewshot_task1_min5_3d8cadcbf035/run/summary.tsv) | fakeddit_fewshot_task1 | smoke | 5 | 15 |
| [fakeddit_fewshot_task1_min5_4051d18f8c31](published/kaggle/fakeddit_fewshot_task1_min5_4051d18f8c31/run/summary.tsv) | fakeddit_fewshot_task1 | full | 3,000 | 15 |
| [fakeddit_hierarchy_dev_85d0ba88b624](published/kaggle/fakeddit_hierarchy_dev_85d0ba88b624/run/summary.tsv) | fakeddit_hierarchy | full | 3,000 | 25 |
| [fakeddit_stata_calip_dev_85c4c3d20e7e](published/kaggle/fakeddit_stata_calip_dev_85c4c3d20e7e/run/summary.tsv) | fakeddit_stata_calip | full | 3,000 | 24 |
| [fakeddit_task1_dev_192c58fde5ad](published/kaggle/fakeddit_task1_dev_192c58fde5ad/run/summary.tsv) | fakeddit_task1 | full | 3,000 | 12 |
| [fakeddit_task1_dev_min5_b05002bc6b94](published/kaggle/fakeddit_task1_dev_min5_b05002bc6b94/run/summary.tsv) | fakeddit_task1 | full | 3,000 | 12 |
| [paper_reference_cub_vitl14_69338b83481b](published/kaggle/paper_reference_cub_vitl14_69338b83481b/run/summary.tsv) | paper_reference | full | 5,794 | 11 |
| [paper_reference_dtd_vitl14_4de91a58e03b](published/kaggle/paper_reference_dtd_vitl14_4de91a58e03b/run/summary.tsv) | paper_reference | full | 1,880 | 11 |
| [paper_reference_eurosat_vitl14_748bccf4b96c](published/kaggle/paper_reference_eurosat_vitl14_748bccf4b96c/run/summary.tsv) | paper_reference | full | 27,000 | 11 |
| [paper_reference_food101_vitl14_6f9a5c56259b](published/kaggle/paper_reference_food101_vitl14_6f9a5c56259b/run/summary.tsv) | paper_reference | full | 25,250 | 11 |
| [paper_reference_imagenet_vitl14_ac110d7d823a](published/kaggle/paper_reference_imagenet_vitl14_ac110d7d823a/run/summary.tsv) | paper_reference | full | 50,000 | 11 |
| [paper_reference_imagenetv2_vitl14_0e9e6407ce90](published/kaggle/paper_reference_imagenetv2_vitl14_0e9e6407ce90/run/summary.tsv) | paper_reference | full | 10,000 | 11 |
| [paper_reference_pets_vitl14_a706e908983e](published/kaggle/paper_reference_pets_vitl14_a706e908983e/run/summary.tsv) | paper_reference | full | 3,669 | 11 |
| [paper_reference_places365_vitl14_1a913aff262c](published/kaggle/paper_reference_places365_vitl14_1a913aff262c/run/summary.tsv) | paper_reference | full | 36,500 | 11 |
| [waffle_semantic_stata_dtd_08236ee40262](published/kaggle/waffle_semantic_stata_dtd_08236ee40262/run/summary.tsv) | waffle_semantic_stata | full | 1,880 | 71 |
| [waffle_semantic_stata_eurosat_664684cd7c96](published/kaggle/waffle_semantic_stata_eurosat_664684cd7c96/run/summary.tsv) | waffle_semantic_stata | full | 27,000 | 71 |
| [waffle_stata_dtd_218418a64888](published/kaggle/waffle_stata_dtd_218418a64888/run/summary.tsv) | waffle_stata | pilot | 256 | 39 |
| [waffle_stata_dtd_full_c86ed2a97772](published/kaggle/waffle_stata_dtd_full_c86ed2a97772/run/summary.tsv) | waffle_stata | full | 1,880 | 39 |
| [waffle_stata_eurosat_a1a6f8433ee9](published/kaggle/waffle_stata_eurosat_a1a6f8433ee9/run/summary.tsv) | waffle_stata | pilot | 256 | 39 |

| [fakeddit_fewshot_cascade_min5_97036eece619](published/kaggle/fakeddit_fewshot_cascade_min5_97036eece619/run/summary.tsv) | fakeddit_fewshot_cascade | smoke | 5 | 30 |
| [fakeddit_fewshot_cascade_min5_b7c54cfcca4a](published/kaggle/fakeddit_fewshot_cascade_min5_b7c54cfcca4a/run/summary.tsv) | fakeddit_fewshot_cascade | full | 3,000 | 30 |

## Kết quả lịch sử và staging

- [Kết quả trước bản hiện tại](../archive/report_cleanup_20261116/results/): baseline nhiều backbone/dataset, giữa kỳ, exploratory và replay cũ. Giữ để truy vết phần tổng kết lịch sử trong báo cáo.
- [results1 đời đầu](../archive/report_cleanup_20261116/results1/).
- [Manifest archive](../archive/report_cleanup_20261116/manifest.json): checksum trước/sau của các file đã chuyển. Các metric lịch sử chưa được tính lại trong audit hiện tại.
- `_system/` giữ receipt/staging/sample locks, có cả cache và bundle cũ; không phải nguồn chọn metric để đưa vào báo cáo. Không dọn cache hoặc thay trạng thái job trong lần này.

## Phạm vi kiểm tra

Audit kiểm tra checksum, ID/cùng cohort/nhãn giữa các phương pháp, accuracy/Macro-F1/BA từ prediction, mean/sample std giữa seed và summary. Top-5 chỉ kiểm tra phép tổng hợp từ bảng seed, không tái suy luận logits. Không đọc dataset hoặc nạp backbone. Smoke không dùng báo cáo chất lượng; pilot/diagnostic không thay kết quả full.

Một số run Fakeddit cùng tên phương pháp thuộc cohort v2 cũ, một số thuộc min5. Kiểm tra `cohort`, `dev_cohort` và hash manifest trong run; không cộng/gộp tùy ý. Nhánh hierarchy cũ không ghép với few-shot min5 để gọi là cascade hoàn chỉnh.
