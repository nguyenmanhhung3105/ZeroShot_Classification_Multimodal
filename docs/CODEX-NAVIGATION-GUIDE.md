# Bản đồ code phục vụ báo cáo

Điểm bắt đầu: `README.md` → `report/main.tex` → `results/README.md`. Đối chiếu metric với `docs/results_audit_20261116.json`; không lấy số từ tên thư mục hoặc bảng pilot.

| Luồng | Điểm vào và phụ thuộc |
|---|---|
| CLI | `run.py` → `src/vlm_project/cli.py` |
| Kaggle | `scripts/launch_runs.py` → `kaggle.py` → packaging / runner theo job |
| Đọc cohort | `io.py` định vị và hash → `data.py` kiểm tra schema/nguồn/cohort |
| Prompt benchmark | `benchmark.py`, `paper_reference.py` → `paper.py`, `backbone.py`, `evaluate.py` |
| Waffle/semantic/StatA | `waffle_semantic_stata.py`, `waffle_stata_full.py`, `dclip_stata.py` → `waffle_stata.py` → `diagnostic.infer_groups` → `adaptation.stata` |
| Fakeddit zero-shot | `fakeddit_task1.py` → profile trong `fakeddit.py`, fusion trong `methods.py`, StatA trong `diagnostic.py` |
| Few-shot hiện tại | `fakeddit_fewshot_support.py` → `fakeddit_fewshot_task1.py` → LogisticRegression; `fewshot_task1_analysis.py` dùng artifact đã lưu |
| Phân tích ngưỡng / protocol / test min5 | `scripts/analyze_fakeddit_support_thresholds.py` → `fewshot_support_thresholds.py`; `scripts/prepare_min5_protocol.py` → `min5_protocol.py`; `scripts/run_min5_test.py` → `min5_test_runner.py` → guard/data/predict/publish; xem `docs/MIN5_TASK1_PROTOCOL.md` (standalone, chưa chạy test thật) |
| Few-shot cascade min5 dev | `scripts/prepare_fakeddit_cascade_min5.py` xuất snapshot support/gate từ artifact nguồn; `scripts/run_fakeddit_cascade_min5.py` hoặc launcher → `fakeddit_fewshot_cascade.py` → shared encoder/probe → flat/hard route/diagnostic; xem `docs/FAKEDDIT_FEWSHOT_CASCADE_MIN5.md` |
| Protocol/test hai tầng min5 | `scripts/prepare_cascade_protocol.py` → `cascade_protocol.py`; `scripts/run_cascade_test.py` → `cascade_test_runner.py` → guard → shared `cascade.infer/evaluate`; xem `docs/CASCADE_TEST_PROTOCOL.md` (đã sealed, chưa mở test) |
| Phân cấp/đối chứng | `fakeddit_hierarchy.py`, `fakeddit_calip_ablation.py`, `cifar100_two_stage.py`, `cifar100_rerank.py`, `cifar100_em_stata.py` |
| Diagnostic | `diagnostic.py`, `calip.py`, `single_stage.candidates`; replay CLI đời cũ ở archive |
| Dữ liệu benchmark nguồn | `scripts/download_shared_benchmarks.py`, `download_two_stage_datasets.py`, `shared_data.py` |
| Fakeddit preprocessing | `scripts/prepare_fakeddit_splits.py` → `src/preprocessing/data_fakeddit/prepare_splits.py`, `helpers.py`; xem README cùng thư mục |
| Artifact | `evaluate.publish` → seed/prediction/summary → `run.json` → `completed.json`; `protocol.completed_run` kiểm tra hash |

Các module `diagnostic`, `waffle_stata`, `waffle_stata_full`, `dclip_stata`, `shared_data` còn trong src vì các runner của báo cáo gọi tới chúng, kể cả khi tên gợi một giai đoạn thử nghiệm trước. Không xóa chỉ theo tên file.

`fakeddit.py` và `protocol.py` còn giữ pipeline/lock v2 cũ; không đồng nhất với protocol dev min5. Source hash đổi sau khi dọn code không làm artifact lịch sử mất hợp lệ, nhưng lock cũ không dùng để khóa test cho code mới.

Graphify dùng AST local trên **src**, không quét toàn repo/dataset. Graph tạo khi dọn được đặt ở `graphify-out/`. Kiểm tra độ mới trước truy vấn; đọc source thật trước khi sửa. Không tạo semantic graph trên data, cache, checkpoint hoặc thư mục archive lớn.

Khi review thay đổi, xem snapshot/manifest và so với `backups/before_report_cleanup_20261116/source_before.tar.gz`; Git đã có nhiều thay đổi/xóa từ trước lần dọn này. Không khôi phục các file bị xóa sẵn, không commit/push khi chưa được yêu cầu.

Bản GitHub không kèm graph, backup hoặc archive local. Hướng dẫn thành viên và phạm vi bản bàn giao ở `docs/GITHUB_HANDOFF.md`; summary chia sẻ nằm trong `results/published/`.
