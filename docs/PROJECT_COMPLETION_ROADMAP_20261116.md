# Lộ trình hoàn thành dự án — 16/11/2026

Đây là đề xuất từ source, cấu hình, báo cáo và kết quả đã lưu; chưa phải protocol thực nghiệm được phê duyệt. Không thay code, dữ liệu, cấu hình hoặc run hiện có. Không đọc dữ liệu thô, chạy encoder, mở test hay gửi job trong lượt phân tích này.

## 1. Quyết định đề xuất

Ưu tiên hoàn thiện Fakeddit min5: câu hỏi nghiên cứu về giá trị ảnh–tiêu đề và ngân sách nhãn, có xác nhận độc lập sau khi khóa phương án. Benchmark prompt là bằng chứng hỗ trợ; mở rộng đủ tám dataset trên MetaCLIP2 là tùy chọn theo thời gian và yêu cầu học phần.

Hai mức hoàn thành cần phân biệt:

- **Phạm vi cốt lõi:** zero-shot và few-shot Real/Fake, phân tích trade-off, protocol min5/test riêng, kết quả tái kiểm tra được và báo cáo nhất quán. Chỉ đủ nếu yêu cầu chính thức cho phép giới hạn ở tầng một.
- **Phạm vi hai tầng:** thêm phân loại năm loại Fake trên cùng cohort min5; so với flat sáu lớp cùng ngân sách nhãn, đo toàn pipeline trên test sau khi khóa toàn bộ. Cần chọn mức này nếu phân cấp là yêu cầu bắt buộc.

Không dùng kết quả hierarchy cohort cũ thay bằng chứng cascade min5. Không cần tăng điểm mọi nhánh để hoàn thành: kết quả âm có đối chứng vẫn trả lời được câu hỏi nghiên cứu.

## 2. Bằng chứng hiện có và cách diễn giải

| Bằng chứng | Ý nghĩa đối với hướng đi |
|---|---|
| Audit lưu ngày 16/11: 32/32 run passed | Có nền artifact tốt; audit này không chứng minh tính tổng quát hóa và chưa được chạy lại trong lượt lập kế hoạch |
| Zero-shot min5: StatA ảnh accuracy 72,02%; fusion hiện tại thấp hơn | Giữ StatA ảnh làm mốc không nhãn train; chưa có bằng chứng tiêu đề giúp top-1 trong fusion này |
| Few-shot ảnh–tiêu đề 64/320: accuracy 81,22%, macro-F1 80,16% | Văn bản bổ trợ ảnh trong linear probing; 384 nhãn train/seed phải được công bố |
| Few-shot ảnh–tiêu đề 192/192: accuracy 82,41%, macro-F1 82,04% | Ứng viên điểm tổng tốt hơn, nhưng recall Fake giảm 86,91% → 80,53% |
| False Connection recall binary giảm 71,22% → 56,08% khi đổi support | Cần chọn gate theo trade-off; accuracy tổng chưa đủ để quyết định |
| Waffle–semantic–StatA: DTD 71,10%, EuroSAT 61,73%; Waffle–semantic EuroSAT 64,42% | Chưa có phương pháp kết hợp thắng ổn định mọi miền; giữ kết quả âm |
| `fakeddit_fewshot_task1.py` cố định train/dev; `protocol.freeze` kiểm `kind=fakeddit` legacy | Test min5 cần triển khai protocol/runner riêng, không chỉ thay đường dẫn hoặc tái sử dụng lock v2 |

Các số Fakeddit trên là dev, cùng cohort min5 trong các so sánh nêu ở đây. Zero-shot và few-shot khác chế độ học. Mean ± sample std theo seed không phải khoảng tin cậy dữ liệu độc lập.

## 3. Thứ tự công việc và điều kiện dừng

### P0 — Chốt câu hỏi và phạm vi trước khi chạy tiếp

Đầu ra: một protocol ngắn xác định task chính, metric chính, baseline, ngân sách, seed, split và phạm vi một/hai tầng. Đề xuất macro-F1 làm metric chất lượng tổng chính, kèm recall Fake và recall Real bắt buộc; nếu có mục tiêu vận hành ưu tiên bỏ sót Fake thì quy định ràng buộc recall Fake trước khi chọn phương án.

Giới hạn một vòng phân tích trade-off có kế hoạch, sau đó quyết định. Không dùng số lần thử không giới hạn trên dev để tìm điểm đẹp. Dev đã được xem nhiều lần, nên kết quả tiếp theo trên dev vẫn là phát triển nội bộ.

### P1 — So sánh hai gate ở cùng mục tiêu recall Fake, ưu tiên CPU

Đầu ra: đường precision–recall, bảng ngưỡng và so sánh hai policy trên cùng ID, từng seed, gồm macro-F1, BA, recall Real/Fake, false positive/negative và recall binary theo subtype.

Đối chiếu mặc định trước; tiếp theo so ở cùng mục tiêu recall Fake được ghi trước, có thể dùng mức baseline legacy làm mốc phát triển. Chọn ngưỡng trên phần selection của dev, áp dụng nguyên trạng sang phần audit tách biệt. Việc đã xem full dev khiến audit này không phải test độc lập. Quy định trước cách xử lý đồng hạng và trường hợp không đạt mục tiêu; không chọn lại ngưỡng trên audit.

Khi so policy, cùng quy trình chọn ngưỡng và cùng phần audit. Nếu chọn ngưỡng riêng theo model/seed, đó phải là quy tắc định trước, được lưu và áp dụng lại khi test.

Không nạp cache/feature/weights theo mặc định trong lượt lập kế hoạch này. Khi triển khai phân tích, chỉ đọc các artifact cần thiết sau kiểm checksum và xác nhận đó là output dự đoán được phép sử dụng. Đường dẫn triển khai gợi ý là script/module phân tích riêng, giữ run gốc bất biến.

Điều kiện dừng: nếu balanced không còn lợi ích ở cùng recall Fake thì giữ legacy theo mục tiêu đó; không cần tìm support ratio mới. Nếu balanced vẫn tốt hơn thì giữ làm ứng viên. Không bắt buộc một policy thắng cho mọi mục tiêu.

### P2 — Hoàn thiện cascade trên dev nếu giữ phạm vi hai tầng

Phải làm trước lần xác nhận test cuối của toàn hệ thống.

- Tầng một: gate đã chọn bằng quy tắc dev cố định.
- Tầng hai: classifier năm loại Fake; baseline ban đầu nên dùng frozen embedding và linear head để giới hạn số biến thay đổi.
- Đối chứng: flat sáu lớp trên đúng cùng hợp support train; phân cấp và flat phải công bố tổng số ID train có nhãn duy nhất và quota sáu lớp. Nếu tầng hai cần thêm support, flat được cấp cùng pool và phải ghi ngân sách mới.
- Đo gate, oracle gate (chỉ chẩn đoán), phân loại subtype trên tập Fake thật cố định, và accuracy/macro-F1 sáu lớp end-to-end trên toàn cohort. Điểm subtype trên những mẫu được gate nhận đúng chỉ là metric có điều kiện, không thay end-to-end.
- Ghi riêng Fake bị gate chuyển thành Real và Real bị chuyển vào tầng Fake. Recall subtype end-to-end không thể vượt tỷ lệ gate nhận ra Fake của chính subtype đó.

Điều kiện giữ cascade: có lợi ích trên metric đã chọn và đáp ứng ràng buộc định tuyến, hoặc cung cấp kết quả âm rõ ràng trả lời câu hỏi phân cấp. Không thêm EM/StatA/bỏ phiếu chỉ vì đã có module; chỉ thêm khi có giả thuyết cụ thể và đối chứng đủ ngân sách.

### P3 — Khóa và xác nhận độc lập trên test min5

Đây là khoảng trống quan trọng nhất trước kết luận cuối. Chưa thực hiện bước này khi ứng viên/phạm vi còn thay đổi.

1. Bổ sung protocol/runner min5 riêng và kiểm tra bằng dữ liệu tổng hợp nhỏ: nhầm split/cohort, đảo mã nhãn, trùng train/dev/test, hash đổi, support ngoài train và không cho fit bằng test.
2. Khóa code, dependency, backbone, prompt/descriptor, công thức fusion, support ID, seed, classifier, ngưỡng, quy tắc grouping StatA và metric; xác định danh sách ứng viên/baseline sẽ được báo cáo trước khi xem test.
3. Kiểm provenance/cohort theo manifest và bộ kiểm đã có trong bước được cho phép; giữ hạn chế đọc dataset của dự án. Nếu cần truy cập ngoài quyền hiện có, xử lý quyền cụ thể trước, không tự chạy audit chạm dữ liệu bị cấm.
4. Với StatA, ghi rõ transductive, thông tin query không nhãn được dùng và cách chia nhóm; tách khỏi linear probing có nhãn train.
5. Sau khi được yêu cầu chạy và được duyệt hành động ngoài hệ thống nếu có, chạy bộ xác nhận đã khóa. Một đợt xác nhận có thể gồm nhiều baseline/seed định trước; không chọn lại winner/ngưỡng dựa vào test.
6. Nếu tiếp tục phát triển sau khi nhìn test, ghi rõ test đã trở thành dữ liệu phát triển đối với thay đổi đó; không gọi lần sửa sau là xác nhận độc lập trên cùng test.

Khoảng tin cậy/chênh lệch nên tính ghép cặp trên cùng mẫu và cùng seed. Nêu đơn vị resampling và phụ thuộc theo bài/ảnh/nhóm nếu có; không coi nhiều seed trên cùng ID là nhiều dataset độc lập. Với StatA, bootstrap trên prediction cố định không bao gồm toàn bộ bất định do đổi nhóm/query pool; công bố giới hạn này.

### P4 — Đóng gói báo cáo và tái lập

Đầu ra: bảng chính phân biệt dev/test, zero/few-shot, cohort và ngân sách; mỗi bảng trỏ được đến config/run/summary/audit. Ghi chính xác đóng góp là pipeline, kiểm chứng và phát hiện thực nghiệm; không tự gọi linear probing hoặc cân bằng support là thuật toán mới.

Đồng bộ tài liệu cũ: `BENCHMARK_PROGRESS.md` còn câu chưa thấy run Fakeddit bản cuối, trong khi đã có Task 1 dev min5; cần mô tả rõ pipeline legacy so với min5 để tránh hiểu nhầm. Đối chiếu `main.tex` với phạm vi đã chốt và bằng chứng mới.

Hoàn thiện hướng dẫn tái hiện, kiểm dependency, test thích hợp, audit artifact, build PDF và kiểm bảng/hình/tham chiếu. Nếu cần demo bảo vệ, thêm giao diện nhỏ sau khi khóa pipeline, công bố ngưỡng và trạng thái dự đoán; demo không tự chứng minh khả năng xác minh sự thật ngoài bộ dữ liệu.

## 4. Nhánh nghiên cứu tùy chọn

Chỉ chọn một sau các ưu tiên chính:

- **Waffle–semantic trên một benchmark bổ sung:** giữ weight 0,5 và cùng seed/quy tắc, chọn miền và giao thức trước khi chạy. Có giá trị hơn mở đồng loạt phần còn thiếu của tám benchmark nếu câu hỏi là khả năng chuyển miền.
- **Fusion zero-shot tối giản:** đối chứng `0,75 StatA + 0,25 title` đã được gợi ý trong review, chỉ là giả thuyết phát triển để tách tác động CLIP khỏi title; không bảo đảm tăng điểm và không quét nhiều trọng số.
- **Chẩn đoán StatA thất bại EuroSAT:** tách từng yếu tố prior, grouping hoặc regularization bằng ablation có kiểm soát; chỉ làm nếu cơ chế adaptation là trọng tâm báo cáo.

Tạm hoãn: backbone mới, fine-tune lớn, trở lại CrisisMMD/MM-IMDb, ma trận prompt/weight rộng, ensemble thêm nhiều solver, mở đủ tám benchmark chỉ để đủ số lượng. Các hướng này tăng chi phí và số giả thuyết nhưng chưa lấp khoảng trống xác nhận độc lập.

## 5. Tiêu chí nghiệm thu

- Phạm vi phù hợp yêu cầu học phần; phần chưa làm được ghi rõ.
- Có protocol min5 riêng và kết quả test độc lập cho ứng viên đã khóa; hoặc công khai chưa có nếu chưa thực hiện, không đổi tên dev thành test.
- Mọi so sánh chính cùng ID/cohort, cùng chế độ hoặc nêu rõ chênh lệch chế độ/ngân sách.
- Nếu tuyên bố hai tầng: có flat đối chứng và end-to-end trên cùng cohort/ngân sách; oracle chỉ chẩn đoán.
- Kết quả có artifact/hash/metric tái kiểm tra; thay đổi code được kiểm tra phù hợp; báo cáo build thành công.
- Kết quả âm, đánh đổi recall và giới hạn xác minh sự thật được giữ trong kết luận.

## Nguồn đã đối chiếu

`README.md`, `main.tex`, `docs/CODEX-NAVIGATION-GUIDE.md`, `results/README.md`, `docs/results_audit_20261116.json`; các review Task 1 min5, few-shot Task 1, support min5, error analysis, hierarchy dev, Waffle–semantic–StatA EuroSAT và benchmark progress. Source đọc trực tiếp: `fakeddit_fewshot_task1.py`, `fakeddit_fewshot_support.py`, `protocol.py`.

Graphify AST chỉ dùng định hướng. Có hai file source mới hơn graph (`packaging.py`, `single_stage.py`); không dựa vào graph để kết luận nội dung mới của chúng, không rebuild cho câu hỏi này.

## Tự rà soát theo agent-self-evaluation

| Tiêu chí | Điểm / 5 | Bằng chứng và giới hạn |
|---|---:|---|
| Chính xác | 4 | Số liệu khớp review và audit đã lưu; chưa chạy lại audit hoặc kiểm trực tiếp mọi prediction |
| Đầy đủ | 4 | Có hai phạm vi, thứ tự và nghiệm thu; chưa biết deadline/rubric chính thức để quyết định tầng hai bắt buộc |
| Rõ ràng | 4 | Phân biệt bằng chứng và đề xuất; tài liệu chi tiết cần đi kèm phần tóm tắt |
| Khả thi | 4 | Ưu tiên CPU trước GPU và chỉ ra lock min5 còn thiếu; chưa ước lượng tài nguyên cho cascade chưa triển khai |
| Súc tích | 4 | Có thứ tự và điều kiện dừng; giữ phần tùy chọn riêng để hạn chế phân tán |

Trung bình 4,0/5. Cải thiện theo tác động: đối chiếu rubric/deadline khi có; triển khai bảng trade-off P1; ước lượng tài nguyên từ thiết kế cascade cụ thể. Người dùng có thể kiểm lại đánh giá qua các nguồn trên; đây không phải chứng nhận chất lượng hệ thống.
