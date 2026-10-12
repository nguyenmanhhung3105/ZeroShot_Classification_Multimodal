# Thử quan hệ ảnh–text theo nhãn — 12-10-2026

## Kết quả chính

Đã thêm cosine ảnh–text cùng mẫu, JS/entropy single-label và hiệu chỉnh nhẹ theo policy lớp. Đối chứng checkpoint thật trên **năm dòng đầu Fakeddit**: công thức cuối không đổi nhãn dự đoán so với baseline ở cả MetaCLIP2, DFN5B và SigLIP2. **Chưa có bằng chứng cải thiện**, và năm mẫu không đủ đánh giá hiệu quả hoặc chọn tham số.

| Model | Số mẫu | Accuracy baseline → mới | Macro-F1 baseline → mới | Balanced accuracy baseline → mới | Nhãn thay đổi |
|---|---:|---|---|---|---:|
| MetaCLIP2 | 5 | 0.20 → 0.20 | 0.066667 → 0.066667 | 0.125 → 0.125 | 0 |
| DFN5B | 5 | 0.40 → 0.40 | 0.111111 → 0.111111 | 0.250 → 0.250 | 0 |
| SigLIP2 | 5 | 0.40 → 0.40 | 0.194444 → 0.194444 | 0.375 → 0.375 | 0 |

Heuristic cũ chỉ cộng evidence cosine vào False Connection với weight 0.1 cũng giữ nguyên các nhãn trên nhóm này. Macro-F1 tính trên đủ 6 lớp cố định, gồm lớp không có support; balanced accuracy xét các lớp có nhãn thật. Nhóm này không có Satire/Imposter trong nhãn thật nên scikit-learn cảnh báo khi dự đoán các lớp ấy.

## Phương pháp và tham số đã khóa

Điểm nền B dùng prompt current/mean_sim, ảnh 0.75, text 0.25, text scale 25; scale ảnh từ từng checkpoint. Công thức mới:

```text
r = cosine(ảnh, text cùng mẫu)
J = JS(P_img, P_text) / ln(2)
R = 1 - max(H(P_img), H(P_text)) / ln(6)
A = ((1+r)/2) × (1-J) × R
D = ((1-r)/2) × J × R
e_0=A; e_2=D; e_1=e_3=e_4=e_5=0
P'_c = B_c (1+0.25 e_c) / Σ_k B_k (1+0.25 e_k)
```

JS ở đây là trung bình hai KL; J và R được clip vào [0,1]. R rất gần 0 được đặt 0 để bỏ sai số số học với phân phối phẳng. Nhãn thật không xuất hiện trong công thức. Cosine và JS/entropy không là bộ kiểm chứng sự thật; mức tự tin cao vẫn có thể sai. Chi tiết/điều kiện fallback ở [trang công thức](so_do_he_thong/13_quan_he_anh_text.md).

Cách dùng match/mismatch theo taxonomy là giả thuyết hợp lý để thử; Fakeddit có nhãn theo subreddit và trường hợp cùng đối tượng nhưng sai ý nghĩa/ngữ cảnh. Không dùng heuristic quan hệ để gán mọi cặp cosine thấp vào fake. [Paper Fakeddit](https://arxiv.org/html/1911.03854)

## Quan sát dữ liệu trong giới hạn cho phép

Đã xem schema và tối đa **năm bản ghi đầu**, giới hạn tại CSV iterator, của ba processed TSV theo cấu hình. Không đọc dataraw/data/raw, không chạy preprocessing, không tìm thêm dòng để đủ lớp. Hai ảnh Fakeddit đã xem trực tiếp là cv9kwi và 7yas5i.

| ID Fakeddit | Nhãn thật | Điều cần lưu ý |
|---|---|---|
| cv9kwi | 2=False Connection | Ảnh bản đồ marathon, title nói về bố đi học: khác nội dung |
| 46j162 | 0=True | Title “snake head … green beans”; cosine không tự xác minh tính xác thực |
| bjkefw | 5=Misleading | Title về cartoon chống vaccine; dạng cartoon không tự đồng nghĩa Satire |
| c6edq0q | 4=Manipulated | Title rất chung, cosine khó phản ánh thao tác chỉnh ảnh |
| 7yas5i | 2=False Connection | Có cùng đối tượng bấm kim trong ảnh/title; nhân hóa “excited” cần hiểu ngữ cảnh |

Chỉ hai ảnh nêu trên được xem trong lượt này; không khẳng định chi tiết ảnh của ba ID còn lại từ việc đọc title/nhãn.

Processed CrisisMMD có nhãn informative/humanitarian chung; không có hai nhãn riêng image/text hoặc annotation khớp cặp. Năm dòng đầu còn có tweet_id định dạng số khoa học kiểu `9,17793E+17` và một tweet đi kèm ba ảnh. Đây là quan sát giới hạn; không suy ra tỷ lệ lỗi toàn tập và không sửa dữ liệu. Vì vậy không ép cosine thấp thành not_informative/not_humanitarian. Paper gốc chú thích hai modality riêng. [Paper CrisisMMD](https://arxiv.org/html/1805.00713)

MM-IMDb có plot dài và genre nhiều nhãn, không có nhãn mismatch. Quan hệ ở task này chỉ diagnostic; dùng mean embedding chunk cho ảnh–plot ngay cả khi quyết định genre dùng max_sim.

## Các lượt thực chạy và artifact

Các lượt công thức cuối có entropy:

| Model | Thư mục |
|---|---|
| MetaCLIP2 | [smoke5_31gx_uv5](../results/relation_ablation/smoke5_31gx_uv5/) |
| DFN5B | [smoke5_0s35r8rk](../results/relation_ablation/smoke5_0s35r8rk/) |
| SigLIP2 | [smoke5_wte6_1ml](../results/relation_ablation/smoke5_wte6_1ml/) |

Mỗi thư mục có frozen_plan.json, source hash, bản sao input giới hạn, baseline_log với TSV/JSON/audit, summary.tsv và comparison.tsv. `final_formula_recheck.json` ghi hash module cuối và kiểm tra tính lại metric từ cùng log với công thức cuối; kết quả khớp trong dung sai 1e-12. Không nạp lại checkpoint để kiểm tra lại một thay đổi xử lý sai số số học.

`sample_set_id` khớp giữa các method trong cùng lượt, nhưng khác giữa model vì path bản sao input/scope lượt khác nhau. Các lượt dùng cùng năm dòng nguồn theo thứ tự và hash input tương ứng; không diễn giải ID khác nhau là đã chọn năm mẫu khác nhau.

Lượt thử trước khi thêm entropy ở `smoke5_c14gkjt3`/`smoke5_099f9f4r` chỉ là thử trung gian, không dùng trong bảng kết quả cuối. Ở DFN5B bản trung gian thay hai nhãn và tăng macro-F1 nhưng không tăng accuracy; không chọn nó làm kết luận tốt hơn từ năm mẫu. Hệ số entropy được thêm vì hai phân phối cùng phẳng không phải đồng thuận có thông tin, và được kiểm bằng dữ liệu giả độc lập.

SigLIP2 thử offline đầu tiên tại `smoke5_ibm0w2kt` dừng vì thiếu file tokenizer trong cache. Sau khi cho phép truy cập Hugging Face/cache ngoài workspace, tokenizer nạp được và lượt thật hoàn tất. Cả ba lượt thành công dùng CPU; không train/fine-tune hoặc mở rộng dataset.

Thông số runtime quan sát: MetaCLIP2 D=1024, ảnh 224×224, L=77, scale≈100; DFN5B D=1024, ảnh 224×224, L=77, scale≈14.28; SigLIP2 ảnh 224×224, L=64, scale≈109.88, bias≈−15.9324. Wrapper tra visual.output_dim của SigLIP2 là None, không tự gán D từ thuộc tính đó.

## Kiểm thử và trạng thái sử dụng

102 test tự động bằng dữ liệu/model giả kiểm công thức, policy/thứ tự lớp, không dùng nhãn thật để routing, fallback text rỗng, λ=0, một modality, phân phối phẳng, diagnostic giữ nguyên đầu ra, chunk mean/max_sim không encode lại, logger khi audit tắt và giới hạn năm dòng trước lọc.

Lúc chạy các lượt trên đã chủ động bật quan hệ để đối chứng. Sau yêu cầu bổ sung flag, **các YAML mặc định tắt**: pipeline chính dùng inference.image_text_relation.enabled, hai runner nghiên cứu có flag riêng. Mọi backbone/task hỗ trợ tính quan hệ khi bật; mode auto chọn class_aware cho Fakeddit 6way và diagnostic cho task khác. Upstream nay cũng hỗ trợ flag; original_image chỉ diagnostic, multimodal theo taxonomy. Các kết quả phía trên là kết quả ở phiên bản đã thử, không phải một lượt mới sau khi mở rộng upstream.

Chưa đánh giá lớn, chưa xác minh nguồn split/test độc lập, chưa có đánh giá thật quan hệ trên CrisisMMD/MM-IMDb. Không dùng kết quả năm mẫu để khẳng định model nào tốt hơn, tối ưu λ hoặc khẳng định phương pháp mới cải thiện tổng quát.
