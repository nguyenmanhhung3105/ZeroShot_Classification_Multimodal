# Giao thức của bản cuối

## Backbone và phạm vi so sánh

Chỉ `metaclip2_vith14`, kiến trúc OpenCLIP `ViT-H-14-worldwide-quickgelu`, pretrained `metaclip2_worldwide`. Dùng preprocessing đi kèm checkpoint, ảnh RGB và unit image/text embeddings. Mean prototype là trung bình embedding từng câu đã chuẩn hóa; chỉ đối chứng unit và voter normalized CLIP chuẩn hóa lại mean. Prompt nhãn quá context bị từ chối; tiêu đề được tokenizer cắt theo context như bản cũ. Batch mặc định 8, không tự chuyển checkpoint khi thiếu VRAM.

Tokenization thực tế là **XLM-V (`facebook/xlm-v-base`), context 77**, theo [cấu hình chính thức OpenCLIP v3.3.0](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip/model_configs/ViT-H-14-worldwide-quickgelu.json). Kiểm tra độ dài prompt bằng tokenizer này trước khi encode; không dùng độ dài BPE OpenAI để suy ra giới hạn XLM-V. Phiên bản thư viện và requirements cũng được ghi trong protocol/lock; giữ cùng môi trường dev/test.

[CLIP](https://arxiv.org/abs/2103.00020), [DCLIP](https://arxiv.org/abs/2210.07183), [WaffleCLIP](https://arxiv.org/abs/2306.07282) được so theo prompt và aggregation trên cùng MetaCLIP2. Ba repo nguồn và commit nằm trong `configs/research_sources.json`; OpenAI_CLIP chỉ cung cấp tokenizer cho bộ sinh prompt tác giả, không nạp trọng số OpenAI. Số công bố dùng backbone, split hoặc tập lớp khác chỉ là tham khảo. Đối chứng chuẩn hóa mean Waffle là ablation, không tuyên bố đóng góp mới chỉ từ thao tác này.

## Fakeddit

Hai ứng viên được giữ từ lock lịch sử `3df5beb0854fce862d80fd70b1011b155e2ba7c28c50d9262975e12a0e8b8ba6`, dev cohort `124b9844b7b88986a3eb`. Prompt được chép nguyên bank cần thiết sang `configs/fakeddit_prompts.json`; nguồn/hash di chuyển ghi trong `configs/migration_sources.json`. CALIP và Waffle lọc không thuộc hai phương pháp thắng dev nên không còn trong đường chạy chính.

2way: normalized CLIP, EM-Dirichlet và StatA bỏ phiếu binary theo nhóm 75 ảnh, grouping seeds [0,1,2]. Voter CLIP dùng fusion 75% xác suất ảnh và 25% xác suất tiêu đề semantic; EM/StatA chỉ dùng ảnh. Logit scale ảnh 100, tiêu đề 25. Majority >=2 phiếu.

6way: dùng cổng Majority **image-only** như cấu hình đã chọn. Cổng real (2way=1) đưa về True (6way=0). Cổng fake chọn argmax trong năm lớp giả, dựa trên fusion 75% xác suất ảnh legacy và 25% xác suất tiêu đề semantic. Không thay cổng 6way bằng cổng multimodal của 2way.

EM dùng softmax(30×mean cosine), 20 vòng ngoài, tối đa 1.000 vòng MM; ghép cụm với lớp bằng Hungarian từ xác suất CLIP. Tham khảo [Transductive Zero-Shot and Few-Shot CLIP](https://arxiv.org/abs/2405.18437), repo `SegoleneMartin/transductive-CLIP`, commit đối chiếu lịch sử `9066ad371757c794e4b8255e852f837ac9a1d3b1`. StatA giữ kNN=3, max_iter=10, inner=5 và các floor ổn định số trong solver cũ. Tham khảo [Realistic Test-Time Adaptation of Vision-Language Models](https://arxiv.org/abs/2501.03729), repo `MaxZanella/StatA`, commit đối chiếu lịch sử `2635f9355f9389756064becb5d9ed54c2756809c`. Nhóm <=3 ảnh giữ normalized CLIP cho StatA. Solver đã có đối chiếu số trong thí nghiệm lịch sử; bản cuối kiểm tra giữ công thức và dự đoán trên fixture tổng hợp.

Đây là thiết lập transductive: nhãn một ảnh phụ thuộc query khác cùng nhóm. Nhãn thật không truyền vào predictor hay solver. Đối chứng ảnh/văn bản/fusion là inductive; báo rõ khác biệt này. Ba grouping seed đo độ nhạy chia nhóm, không tự chứng minh ý nghĩa thống kê.

DCLIP/Waffle ở Fakeddit là đối chứng image-only thích nghi theo taxonomy, descriptor do dự án viết. Chỉ kết luận lợi ích multimodal khi xem cả đối chứng title-only, fusion và semantic title fusion; không quy toàn bộ chênh lệch với image-only cho một thành phần riêng.

## Dữ liệu và khóa test

Benchmark dùng đủ split trong README. Ảnh thiếu, sai checksum, ID trùng hoặc nhãn ngoài taxonomy làm dừng. Không bỏ ảnh lỗi để chạy tiếp và làm sai mẫu số.

Fakeddit giữ 3.000 mẫu/cohort đã chuẩn bị, train→train, dev→validate, test→test_public. Bản source cuối không có downloader/preprocessor raw; các chương trình đó được lưu nguyên trạng trong archive. Bộ dữ liệu đã audit từ trước vẫn ở local/Kaggle. File TSV, ảnh và thứ tự mẫu phải khớp manifest. Khi test, kiểm tra không trùng dev theo ID, SHA256 ảnh, SHA256 tiêu đề. Audit gần trùng pixels/dHash và loại trùng train/dev/test thuộc bước chuẩn bị lịch sử; kiểm tra runtime không thay thế audit đó.

Dù giữ công thức, code tổ chức lại được xem là phiên bản mới. Chạy đủ dev rồi `freeze`; không dùng trực tiếp lock cũ. Freeze giữ hai phương pháp cố định, không tự chọn phương án mới sau mỗi lần chạy. Lock bao gồm hash code, prompt, nguồn descriptor, tham số suy luận và bằng chứng dev hoàn tất. Test yêu cầu cùng hash, không chạy smoke và không được điều chỉnh prompt từ kết quả test.

## Đầu ra và báo cáo

Accuracy, Macro-F1 (toàn bộ lớp), balanced accuracy và precision/recall/F1/support từng lớp; benchmark thêm top-5. Đơn vị metric TSV/JSON là [0,1]. Waffle: 7 seed; Majority: 3 grouping seed; phương pháp xác định: 1 lượt. Báo mean ± sample std, ddof=1. Quy ước std khác một số analyzer lịch sử dùng ddof=0, nên không chép std cũ sang bảng mới.

Giữ riêng smoke và full. Không điền bảng so sánh bằng số giả hoặc coi fixture kiểm thử là số ảnh thật. `completed.json` chỉ được ghi sau khi có đủ bảng metric và checksum. Các lượt đang dở vẫn có `run.json` trạng thái running và không được freeze/report như lượt hoàn tất.

Smoke benchmark MetaCLIP2 chỉ chạy CLIP với 5 ảnh và toàn bộ lớp; full giữ nguyên 16 lượt metric (CLIP, DCLIP, Waffle 7 seed và ablation 7 seed). Smoke của nhánh tham chiếu OpenAI và luồng Fakeddit giữ cách kiểm tra riêng. Notebook MetaCLIP2 tái sử dụng một backbone/cache prototype trong RAM qua smoke→full, giải phóng sau full hoặc khi smoke lỗi. Cache gắn với model đang nạp và hash bank prompt có giữ thứ tự lớp, không dùng qua checkpoint/tiến trình khác.

Batch prompt MetaCLIP2 mặc định 64, ghép xuyên lớp rồi chia lại theo ranh giới lớp để tính trung bình embedding unit. CUDA OOM giảm về 32/16 và thử lại đúng đoạn chưa mã hóa; không bỏ prompt hoặc đổi precision. Metadata ghi batch thực tế; vẫn FP32, không bật autocast. Phép toán tổng hợp được kiểm tra với cách cũ trên vector tổng hợp; thay kích thước batch trên GPU có thể gây sai số số thực nhỏ, chưa khẳng định dự đoán bit-identical trước khi chạy GPU. Code/requirements/model runtime vẫn được lưu để phân biệt các run trước và sau tối ưu.
