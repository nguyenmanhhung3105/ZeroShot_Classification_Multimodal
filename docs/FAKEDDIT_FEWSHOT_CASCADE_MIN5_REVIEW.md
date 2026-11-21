# Kiểm tra full dev few-shot cascade min5

Run: `results/final/kaggle/fakeddit_fewshot_cascade_min5_b7c54cfcca4a/run/`. Receipt: `nguyenhung3105/vlm-min5-fakeddit-fewshot-cascade-min5/2`, trạng thái local `collected`. Chỉ kiểm version 2 đã tải, không ghép với version 1 hoặc smoke.

**Audit PASS**: 18 artifact đúng checksum; 30 tổ hợp method–seed, 90.000 prediction và 3.000 ID dev duy nhất. Source code/snapshot khớp bản gửi. Hai policy dùng đúng support nguồn, 384 ID/seed, hợp train mã hóa 1.591 ID; same modality ảnh–tiêu đề, backbone đóng băng và C=1 balanced. Manifest train/dev và thứ tự ID/nhãn/hash cohort khớp nguồn min5.

Đã tính lại accuracy/macro-F1/BA, mean/sample std từ prediction; replay flat/hard routing theo posterior và ngưỡng cố định khớp mọi method/seed. Kiểm lại bảng audit 1.500 ID, paired delta/sửa sai/làm sai, gate, child, oracle, subtype và decomposition lỗi. Posterior binary gate khớp **bit-for-bit** cả 10 bộ của run Task 1 nguồn (max difference 0; không đổi argmax). Thêm tầng hai không làm thay đổi gate nguồn.

Không nạp backbone, đọc dataset/ảnh/text gốc, đọc feature cache hoặc gửi GPU trong lượt review. Checksum head artifact được xác minh, không fit lại classifier. Chi tiết máy đọc được: [full_audit.json](../results/final/kaggle/fakeddit_fewshot_cascade_min5_b7c54cfcca4a/review/full_audit.json).

## Kết quả sáu lớp trên full dev

Mỗi ô là mean ± sample std qua năm seed, đơn vị %. Std trên cùng dev và các support có thể chồng lấp, không phải CI.

| Policy | Phương pháp | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| 64 Real / 320 Fake | Flat 6way | 70,19 ± 1,46 | 60,35 ± 1,79 | 68,62 ± 1,17 |
| 64 Real / 320 Fake | Cascade default | 74,35 ± 1,09 | 62,42 ± 1,61 | 68,22 ± 1,11 |
| 64 Real / 320 Fake | Cascade target recall 87% | 74,60 ± 1,33 | 62,61 ± 1,73 | 68,11 ± 1,12 |
| 192 Real / 192 Fake | Flat 6way | 74,07 ± 1,29 | **62,84 ± 1,15** | 67,97 ± 0,89 |
| 192 Real / 192 Fake | Cascade default | **76,22 ± 0,50** | 62,49 ± 0,98 | 65,26 ± 1,05 |
| 192 Real / 192 Fake | Cascade target recall 87% | 74,70 ± 1,01 | 62,31 ± 1,38 | 68,15 ± 0,74 |

Target 87% là mục tiêu recall Fake dùng chọn ngưỡng trên selection dev trước đó, không phải probability threshold=0,87 và không đảm bảo recall đúng 87% trên toàn dev/audit.

## Đối chứng ghép cặp với flat cùng policy

| Policy / gate | Δ Accuracy (điểm %) | Δ Macro-F1 (điểm %) | Δ BA (điểm %) |
|---|---:|---:|---:|
| Legacy / default | +4,16 | +2,07 | −0,39 |
| Legacy / target 87% | +4,41 | +2,26 | −0,50 |
| Balanced / default | +2,15 | −0,35 | −2,71 |
| Balanced / target 87% | +0,63 | −0,53 | +0,18 |

Với legacy, hai cascade tăng accuracy và macro-F1 ở cả năm seed so flat cùng support; balanced accuracy trung bình giảm nhẹ. Với balanced, cascade default tăng accuracy ở cả năm seed nhưng macro-F1 chỉ tăng 2/5 seed; target tăng macro-F1 1/5 seed. Không kết luận phân cấp cải thiện mọi metric hoặc mọi policy.

Macro-F1 trung bình cao nhất trong sáu nhánh là **flat balanced 62,84%**. Cascade có macro-F1 cao nhất là legacy target 62,61%, thấp hơn 0,23 điểm %. Chênh nhỏ này chưa có kiểm định độc lập; không tuyên bố flat thắng có ý nghĩa thống kê, cũng không coi accuracy 76,22% là bằng chứng cascade vượt baseline mạnh nhất về metric chính macro-F1.

## Phần audit dev đã tách trước

Tất cả phương pháp dùng đúng cùng 1.500 audit ID; không chọn lại ngưỡng trên phần này.

| Nhánh | Accuracy (%) | Macro-F1 (%) | BA (%) |
|---|---:|---:|---:|
| Flat legacy | 70,45 | 60,23 | 68,26 |
| Cascade legacy default | 74,65 | 62,55 | 68,39 |
| Cascade legacy target | 74,83 | 62,63 | 68,29 |
| Flat balanced | 74,56 | **62,85** | 67,52 |
| Cascade balanced default | **76,03** | 61,77 | 64,36 |
| Cascade balanced target | 74,77 | 62,08 | 67,94 |

Thứ tự theo macro-F1 vẫn chưa ủng hộ cascade vượt flat balanced. Đây là kiểm tra phát triển nội bộ: full dev đã được xem khi thiết kế thí nghiệm, không gọi phần audit này là test độc lập.

## Gate, tầng con và lỗi định tuyến

| Policy / gate | Binary accuracy (%) | Recall Real (%) | Recall Fake (%) |
|---|---:|---:|---:|
| Legacy default | 81,22 | 72,71 | 86,91 |
| Legacy target | 81,38 | 73,89 | 86,38 |
| Balanced default | 82,41 | 85,21 | 80,53 |
| Balanced target | 82,11 | 75,39 | 86,60 |

Child năm lớp đo trên **tất cả Fake thật**, không điều kiện gate nhận đúng: legacy accuracy 84,63%, macro-F1 71,13%; balanced accuracy 83,55%, macro-F1 69,88%. Cả hai vẫn có cùng 384 support tổng/seed, nhưng child legacy dùng 320 Fake, balanced dùng 192 Fake. Không quy chênh lệch riêng cho chất lượng thuật toán khi ngân sách child khác nhau.

Số lỗi trung bình/seed trên 3.000 dev:

| Gate | Real→Fake | Fake→Real | Sai subtype sau gate Fake đúng | Tổng lỗi |
|---|---:|---:|---:|---:|
| Legacy default | 328,0 | 235,4 | 206,0 | 769,4 |
| Legacy target | 313,8 | 244,8 | 203,4 | 762,0 |
| Balanced default | 177,8 | 350,0 | 185,6 | 713,4 |
| Balanced target | 295,8 | 241,0 | 222,2 | 759,0 |

Ba nhóm lỗi tách rời cộng đúng tổng lỗi sáu lớp ở từng seed. Lỗi định tuyến binary chiếm khoảng 71–74% lỗi end-to-end trong các nhánh này; child vẫn còn lỗi riêng. Accuracy child chỉ trên Fake được gate nhận đúng cao hơn accuracy trên mọi Fake, nên không dùng conditional accuracy để thay metric toàn tầng con.

Oracle gate dùng binary label thật cho accuracy sáu lớp khoảng 90,79% legacy và 90,14% balanced. Đây là chẩn đoán ảnh hưởng gate khi cố định child, không phải phương pháp dùng được khi inference và không là cận trên cho mọi metric.

## Subtype khó

Recall end-to-end (%), trung bình năm seed:

| Lớp | Legacy default | Legacy target | Balanced default | Balanced target |
|---|---:|---:|---:|---:|
| Real | 72,71 | 73,89 | 85,21 | 75,39 |
| Satire/Parody | 63,74 | 63,63 | 59,89 | 63,30 |
| False Connection | 47,13 | 46,12 | 33,74 | 43,01 |
| Imposter | 49,06 | 49,06 | 41,88 | 50,00 |
| Manipulated | 98,11 | 98,11 | 98,18 | 98,29 |
| Misleading | 78,57 | 77,86 | 72,68 | 78,93 |

Balanced default phục hồi Real nhưng mất recall Fake, đặc biệt False Connection. Điều chỉnh ngưỡng giúp phục hồi phần Fake nhưng không tạo lợi ích macro-F1 so flat balanced. Recall subtype luôn bị chặn bởi khả năng gate nhận ra Fake của subtype đó; không thể chỉ cải tiến child rồi kỳ vọng sửa các Fake đã bị gate chuyển thành Real.

## Quyết định sau review

- Đã có bằng chứng vận hành và metric cho pipeline few-shot hai tầng trên **cùng min5 dev**, tách khỏi hierarchy cohort cũ.
- Giữ flat balanced làm đối chứng mạnh theo macro-F1 và legacy cascade làm ứng viên phân cấp. Chưa thay cấu hình nguồn, chưa chọn seed tốt nhất hoặc mở test.
- Nếu mục tiêu là hoàn thành báo cáo, kết quả có cả lợi ích và đánh đổi đã trả lời được câu hỏi phân cấp. Không bắt buộc thêm module hoặc quét ngưỡng để tìm một cascade thắng mọi metric.
- Đã chốt giữ sáu nhánh và macro-F1 chính, có [draft protocol và runner test hai tầng](CASCADE_TEST_PROTOCOL.md). Còn cần runtime đích và manifest test min5 để niêm phong; chưa mở test. Không dùng lock Task 1 một tầng thay thế hoặc tune bằng test. Full dev và audit dev đều còn giới hạn phát triển.

Nạp model/mã hóa khoảng 981,97 giây; toàn runner 1.040,97 giây (~17 phút 21 giây), không gồm queue/bootstrap notebook. Không đưa kết quả smoke vào bảng chất lượng này.

Tự rà soát theo `agent-self-evaluation`: chính xác 4/5 (90.000 prediction/checksum/replay khớp, chưa tái suy luận encoder); đầy đủ 4/5 (đủ full/audit/gate/child/subtype, chưa có CI độc lập); rõ ràng 4/5 (tách six-way khỏi binary và oracle, bảng tương đối dài); khả thi 4/5 (nêu rõ ứng viên và protocol tiếp theo, chưa chọn test suite cuối); súc tích 4/5 (chi tiết để truy vết, phần tóm tắt trả lời chỉ giữ bảng chính). Trung bình 4,0/5. Ưu tiên cải thiện: test protocol hai tầng có kiểm tra trước model, xác nhận độc lập theo suite đã khóa; không dùng std seed làm CI. Kết luận có thể được đối chiếu từ các artifact liên kết, không phải đánh giá ngoài dữ liệu phát triển.
