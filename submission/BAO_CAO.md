# Báo cáo lab: chọn tracker cho 5 video

**Người thực hiện:** Phan Danh Đạt — **Mã học viên:** 02627

**Bài học:** K4, Track 4, Day 22 — **Ngày hoàn thiện:** 08/10/2026

Detector cố định `yolo26n.pt`, kích thước đầu vào 640 px, chỉ lớp người (`classes=[0]`); Re-ID cố định `osnet_x0_25_msmt17.pt`. Chỉ thay tracker và `conf` / `iou` của detector; giữ cấu hình nội bộ tracker mặc định của BoxMOT.

## 1. Cấu hình đã chọn

Các file nộp nằm trong `runs/nop_bai/`, đều chạy toàn bộ chuỗi ảnh, không giới hạn frame.

| Video | Tracker | conf | iou | Quan sát trên khung hình có ID | Cấu hình đã thử nhưng không chọn |
|---|---|---|---|---|---|
| video_1 — quảng trường, camera tĩnh | BoT-SORT | 0.30 | 0.50 | Ở frame 50 và 150, có thêm hộp ở phía cửa hàng xa bên phải so với ByteTrack; các người ở tiền cảnh vẫn có hộp. Nhiều người nhỏ phía xa còn bị bỏ sót. | ByteTrack 0.30/0.50 và 0.15/0.50: cả ba điểm trên 600 frame thấp hơn BoT-SORT 0.30/0.50. |
| video_2 — phố đêm, rất đông | DeepOCSORT | 0.25 | 0.50 | Người sát mép phải giữ ID 4 ở frame 1/50/100; người áo trắng phía trên giữ ID 7. Frame 150 có hộp cho người cạnh cột đèn phía dưới mà ByteTrack chưa xuất track; nhóm đông phía trên trái vẫn thiếu nhiều hộp. | ByteTrack 0.25/0.50: chạy nhanh hơn, nhưng thiếu một số track nhìn thấy trong đoạn đối chiếu. |
| video_3 — camera di động, ảnh nhỏ | ByteTrack | 0.25 | 0.50 | Người áo sọc ở tiền cảnh giữ ID 1 tại frame 1/50/100/150; người áo đen phía trước giữ ID 24 tại frame 50/100/150. Người nhỏ và bị che vẫn khó theo dõi. | BoT-SORT 0.25/0.50: người áo sọc cũng giữ ID ở các mốc này, nhưng vòng lặp thử chậm hơn (30.7 so với 59.2 FPS); chưa thấy lợi ích đủ rõ để đổi cấu hình. |
| video_4 — trong nhà, camera tiến tới | OC-SORT | 0.40 | 0.50 | Người áo đỏ và áo trắng phía trước giữ ID 2 và 6 tại frame 50/100/150; ID 6 vẫn thấy ở frame 450/465. Hộp bám các người gần camera, còn người xa hoặc bị che thiếu hộp. | BoT-SORT 0.40/0.50: ở các mốc 50/100/150, hai người chính cũng giữ ID, nhưng vòng lặp thử chậm hơn (20.0 so với 33.0 FPS). |
| video_5 — camera trên xe bus | BoT-SORT | 0.30 | 0.50 | Frame 50/100 có nhiều người được bám ở vỉa hè tối bên trái hơn ByteTrack. Ở frame 375/390, các hộp tiếp tục bám nhóm người bên phải khi camera chuyển hướng; người rất xa vẫn dễ mất track. | ByteTrack 0.30/0.50: nhanh hơn nhưng bỏ qua một số người bên trái trong các frame đối chiếu. |

Nhận xét định tính dựa trên các khung hình trích từ preview: đầu, giữa, gần cuối chuỗi và các mốc 50/100/150 của lượt đối chiếu. Đây là quan sát tại các mốc cụ thể; không suy ra số lần đổi ID hoặc độ chính xác cho toàn bộ `video_2`–`video_5`.

### Cách thử và bằng chứng

Mỗi video được thử ít nhất hai tracker: một tracker theo chuyển động và một tracker có Re-ID. Các lượt thử ngắn dùng cùng 150 frame đầu, `--device cuda:0`, có xuất preview. Khi quét `conf`, giữ `iou=0.50`; khi quét `iou=0.40/0.50/0.70`, giữ `conf` của cấu hình đối chiếu. Đã thử các mức `conf=0.15/0.30/0.50`, ngoài các mức 0.25 và 0.40 dùng trong bản nộp.

Nhật ký cấu hình và thời gian thực đo nằm trong [THU_NGHIEM.csv](THU_NGHIEM.csv). Số dòng và số ID trong nhật ký chỉ mô tả đầu ra, không phải thước đo chất lượng. Các lần thử 150 frame không thay thế kết quả đủ frame. Riêng `video_1`, chạy thêm BoT-SORT 0.30/0.50 và ByteTrack 0.15/0.50 đủ 600 frame để chấm cùng bản ByteTrack 0.30/0.50 ban đầu.

Hoàn thành **39 lượt thử**: 37 lượt 150 frame và 2 lượt 600 frame. Với BoT-SORT trên 150 frame đầu `video_1`, `conf=0.15/0.30/0.50` cho 979/847/560 dòng track; giữ `conf=0.30`, đổi `iou=0.40/0.50/0.70` cho 784/847/905 dòng. Việc giảm `conf` hoặc tăng `iou` trong các lượt này giữ lại nhiều đầu ra hơn, nhưng chưa cho biết có thêm người đúng hay hộp giả. Vì vậy lựa chọn cuối của `video_1` dựa trên các cấu hình đã chấm đủ 600 frame, không chỉ dựa vào số dòng của lượt thử ngắn.

Môi trường thực chạy: Windows, Python 3.11, RTX 3060, PyTorch `2.6.0+cu124`, Ultralytics `8.4.174`, BoxMOT `10.0.42`, NumPy `1.26.4`. FPS ghi trong nhật ký bao gồm phát hiện, tracking và ghi preview; thời gian toàn lượt còn gồm nạp mô hình. Những số này phục vụ so sánh thực nghiệm trên máy đang dùng, không phải tốc độ chuẩn của thuật toán.

## 2. Số liệu video_1

Chấm bằng `scripts/evaluate_practice.py`, chỉ dùng nhãn của `video_1`. Kết quả đủ 600 frame; các chỉ số HOTA/MOTA/IDF1 dưới đây theo thang 0–100 của TrackEval.

| Cấu hình | HOTA | MOTA | IDF1 | Quyết định |
|---|---:|---:|---:|---|
| ByteTrack, conf 0.30, iou 0.50 | 26.911 | 17.292 | 25.713 | Bản ban đầu, không chọn |
| ByteTrack, conf 0.15, iou 0.50 | 27.313 | 18.314 | 26.987 | Có cải thiện nhưng vẫn thấp hơn BoT-SORT |
| **BoT-SORT, conf 0.30, iou 0.50** | **29.462** | **19.816** | **29.353** | **Bản nộp** |

Trích các cột từ output chấm bản nộp:

```text
Video      HOTA     MOTA     IDF1     DetA     AssA
video_1    29.462   19.816   29.353   18.097   48.225

Video      CLR_TP   CLR_FN   CLR_FP   IDSW    Frag
video_1    4044     14537    337      25      99
```

Output tóm tắt nguyên gốc được lưu trong [VIDEO_1_METRICS.txt](VIDEO_1_METRICS.txt); bảng so sánh có thêm TP/FN/FP/IDSW nằm trong [SO_SANH_VIDEO_1.csv](SO_SANH_VIDEO_1.csv). BoT-SORT tăng HOTA **2.551 điểm**, MOTA **2.524 điểm**, IDF1 **3.640 điểm** so với bản đầu. Tuy nhiên, số FP tăng từ 107 lên 337 và IDSW từ 12 lên 25; vì thế không kết luận BoT-SORT ít đổi ID hơn ByteTrack. Phần lợi ích chính là tăng số phát hiện đúng từ 3332 lên 4044, trong khi bỏ sót vẫn rất lớn.

`video_2`–`video_5` không có nhãn trong gói lab; không chấm hoặc điền HOTA/MOTA/IDF1 cho bốn video đó.

## 3. Phân tích

### video_1: ưu tiên điểm tổng hợp, thừa nhận hạn chế phát hiện

Camera tĩnh giúp việc nối vị trí giữa các frame tương đối thuận lợi, nhưng ảnh chứa nhiều người rất nhỏ hoặc ở vùng tối. Khi đối chiếu frame 50/150, BoT-SORT bám thêm một số người phía cửa hàng bên phải mà ByteTrack chưa xuất track. Kết quả chấm đủ chuỗi xác nhận lợi ích tổng hợp của cấu hình này qua cả HOTA, MOTA và IDF1, dù FP và số lần đổi ID đều tăng. DetA chỉ 18.097, thấp hơn AssA 48.225, cùng 14537 FN cho thấy bỏ sót còn là hạn chế lớn; không thể chỉ đổi tracker là giải quyết hết. Chọn BoT-SORT vì điểm tổng thể cao nhất trong ba cấu hình đã chấm, chấp nhận thời gian chạy lớn hơn.

### video_2: ngoại hình hỗ trợ trong cảnh đông

Nhiều người đi gần nhau, có che khuất và ánh sáng ban đêm không đồng đều. DeepOCSORT kết hợp thông tin ngoại hình với chuyển động, nên là lựa chọn đáng thử khi chỉ dựa vào vị trí dễ gây nhầm giữa người đi sát nhau. Trong các mốc đầu, một số ID như 4 và 7 được giữ; frame 150 còn có thêm track cạnh cột đèn so với ByteTrack. Chọn `conf=0.25` để giữ các phát hiện yếu hơn mức 0.50, nhưng vùng đông người phía trên trái vẫn thiếu hộp rõ rệt. Chưa có nhãn nên đây là quyết định từ quan sát và đặc điểm cảnh, không phải kết luận DeepOCSORT thắng trên mọi trường hợp.

### video_3: cân bằng tốc độ và khả năng giữ người ở gần

Camera chuyển động và ảnh nhỏ làm hộp thay đổi vị trí, kích thước nhanh. Hai tracker thử đều giữ người áo sọc ở các mốc 50/100/150; ByteTrack đạt 59.2 FPS so với 30.7 FPS của BoT-SORT trong lượt thử ngắn. Vì chưa thấy cải thiện rõ trên người chính ở những mốc đã xem, giữ ByteTrack để giảm chi phí xử lý. Những người nhỏ phía xa và lúc che khuất vẫn là điểm cần kiểm tra thêm; số ID khác nhau không thể dùng trực tiếp để đếm lỗi đổi ID.

### video_4: chuyển động khá đều của người phía trước

OC-SORT giữ hộp cho hai người áo đỏ và áo trắng trong các mốc đầu khi camera tiến tới, tương tự BoT-SORT ở đoạn đối chiếu. Chọn cấu hình OC-SORT `conf=0.40` vì đáp ứng việc bám nhóm người chính và chạy nhanh hơn trong thử nghiệm. Ngưỡng 0.40 là lựa chọn tương đối chặt trong cảnh có kính và phản chiếu; những ảnh đã xem chưa đủ để khẳng định ngưỡng này loại hết hộp giả. Cần lưu ý việc tăng ngưỡng cũng có thể bỏ mất người nhỏ phía xa, nên không chọn chỉ dựa trên số hộp ít đi.

### video_5: camera rung và chuyển hướng

Chuyển động của xe làm cả nền lẫn người dịch chuyển trong ảnh, trong khi kích thước người thay đổi khi xe tiến gần giao lộ. BoT-SORT có thêm ngoại hình và bù chuyển động camera, phù hợp để thử trong bối cảnh này. Ở frame 50/100, cấu hình BoT-SORT bám thêm một số người bên trái so với ByteTrack; ở cặp 375/390, hộp vẫn đi theo nhóm người bên phải khi góc nhìn đổi. Vì vậy giữ BoT-SORT dù chậm hơn ByteTrack trong thử ngắn (18.9 so với 34.0 FPS). Không xem việc có thêm hộp là bằng chứng tự thân của chất lượng: các người xa và vùng tối vẫn cần quan sát kỹ.

## 4. Nếu có thêm thời gian

Ưu tiên xem lại các frame gây FN và đổi ID trên `video_1`, quét `conf` mịn hơn quanh cấu hình đang chọn nhưng giữ nguyên detector, kích thước ảnh và Re-ID. Với bốn video không có nhãn, mở rộng so sánh sang đoạn giữa/cuối và các đoạn che khuất, thay vì chỉ dựa vào 150 frame đầu.

## 5. Kiểm tra bài nộp và chạy lại

| File | Số frame đầu vào | Số frame preview đã kiểm tra |
|---|---:|---:|
| `runs/nop_bai/video_1.txt` | 600 | 600 |
| `runs/nop_bai/video_2.txt` | 1050 | 1050 |
| `runs/nop_bai/video_3.txt` | 837 | 837 |
| `runs/nop_bai/video_4.txt` | 900 | 900 |
| `runs/nop_bai/video_5.txt` | 750 | 750 |

Cả năm file có đúng 10 cột mỗi dòng, frame bắt đầu từ 1, ID nguyên dương, kích thước hộp dương, confidence trong [0, 1], ba cột cuối bằng -1 và không trùng cặp `(frame, ID)`. Các frame trong file kết quả được sắp tăng dần; file có track đến frame cuối. Số dòng không bằng số frame vì mỗi frame có thể có nhiều người hoặc không có track nào. [KIEM_TRA_DAU_RA.csv](KIEM_TRA_DAU_RA.csv) lưu số dòng, số ID và SHA-256 để đối chiếu đúng bản nộp.

Notebook đã hoàn thành ba câu hỏi (`True`, `False`, `True`) và chạy detector trên frame đầu `video_1`: `conf=0.15/0.30/0.50` lần lượt cho 14/6/5 hộp. Output chỉ có detection, chưa có track ID.

Sau khi kích hoạt môi trường, chạy từ thư mục repo bằng PowerShell:

```powershell
$env:LAB_DATA = "D:\du_lieu\lab_data"  # Thay bằng thư mục dữ liệu thực tế
$trackEvalRoot = "D:\cong_cu\TrackEval"  # Thay bằng bản cài TrackEval
python scripts/check_data.py --lab-data-root "$env:LAB_DATA"
python run_batch.py --out runs/chay_lai --device cuda:0 --save-video
python scripts/evaluate_practice.py --trackeval-root "$trackEvalRoot" --lab-data-root "$env:LAB_DATA" --submission runs/chay_lai/video_1.txt --run-name dat02627_video1_chay_lai
pytest
```

`run_batch.py` chứa đúng năm cấu hình ở mục 1 và mặc định từ chối ghi đè file đã có. Bài nộp Git gồm mã nguồn, notebook, báo cáo, bằng chứng dạng văn bản và năm file kết quả; `.gitignore` loại dữ liệu ảnh, trọng số, preview và toàn bộ lượt thử. Preview vẫn được giữ cục bộ để xem lại.
