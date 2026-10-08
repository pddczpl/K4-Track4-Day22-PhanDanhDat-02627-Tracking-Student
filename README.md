# Lab Tracking — 2 giờ

Nhóm 2 người một máy. Detector đã khóa. Bạn chọn tracker và ngưỡng cho năm video khác cảnh.

## Bài nộp của Phan Danh Đạt — 02627

- [Báo cáo đã điền](submission_template/BAO_CAO_mau.md): cấu hình, số liệu và nhận xét từng video.
- Năm file kết quả đủ frame: [video_1](runs/nop_bai/video_1.txt), [video_2](runs/nop_bai/video_2.txt), [video_3](runs/nop_bai/video_3.txt), [video_4](runs/nop_bai/video_4.txt), [video_5](runs/nop_bai/video_5.txt).
- [Kết quả chấm video_1](submission_template/VIDEO_1_METRICS.txt), [so sánh ba cấu hình đủ frame](submission_template/SO_SANH_VIDEO_1.csv), [nhật ký thử cấu hình](submission_template/THU_NGHIEM.csv), [kiểm tra đầu ra và SHA-256](submission_template/KIEM_TRA_DAU_RA.csv).
- [Notebook ôn tập](on_tap_metrics.ipynb) đã điền đáp án và lưu kết quả chạy dạng văn bản.

`.gitignore` cho phép đưa đúng năm file kết quả vào Git. Dữ liệu, trọng số, video xem thử và các lần chạy thử được giữ ngoài bài nộp.

Để chạy lại năm cấu hình đã chọn trên Windows, kích hoạt môi trường đã cài các thư viện rồi chạy từ thư mục repo:

```powershell
$env:LAB_DATA = "D:\du_lieu\lab_data"  # Thay bằng thư mục dữ liệu của bạn
python scripts/check_data.py --lab-data-root "$env:LAB_DATA"
python run_batch.py --lab-data-root "$env:LAB_DATA" --out runs/chay_lai --device cuda:0 --save-video
pytest
```

`run_batch.py` cũng tự đọc `LAB_DATA` nếu không truyền `--lab-data-root`. Dùng `--device cpu` khi không có GPU. Script từ chối ghi đè file đã có; chỉ thêm `--overwrite` khi muốn thay kết quả. Để chấm lần chạy lại, truyền `runs/chay_lai/video_1.txt` cho lệnh chấm bên dưới. Số frame của năm video lần lượt là **600 / 1050 / 837 / 900 / 750**.

## Việc cần làm

1. Tạo môi trường một lần:

```bash
conda env create -f environment.yml
conda activate cv_robotics_lab21
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -e TrackEval/
```

2. Tải ảnh năm video: [data_lab21.zip](https://drive.google.com/file/d/1UeVPQd6j5pSzxoJDcKJrerT9SL3vJLDt/view?usp=sharing). Giải nén, rồi gán đường dẫn thư mục chứa `video_1` … `video_5`:

```bash
export LAB_DATA=/đường/dẫn/lab_data
python scripts/check_data.py --lab-data-root "$LAB_DATA"
```

Cả năm video phải có ảnh. Chỉ `video_1` có nhãn.

3. Mở `on_tap_metrics.ipynb` bằng kernel env này. Đọc bảng MOTA / IDF1 / HOTA, rồi chạy YOLO trên một ảnh `video_1`.

4. Chạy thử tracker trên 150 frame. Dùng `run_batch.py` ở đầu trang để chạy lại cấu hình bản nộp đủ frame.

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" \
  --seq-name video_1 \
  --tracker bytetrack --conf 0.3 --iou 0.5 \
  --out runs/thu_nhanh --save-video --max-frames 150
```

Đổi `--seq-name`, thư mục `img1` và thư mục `--out` khi thử `video_2` … `video_5` hoặc cấu hình khác. Tracker được chọn: `bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort`.

5. Chấm số **chỉ** `video_1`:

```bash
python scripts/evaluate_practice.py \
  --trackeval-root ~/TrackEval \
  --lab-data-root "$LAB_DATA" \
  --submission runs/nop_bai/video_1.txt \
  --run-name nhom01_video1
```

`video_2` đến `video_5` không có nhãn. Xem `preview/video_N.mp4` và video có vẽ ID, rồi ghi điều bạn thấy.

## Luật chơi

| Khóa | Bạn chọn |
|---|---|
| Detector `yolo26n.pt`, ảnh 640 px, lớp người, Re-ID `osnet_x0_25_msmt17.pt` | Tracker, `--conf`, `--iou` của detector |

## Nộp

- `video_1.txt` … `video_5.txt` trong `runs/nop_bai/` (đủ frame, đúng tên).
- `submission_template/BAO_CAO_mau.md` đã điền. Số HOTA / MOTA / IDF1 chỉ bắt buộc cho `video_1`.

Chi tiết từng bước, sự cố, và lịch 2 giờ: [HUONG_DAN.md](HUONG_DAN.md).
