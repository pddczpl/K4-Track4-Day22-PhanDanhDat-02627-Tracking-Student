#!/usr/bin/env python
"""Chạy lại năm video với đúng cấu hình đã chọn cho bài nộp.

Ví dụ:
    python run_batch.py --lab-data-root "duong/dan/lab_data" --out runs/chay_lai
    python run_batch.py --out runs/chay_lai --save-video

Lệnh thứ hai đọc thư mục dữ liệu từ biến môi trường LAB_DATA.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    {"seq": "video_1", "tracker": "botsort", "conf": 0.30, "iou": 0.5},
    {"seq": "video_2", "tracker": "deepocsort", "conf": 0.25, "iou": 0.5},
    {"seq": "video_3", "tracker": "bytetrack", "conf": 0.25, "iou": 0.5},
    {"seq": "video_4", "tracker": "ocsort", "conf": 0.40, "iou": 0.5},
    {"seq": "video_5", "tracker": "botsort", "conf": 0.30, "iou": 0.5},
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh và thư mục dữ liệu từ LAB_DATA.

    Args:
        argv: Danh sách tham số; None để đọc dòng lệnh hiện tại.

    Returns:
        Namespace chứa đường dẫn dữ liệu, đầu ra và tùy chọn chạy.

    Raises:
        SystemExit: Khi thiếu thư mục dữ liệu hoặc tham số không hợp lệ.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lab-data-root", type=Path, default=os.environ.get("LAB_DATA") or None,
        help="Thư mục chứa video_1 đến video_5; mặc định đọc LAB_DATA",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs" / "nop_bai",
        help="Thư mục xuất kết quả; mặc định runs/nop_bai trong repo",
    )
    parser.add_argument("--device", default="cuda:0", help="Thiết bị, ví dụ cuda:0 hoặc cpu")
    parser.add_argument("--save-video", action="store_true", help="Xuất thêm video xem thử")
    parser.add_argument(
        "--overwrite", action="store_true", help="Cho phép ghi đè kết quả đã có"
    )
    args = parser.parse_args(argv)
    if args.lab_data_root is None:
        parser.error("Hãy truyền --lab-data-root hoặc đặt biến môi trường LAB_DATA.")
    args.lab_data_root = args.lab_data_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    return args


def build_command(
    experiment: dict[str, str | float],
    lab_data_root: Path,
    out_dir: Path,
    device: str,
    save_video: bool,
) -> list[str]:
    """Tạo lệnh chạy một video từ cấu hình đã chọn.

    Args:
        experiment: Cấu hình gồm seq, tracker, conf và iou.
        lab_data_root: Thư mục chứa năm video.
        out_dir: Thư mục nhận kết quả.
        device: Thiết bị truyền cho bộ theo dõi.
        save_video: Có xuất video xem thử hay không.

    Returns:
        Danh sách đối số dùng trực tiếp với subprocess.run.
    """
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tracking.py"),
        "--source", str(lab_data_root / str(experiment["seq"]) / "img1"),
        "--seq-name", str(experiment["seq"]),
        "--tracker", str(experiment["tracker"]),
        "--conf", str(experiment["conf"]),
        "--iou", str(experiment["iou"]),
        "--device", device,
        "--out", str(out_dir),
    ]
    if save_video:
        command.append("--save-video")
    return command


def main(argv: list[str] | None = None) -> int:
    """Kiểm tra đầu vào rồi lần lượt chạy năm video.

    Args:
        argv: Danh sách tham số; None để đọc dòng lệnh hiện tại.

    Returns:
        Mã 0 khi thành công, 2 khi đầu vào chưa sẵn sàng, hoặc mã lỗi tiến trình con.

    Raises:
        SystemExit: Khi tham số dòng lệnh không hợp lệ.
    """
    args = parse_args(argv)
    for experiment in EXPERIMENTS:
        seq = str(experiment["seq"])
        source_dir = args.lab_data_root / seq / "img1"
        if not source_dir.is_dir() or not any(source_dir.glob("*.jpg")):
            print(f"Lỗi: không tìm thấy thư mục ảnh .jpg hợp lệ: {source_dir}", file=sys.stderr)
            return 2
        outputs = [args.out / f"{seq}.txt"]
        if args.save_video:
            outputs.append(args.out / f"{seq}_preview.mp4")
        for output in outputs:
            if output.exists() and not args.overwrite:
                print(
                    f"Lỗi: đã có {output}. Chọn --out khác hoặc dùng --overwrite để chạy lại.",
                    file=sys.stderr,
                )
                return 2

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Bắt đầu chạy 5 video với YOLO26n trên {args.device}.", flush=True)
    for experiment in EXPERIMENTS:
        seq = str(experiment["seq"])
        print(
            f"\nĐang chạy {seq}: tracker={experiment['tracker']}, "
            f"conf={experiment['conf']}, iou={experiment['iou']}...",
            flush=True,
        )
        started = time.perf_counter()
        result = subprocess.run(
            build_command(experiment, args.lab_data_root, args.out, args.device, args.save_video),
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"Lỗi khi chạy {seq} (mã {result.returncode}).", file=sys.stderr)
            return result.returncode
        print(f"Xong {seq} trong {time.perf_counter() - started:.1f}s.", flush=True)

    print("\nTất cả 5 video đã chạy xong!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
