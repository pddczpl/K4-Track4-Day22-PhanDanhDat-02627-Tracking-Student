"""Kiểm tra lệnh chạy hàng loạt bằng dữ liệu tạm, không nạp mô hình."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "run_batch", Path(__file__).resolve().parents[1] / "run_batch.py"
)
run_batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_batch)


@pytest.fixture
def lab_root(tmp_path: Path) -> Path:
    """Tạo cây thư mục giả cho năm video.

    Args:
        tmp_path: Thư mục tạm do pytest cung cấp.

    Returns:
        Thư mục dữ liệu chứa các file .jpg rỗng, không có ảnh thật.
    """
    root = tmp_path / "du lieu"
    for index in range(1, 6):
        images = root / f"video_{index}" / "img1"
        images.mkdir(parents=True)
        (images / "000001.jpg").touch()
    return root


@pytest.mark.parametrize(
    "index,tracker,conf",
    [(1, "botsort", 0.30), (2, "deepocsort", 0.25), (3, "bytetrack", 0.25),
     (4, "ocsort", 0.40), (5, "botsort", 0.30)],
)
def test_commands_preserve_submission_config(tmp_path, index, tracker, conf):
    root = tmp_path / "du lieu"
    out = tmp_path / "ket qua"
    command = run_batch.build_command(
        run_batch.EXPERIMENTS[index - 1], root, out, "cpu", False
    )
    assert command == [
        sys.executable, str(run_batch.REPO_ROOT / "scripts" / "run_tracking.py"),
        "--source", str(root / f"video_{index}" / "img1"),
        "--seq-name", f"video_{index}", "--tracker", tracker,
        "--conf", str(conf), "--iou", "0.5", "--device", "cpu", "--out", str(out),
    ]


def test_command_adds_preview_only_when_requested(tmp_path):
    command = run_batch.build_command(
        run_batch.EXPERIMENTS[0], tmp_path, tmp_path / "out", "cuda:0", True
    )
    assert command[-1] == "--save-video"


def test_parse_args_uses_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_DATA", str(tmp_path))
    args = run_batch.parse_args([])
    assert args.lab_data_root == tmp_path.resolve()
    assert args.out == run_batch.REPO_ROOT / "runs" / "nop_bai"
    assert args.device == "cuda:0"
    assert not args.overwrite


def test_explicit_data_root_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_DATA", str(tmp_path / "unused"))
    args = run_batch.parse_args(["--lab-data-root", str(tmp_path)])
    assert args.lab_data_root == tmp_path.resolve()


def test_missing_data_root_is_rejected(monkeypatch):
    monkeypatch.delenv("LAB_DATA", raising=False)
    with pytest.raises(SystemExit) as error:
        run_batch.parse_args([])
    assert error.value.code == 2


def test_checks_all_sources_before_running(monkeypatch, lab_root, tmp_path):
    (lab_root / "video_5" / "img1" / "000001.jpg").unlink()
    runner = Mock()
    monkeypatch.setattr(run_batch.subprocess, "run", runner)
    out = tmp_path / "out"
    assert run_batch.main(["--lab-data-root", str(lab_root), "--out", str(out)]) == 2
    runner.assert_not_called()
    assert not out.exists()


@pytest.mark.parametrize("existing", ["video_5.txt", "video_5_preview.mp4"])
def test_existing_outputs_are_preserved(monkeypatch, lab_root, tmp_path, existing):
    out = tmp_path / "out"
    out.mkdir()
    result = out / existing
    result.write_bytes(b"giu nguyen")
    runner = Mock()
    monkeypatch.setattr(run_batch.subprocess, "run", runner)
    assert run_batch.main([
        "--lab-data-root", str(lab_root), "--out", str(out), "--save-video"
    ]) == 2
    runner.assert_not_called()
    assert result.read_bytes() == b"giu nguyen"


def test_explicit_overwrite_runs_five_commands(monkeypatch, lab_root, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "video_1.txt").touch()
    runner = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(run_batch.subprocess, "run", runner)
    assert run_batch.main([
        "--lab-data-root", str(lab_root), "--out", str(out),
        "--overwrite", "--device", "cpu", "--save-video",
    ]) == 0
    assert runner.call_count == 5
    for call, experiment in zip(runner.call_args_list, run_batch.EXPERIMENTS):
        assert call.args[0] == run_batch.build_command(experiment, lab_root, out, "cpu", True)
        assert call.kwargs == {"cwd": run_batch.REPO_ROOT}


def test_child_failure_stops_remaining_videos(monkeypatch, lab_root, tmp_path):
    runner = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(run_batch.subprocess, "run", runner)
    assert run_batch.main([
        "--lab-data-root", str(lab_root), "--out", str(tmp_path / "out")
    ]) == 7
    assert runner.call_count == 1
