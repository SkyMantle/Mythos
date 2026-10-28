#!/usr/bin/env python3
"""Регресія: мертвий ffmpeg не повинен лишати «Запис» увімкненим."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.engine import Engine
from fpvscan.recorder import VideoRecorder


FAKE_FFMPEG = r"""#!/usr/bin/env python3
import sys
sys.stderr.write("Unknown encoder 'libx264'\n")
sys.stderr.flush()
sys.exit(1)
"""


def _fake_ffmpeg() -> Path:
    fd, name = tempfile.mkstemp(prefix="fake-ffmpeg-", suffix=".py")
    os.close(fd)
    p = Path(name)
    p.write_text(FAKE_FFMPEG, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_recorder_not_alive_after_ffmpeg_exits():
    exe = _fake_ffmpeg()
    out = Path(tempfile.mkdtemp()) / "rec.mp4"
    rec = VideoRecorder(out, 64, 48, fps=20, exe=str(exe))
    rec.start()
    rec.push(np.zeros((48, 64), dtype=np.uint8))
    deadline = time.time() + 2.0
    while rec.alive and time.time() < deadline:
        time.sleep(0.05)
    try:
        assert not rec.alive, "ffmpeg уже вийшов — alive має бути False"
        assert rec.active is False
        assert rec._failed
    finally:
        st = rec.stop()
        exe.unlink(missing_ok=True)
    assert st["bytes"] == 0
    assert "libx264" in st["error"]


class _Source:
    name = "mock"

    def __init__(self):
        self._fs = 2_000_000.0

    @property
    def sample_rate(self):
        return self._fs

    @property
    def center_freq(self):
        return 5800e6

    def open(self):
        pass

    def close(self):
        pass

    def set_gain(self, db):
        pass

    def set_sample_rate(self, hz):
        self._fs = float(hz)

    def read(self, n: int) -> np.ndarray:
        return np.zeros(int(n), dtype=np.complex64)

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        return self.read(n)


def test_engine_snapshot_clears_recording_and_emits_error():
    exe = _fake_ffmpeg()
    events = Queue()
    eng = Engine(_Source(), {
        "sdr": {"gain_db": 30},
        "scan": {"sample_rate": 2e6, "start_hz": 400e6, "stop_hz": 6000e6,
                 "confirm_hits": 2},
        "video": {"sample_rate": 2e6, "channel_bw_hz": 8e6, "idle_ms": 0,
                  "capture_ms": 10, "ring_seconds": 0.05, "afc": False,
                  "width": 64, "spectrum_every": 10_000, "min_lines": 250},
    }, events)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6

    out = Path(tempfile.mkdtemp()) / "rec.mp4"
    rec = VideoRecorder(out, 64, 48, fps=20, exe=str(exe))
    rec.start()
    rec.push(np.zeros((48, 64), dtype=np.uint8))
    deadline = time.time() + 2.0
    while rec.alive and time.time() < deadline:
        time.sleep(0.05)
    eng._rec = rec

    snap = eng.snapshot()
    assert snap["recording"] is False, snap
    assert snap["rec_seconds"] == 0

    eng._rec_watchdog()
    assert eng._rec is None

    notices = []
    while True:
        try:
            ev = events.get_nowait()
        except Empty:
            break
        if ev.get("type") == "notice":
            notices.append(ev["data"])
    try:
        assert notices, "оператор має побачити помилку, не мовчазний зрив"
        assert notices[-1]["level"] == "error"
        assert "обірвався" in notices[-1]["text"]
        assert "libx264" in notices[-1]["text"]
    finally:
        exe.unlink(missing_ok=True)


if __name__ == "__main__":
    test_recorder_not_alive_after_ffmpeg_exits()
    test_engine_snapshot_clears_recording_and_emits_error()
    print("OK")
