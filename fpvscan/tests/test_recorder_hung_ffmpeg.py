"""Hung ffmpeg must be killed; the pump must not write before a frame exists."""
from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path

from fpvscan.recorder import VideoRecorder


def test_stop_kills_ffmpeg_after_wait_timeout(tmp_path: Path):
    """Pi disk fills if a wedged x264 is left as a zombie. stop() waits
    10 s then kill(); stderr is returned so the notice is not empty.
    """
    rec = VideoRecorder(tmp_path / "clip.mp4", 8, 8, fps=5)
    rec.started_at = time.time() - 1.0

    class Fake:
        stdin = io.BytesIO()
        stderr = io.BytesIO(b"x264 stalled")
        killed = False

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)

        def kill(self):
            self.killed = True

    fake = Fake()
    rec._proc = fake
    st = rec.stop()
    assert fake.killed
    assert rec._proc is None
    assert rec.active is False
    assert "stalled" in st["error"]
    assert st["frames"] == 0


def test_pump_skips_when_no_frame_has_been_pushed():
    """First seconds of a recording used to write whatever was in
    _latest (None → TypeError / garbage bytes). The skip is the
    contract; we stop immediately so this stays deterministic.
    """
    rec = VideoRecorder(Path("/tmp/unused.mp4"), 4, 4, fps=50)
    rec._latest = None
    rec._stop.set()
    rec._pump()
    assert rec.frames_written == 0
    src = Path(__file__).resolve().parents[1] / "fpvscan" / "recorder.py"
    body = src.read_text(encoding="utf-8")
    pump = body[body.index("def _pump"):body.index("def stop")]
    assert "if fr is None or self._proc is None" in pump
