"""Recorded mp4 must play in a browser, not only in ffplay."""
from __future__ import annotations

from fpvscan.recorder import VideoRecorder


class _FakeProc:
    stdin = None
    stderr = None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_ffmpeg_argv_is_even_yuv420_with_faststart(tmp_path, monkeypatch):
    """yuv420p + even scale is what Chrome/Safari accept. faststart
    moves the moov atom so the file plays while still downloading.
    GOP = 2×fps keeps a seek point every other second at rec_fps=24.
    Drop any of these and the operator gets a black or unseekable clip.
    """
    seen: list[list[str]] = []

    def fake_popen(cmd, **_kw):
        seen.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr("fpvscan.recorder.ffmpeg_path",
                        lambda explicit=None: "/bin/ffmpeg")
    monkeypatch.setattr("fpvscan.recorder.subprocess.Popen", fake_popen)

    rec = VideoRecorder(tmp_path / "t.mp4", 640, 288, fps=24, crf=24,
                        preset="veryfast")
    rec.start()
    rec.stop()

    assert seen, "ffmpeg was never spawned"
    cmd = seen[0]
    assert cmd[cmd.index("-pix_fmt") + 1] == "gray"
    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in cmd
    assert "+faststart" in cmd
    assert cmd[cmd.index("-g") + 1] == "48"
    assert cmd[cmd.index("-framerate") + 1] == "24"
    assert cmd[cmd.index("-preset") + 1] == "veryfast"
