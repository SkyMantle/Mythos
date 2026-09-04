"""ffmpeg lookup and frame resize — recording must fail loudly, not write junk."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvscan.recorder import FfmpegMissing, ffmpeg_path, _resize_nearest


def test_ffmpeg_path_missing_explicit_raises(tmp_path: Path):
    with pytest.raises(FfmpegMissing, match="не існує"):
        ffmpeg_path(str(tmp_path / "no-such-ffmpeg"))


def test_ffmpeg_path_accepts_existing_file(tmp_path: Path):
    exe = tmp_path / "ffmpeg"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    assert ffmpeg_path(str(exe)) == str(exe)


def test_resize_nearest_shape_and_corners():
    src = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = _resize_nearest(src, 6, 8)
    assert out.shape == (6, 8)
    assert out[0, 0] == src[0, 0]
    assert out[-1, -1] == src[-1, -1]
