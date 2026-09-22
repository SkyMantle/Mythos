"""File replay realtime=True must pace at n/fs, and only then."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import time

from fpvscan.sdr.file import FileSource, write_capture


def test_realtime_sleeps_n_over_fs(tmp_path: Path, monkeypatch):
    """Windows/CI debug path. A forgotten import of wall-clock sleep on
    the default (realtime=False) path is already covered. This pins the
    *on* path: 16 samples @ 8 Msps is 2 µs, not 16 seconds.
    """
    p = tmp_path / "cap.cf32"
    write_capture(p, np.ones(32, dtype=np.complex64), 5800e6, 8e6)
    src = FileSource(p, realtime=True)
    src.open()
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(float(s)))
    try:
        out = src.read(16)
        assert len(out) == 16
        assert slept == [16 / 8e6]
    finally:
        src.close()


def test_realtime_false_does_not_import_sleep(tmp_path: Path, monkeypatch):
    p = tmp_path / "cap.cf32"
    write_capture(p, np.ones(8, dtype=np.complex64), 5800e6, 1e6)
    src = FileSource(p, realtime=False)
    src.open()
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(float(s)))
    try:
        src.read(8)
        assert slept == []
    finally:
        src.close()
