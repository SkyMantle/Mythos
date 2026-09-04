"""Source factory and file replay — the offline debug path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvscan.sdr.factory import make_source
from fpvscan.sdr.file import FileSource, write_capture
from fpvscan.sdr.sim import SimSource


def test_make_source_sim():
    src = make_source({"driver": "sim"})
    assert isinstance(src, SimSource)
    assert src.name == "sim"


def test_make_source_file(tmp_path: Path):
    p = tmp_path / "cap.cf32"
    iq = np.ones(64, dtype=np.complex64)
    write_capture(p, iq, center_hz=1280e6, sample_rate=10e6, gain_db=12, note="t")
    src = make_source({"driver": "file", "path": str(p), "loop": True})
    assert isinstance(src, FileSource)
    src.open()
    try:
        assert src.fixed_freq
        assert src.center_freq == 1280e6
        assert src.sample_rate == 10e6
        chunk = src.read(16)
        assert chunk.shape == (16,)
        assert chunk.dtype == np.complex64
    finally:
        src.close()


def test_make_source_unknown_driver():
    with pytest.raises(ValueError, match="Невідомий"):
        make_source({"driver": "hackrf"})


def test_file_source_loops_and_raises_without_loop(tmp_path: Path):
    p = tmp_path / "short.cf32"
    write_capture(p, np.arange(8, dtype=np.complex64), 5800e6, 1e6)
    looping = FileSource(p, loop=True)
    looping.open()
    a = looping.read(8)
    b = looping.read(8)
    np.testing.assert_array_equal(a, b)
    looping.close()

    once = FileSource(p, loop=False)
    once.open()
    once.read(8)
    with pytest.raises(EOFError):
        once.read(1)
    once.close()


def test_file_source_empty_file_raises(tmp_path: Path):
    p = tmp_path / "empty.cf32"
    p.write_bytes(b"")
    src = FileSource(p)
    with pytest.raises(IOError, match="Порожній"):
        src.open()


def test_file_source_read_before_open_raises(tmp_path: Path):
    src = FileSource(tmp_path / "nope.cf32")
    with pytest.raises(IOError, match="не відкрите"):
        src.read(4)


def test_write_capture_writes_sidecar_json(tmp_path: Path):
    p = tmp_path / "x.cf32"
    write_capture(p, np.zeros(4, dtype=np.complex64), 100e6, 2e6)
    meta = (tmp_path / "x.json").read_text(encoding="utf-8")
    assert "100000000" in meta or "100e6" in meta or "1e+08" in meta
    assert "complex64" in meta
