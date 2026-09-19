"""record.py / bringup --save write a sidecar that FileSource must read.

Renaming center_hz → freq_hz (or sample_rate → fs) makes replay silently
tune to 0 Hz at 40 Msps — the capture looks empty and the VTx is blamed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_write_capture_keys_are_what_file_source_reads(tmp_path: Path):
    p = tmp_path / "vtx.cf32"
    iq = np.arange(32, dtype=np.complex64)
    write_capture(p, iq, center_hz=1280e6, sample_rate=10e6,
                  gain_db=12.0, note="bench")
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(meta) >= {
        "center_hz", "sample_rate", "gain_db", "samples",
        "duration_s", "format", "note",
    }
    assert meta["center_hz"] == 1280e6
    assert meta["sample_rate"] == 10e6
    assert meta["gain_db"] == 12.0
    assert meta["samples"] == 32
    assert meta["duration_s"] == 32 / 10e6
    assert meta["format"] == "complex64"
    assert meta["note"] == "bench"

    src = FileSource(p)
    src.open()
    try:
        assert src.center_freq == meta["center_hz"]
        assert src.sample_rate == meta["sample_rate"]
    finally:
        src.close()

    open_src = (Path(__file__).resolve().parents[1]
                / "fpvscan" / "sdr" / "file.py").read_text(encoding="utf-8")
    assert 'self._meta.get("center_hz", 0.0)' in open_src
    assert 'self._meta.get("sample_rate", 40e6)' in open_src
