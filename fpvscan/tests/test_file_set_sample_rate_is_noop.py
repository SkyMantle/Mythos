"""File replay must keep the sidecar rate after Engine set_sample_rate."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource


def test_file_set_sample_rate_does_not_change_recorded_rate(tmp_path: Path):
    """#17 `_run` copies `src.sample_rate` into scan/video after
    `set_sample_rate`. #65 a USB-zero is not copied. FileSource is
    the other boot path: the sidecar is 40 Msps, YAML scan is 35.
    Honoring `set_sample_rate(35e6)` would make the copy write 35
    into cfg, LOCK would size grabs/rings at the wrong rate, and
    classify would decimate a 40 Msps dump as if it were 35 — the
    16.2 kHz line falls in the wrong bin and a real VTx looks like
    Wi-Fi. Replay must ignore the request and keep the recorded fs.
    """
    p = tmp_path / "vtx.cf32"
    np.zeros(256, dtype=np.complex64).tofile(p)
    (tmp_path / "vtx.json").write_text(
        '{"center_hz": 5800000000.0, "sample_rate": 40000000.0}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.sample_rate == 40e6
        src.set_sample_rate(35e6)
        assert src.sample_rate == 40e6
        assert src.center_freq == 5800e6
    finally:
        src.close()
