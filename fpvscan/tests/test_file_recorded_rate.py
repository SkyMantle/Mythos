"""File replay must keep the recorded rate and not dump PLL settle samples."""
from __future__ import annotations

from pathlib import Path
from queue import Queue

import numpy as np

from fpvscan.engine import Engine
from fpvscan.sdr.file import FileSource, write_capture
from tests.helpers import engine_cfg


def test_file_retune_does_not_dump_pll_settle(tmp_path: Path):
    """Base.retune_and_read discards n//4 for bladeRF PLL. A recording has
    no settle: that skip desynchronizes inspect/lock against the capture
    and makes offline replay non-reproducible.
    """
    p = tmp_path / "cap.cf32"
    iq = np.arange(80, dtype=np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=10e6)
    src = FileSource(p, loop=False)
    src.open()
    try:
        a = src.retune_and_read(999e6, 10)
        b = src.retune_and_read(999e6, 10)
        np.testing.assert_array_equal(a, iq[:10])
        np.testing.assert_array_equal(b, iq[10:20])
    finally:
        src.close()


def test_run_copies_recorded_sample_rate_not_yaml_rate(tmp_path: Path):
    """A 40 Msps capture opened under a 35 Msps YAML must not silently
    resample in software — FileSource cannot, and LOCK math would be wrong.
    """
    p = tmp_path / "cap.cf32"
    write_capture(p, np.ones(32, dtype=np.complex64), 1280e6, 40e6)
    src = FileSource(p, loop=True)
    cfg = engine_cfg()
    cfg["scan"]["sample_rate"] = 35e6
    cfg["video"]["sample_rate"] = 35e6
    eng = Engine(src, cfg, Queue())
    eng._stop.set()
    eng._run()
    assert cfg["scan"]["sample_rate"] == 40e6
    assert cfg["video"]["sample_rate"] == 40e6
    assert src.sample_rate == 40e6
    assert src._data is None  # _run finally closes the file
