"""File replay through Engine._sweep_plan is one LO — the recorded center."""
from __future__ import annotations

from pathlib import Path
from queue import Queue

import numpy as np

from fpvscan.engine import Engine
from fpvscan.sdr.file import FileSource, write_capture
from tests.helpers import engine_cfg


def test_file_source_sweep_plan_is_the_recorded_center(tmp_path: Path):
    """#28 pins FileSource.fixed_freq; #5 pins the Engine short-circuit
    on a mock. Together they can still drift: if Engine starts
    checking `name == "file"` or the flag is only set after open(),
    a 3 s F4 recording is «found» on every 26.5 MHz hop and the hit
    list is unusable for algorithm work.
    """
    p = tmp_path / "f4.cf32"
    write_capture(p, np.zeros(32, dtype=np.complex64), 5800e6, 10e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.fixed_freq is True
        eng = Engine(src, engine_cfg(), Queue())
        assert eng._sweep_plan() == [5800e6]
    finally:
        src.close()
