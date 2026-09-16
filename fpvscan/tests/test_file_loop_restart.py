"""File replay loop must restart the capture, not splice tail+head."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_loop_restarts_from_start_when_a_read_would_straddle_eof(tmp_path: Path):
    """A 3 s recording used as the whole world loops like a tape:
    the leftover tail of the last chunk is dropped and the next read
    begins at sample 0. Splicing tail+head looks smoother but jumps
    CVBS phase to a non-adjacent instant and LOCK rolls one field
    every loop. Operators debugging with short .cf32 files need the
    seam to stay a clean restart.
    """
    p = tmp_path / "cap.cf32"
    iq = np.arange(10, dtype=np.complex64)
    write_capture(p, iq, 5800e6, 1e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        a = src.read(6)
        np.testing.assert_array_equal(a, iq[:6])
        b = src.read(6)
        np.testing.assert_array_equal(b, iq[:6])
        wrapped = np.concatenate([iq[6:], iq[:2]])
        assert not np.array_equal(b, wrapped)
    finally:
        src.close()
