"""File replay must not let the consumer mutate the capture in RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_read_returns_a_copy(tmp_path: Path):
    """read() ends with `out.copy()`. Inspect/LOCK subtract the mean and
    channelize in place on that array. Without the copy, a looped
    replay (the default debug path) sees a DC-stripped / rotated
    capture on the second pass and classify_video flips from PAL to ?.
    """
    p = tmp_path / "cap.cf32"
    iq = np.arange(16, dtype=np.complex64)
    write_capture(p, iq, 5800e6, 10e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        a = src.read(16)
        a[:] = 0
        src._pos = 0
        b = src.read(16)
        np.testing.assert_array_equal(b, iq)
        assert not np.allclose(b, 0)
    finally:
        src.close()
