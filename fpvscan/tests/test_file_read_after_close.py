"""File replay must not keep serving IQ after close()."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvscan.sdr.file import FileSource, write_capture


def test_read_after_close_raises(tmp_path: Path):
    """Engine `_run` finally closes the source. A leftover LOCK
    reader that still `read()`s after that would either crash on
    None or — if close() only dropped the path — keep walking a
    stale mmap and look like a live VTx.
    """
    p = tmp_path / "vtx.cf32"
    write_capture(p, np.ones(64, dtype=np.complex64), 5800e6, 40e6)
    src = FileSource(p, loop=True)
    src.open()
    assert src.read(8).shape == (8,)
    src.close()
    with pytest.raises(IOError, match="не відкрите"):
        src.read(8)
