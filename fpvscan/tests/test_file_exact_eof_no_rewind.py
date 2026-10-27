"""An exact-fit read to EOF must deliver the tail, not rewind first."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvscan.sdr.file import FileSource, write_capture


def test_file_exact_eof_read_does_not_rewind(tmp_path: Path):
    """#9 tiles when the request is longer than the capture.
    #44 `loop=False` EOFError's past the end; `loop=True`
    rewinds only when `_pos + n > size`. The inclusive
    leftover is a grab that lands exactly on the last
    sample: `_pos + n == size` must return the tail and
    leave the cursor at EOF. A leftover `>=` rewinds
    *before* the copy, so classify sees the .cf32 head
    twice and `confirm_hits=2` lights a ghost — or LOCK
    rolls genlock against the capture start. The next
    looping read is what wraps; a one-shot then EOFError's.
    """
    p = tmp_path / "vtx.cf32"
    iq = (np.arange(64, dtype=np.float32)
          + 1j * np.linspace(-1.0, 1.0, 64)).astype(np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=40e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        head = src.read(32)
        np.testing.assert_array_equal(head, iq[:32])
        assert src._pos == 32
        tail = src.read(32)
        np.testing.assert_array_equal(tail, iq[32:64])
        assert not np.array_equal(tail, iq[:32])
        assert src._pos == 64
        wrapped = src.read(16)
        np.testing.assert_array_equal(wrapped, iq[:16])
        assert src._pos == 16
    finally:
        src.close()

    oneshot = FileSource(p, loop=False)
    oneshot.open()
    try:
        oneshot.read(64)
        assert oneshot._pos == 64
        with pytest.raises(EOFError):
            oneshot.read(1)
        assert oneshot._pos == 64
    finally:
        oneshot.close()
