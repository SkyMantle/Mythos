"""Reopening a .cf32 must resume the cursor, not rewind the tape."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_close_open_keeps_cursor_and_recorded_lo(tmp_path: Path):
    """#72 setters keep `_pos`; #73 `retune_and_read` keeps it
    across an inspect hop. Handle lifetime is the other half:
    `close()` only drops `_data`, `open()` reloads the same file
    and never writes `_pos = 0`. A leftover rewind on reopen
    splices the capture head onto a mid-file chunk — classify
    sees the same 16.2 kHz line twice and `confirm_hits=2`
    lights a ghost, or LOCK rolls genlock against a leftover
    DecodeState. Engine `_run` opens once, but bringup /
    selftest / a second `with FileSource()` on a reused
    instance would restart the tape. Recorded LO must also
    stay (sidecar 5800, not a later hop).
    """
    p = tmp_path / "vtx.cf32"
    iq = (np.arange(64, dtype=np.float32)
          + 1j * np.linspace(-1.0, 1.0, 64)).astype(np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=40e6,
                  gain_db=30.0)
    src = FileSource(p, loop=True)
    src.open()
    try:
        head = src.read(16)
        np.testing.assert_array_equal(head, iq[:16])
        assert src._pos == 16
        src.close()
        assert src._pos == 16
        src.open()
        assert src._pos == 16
        mid = src.read(16)
        np.testing.assert_array_equal(mid, iq[16:32])
        assert not np.array_equal(mid, iq[:16])
        assert src._pos == 32
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
    finally:
        src.close()
