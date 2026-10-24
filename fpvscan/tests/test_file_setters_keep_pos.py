"""File replay must not rewind the cursor when radio knobs are poked."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_setters_do_not_rewind_read_cursor(tmp_path: Path):
    """#66 `set_sample_rate` / #68 `set_center_freq` / #69 `set_gain`
    are no-ops for fs / LO / amplitude. A leftover radio-style
    flush (`_pos = 0` on any setter) would restart the .cf32 from
    sample 0. `_run` calls set_sample_rate then set_gain after
    open(); LOCK calls set_sample_rate when |Δfs|>1. Rewinding
    there splices the capture head onto a mid-file reader chunk,
    genlock rolls, and classify of the next inspect grab sees the
    start of the file twice instead of the 16.2 kHz line that was
    about to arrive. Replay must keep the cursor.
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
        src.set_center_freq(1280e6)
        src.set_sample_rate(35e6)
        src.set_gain(0.0)
        mid = src.read(16)
        np.testing.assert_array_equal(mid, iq[16:32])
        assert not np.array_equal(mid, iq[:16])
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
    finally:
        src.close()
