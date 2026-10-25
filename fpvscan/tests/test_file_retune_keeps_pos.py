"""Inspect hops must walk the .cf32, not rewind or PLL-dump it."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_retune_and_read_keeps_cursor_and_recorded_lo(tmp_path: Path):
    """#17 file retune is not the base-class `read(n//4)` PLL dump;
    #72 setters keep `_pos` when radio knobs are poked. The hop
    path is the other half: `_inspect` and `_start_reader` both
    go through `retune_and_read`. A leftover `_pos = 0` at the
    start of every hop restarts the capture, so every occupancy
    classifies the same head and `confirm_hits=2` lights a ghost
    (or never sees the 16.2 kHz line that was about to arrive).
    The base-class dump would skip a quarter of the file per
    inspect and splice the next bird onto the previous tail.
    `retune_and_read` must not call `set_center_freq` either —
    #68 is a no-op today, but a later honoring write would
    rewrite the recorded LO and mis-bin occupancy.
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
        hopped = src.retune_and_read(1280e6, 16)
        np.testing.assert_array_equal(hopped, iq[16:32])
        dumped = iq[16 + 16 // 4:16 + 16 // 4 + 16]
        assert not np.array_equal(hopped, iq[:16])
        assert not np.array_equal(hopped, dumped)
        assert src._pos == 32
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
        third = src.read(16)
        np.testing.assert_array_equal(third, iq[32:48])
    finally:
        src.close()
