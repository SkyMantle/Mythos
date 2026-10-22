"""Auto-peek must not rewind the console pass counter."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_keeps_sweeps_done():
    """#66 lock() and #68 expiry keep `sweeps_done`. #65 Sweep /
    Clear keep it so Сканувати / Очистити do not look like boot.
    `_maybe_peek` is the other hop onto LOCK. Zeroing the console
    pass counter when a confirm glances at a bird after a finished
    400 MHz–6 GHz walk makes pass 3 look like boot, and the
    operator waits for a cycle that already finished.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng.state.sweeps_done = 3
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.sweeps_done == 3
    assert eng.snapshot()["sweeps_done"] == 3
