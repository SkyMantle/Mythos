"""Auto-peek is a glance, not a rewind of the sweep plan."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_keeps_sweep_index():
    """#64 lock() keeps `_sweep_i`; #52 expiry keeps it; Clear is
    the rewind. `_maybe_peek` is the shipped hop onto LOCK.
    Resetting the plan index when a confirm glances at a bird
    restarts the 400 MHz–6 GHz walk after the 8 s timer — a hop-on
    VTx the sweep already waited past is missed for a full pass.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._sweep_i = 17
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._sweep_i == 17
