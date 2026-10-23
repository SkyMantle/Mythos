"""Auto-peek must not reset the LOCK waterfall cadence."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_keeps_lock_n():
    """#58 omitted `spectrum_every=8` still emits on `_lock_n`
    8→9 (`9%8==1`); #66 LOCK FFT is hardcoded 4096×4. Sweep /
    lock / Clear leftover `_lock_n` was deferred — `_maybe_peek`
    is the shipped hop onto LOCK (`auto_peek: true`).
    Resetting the counter when a confirm glances would make the
    first field `1%8==1` and fire a 4096×4 waterfall on every
    bird, blowing Pi idle_ms. Leftover cadence is current main
    (a hold that stopped on `_lock_n%8!=0` can skip one paint).
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._lock_n = 7
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.auto is True
    assert eng._lock_n == 7
