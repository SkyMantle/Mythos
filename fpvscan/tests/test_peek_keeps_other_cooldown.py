"""Auto-peek must not wipe cooldown for the other analog hits."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_keeps_other_mhz_cooldown():
    """#64 Sweep / lock keep `_peeked`. #68 expiry keeps the table
    so the bird we just watched is not re-peeked. `_maybe_peek`
    writes the current MHz; leftover `_peeked.clear()` before that
    write would forget every other confirmed bird and glance at
    each of them again on the next confirm_hits cycle instead of
    waiting `auto_peek_cooldown_s`.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._peeked[5725] = 1.0
    eng._peeked[1280] = 2.0
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._peeked[5725] == 1.0
    assert eng._peeked[1280] == 2.0
    assert 5800 in eng._peeked
