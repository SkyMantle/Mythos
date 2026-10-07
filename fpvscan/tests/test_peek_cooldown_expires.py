"""Auto-peek cooldown must be a duration, not a one-shot latch."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_same_channel_can_peek_again_after_cooldown():
    """#31 only checked that 5800.2 and 5800.8 share a MHz bucket.
    `now - peeked[key] < cooldown_s` is the other half: if the
    comparison flips to `>` or the stored stamp is milliseconds, a
    live VTx is either peeked every sweep (LOCK tears the clip) or
    never peeked again after the first look.
    """
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = True
    eng.cfg["scan"]["auto_peek_min_conf"] = 0.0
    eng.cfg["scan"]["auto_peek_cooldown_s"] = 60
    eng.cfg["scan"]["auto_peek_secs"] = 5
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12.0, confidence=0.9,
    )

    with patch("fpvscan.engine.time.time", return_value=1_000.0):
        eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng._peeked[5800] == 1_000.0

    eng.state.mode = "SWEEP"
    eng.state.lock_target = None
    eng.state.auto = False

    with patch("fpvscan.engine.time.time", return_value=1_000.0 + 59.0):
        eng._maybe_peek(det)
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None

    with patch("fpvscan.engine.time.time", return_value=1_000.0 + 60.0):
        eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
