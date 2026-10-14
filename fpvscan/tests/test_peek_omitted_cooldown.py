"""Omitted auto_peek_cooldown_s must still be a 60 s duration."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_omitted_auto_peek_cooldown_is_60s():
    """#53 used an explicit 60 s so 59 skipped and 60 allowed.
    helpers.py always passes 60. Dropping the key must keep that
    window. A leftover 0 re-peeks every sweep pass (LOCK tears the
    clip and the rest of the air waits). A leftover milliseconds
    stamp never peeks the same VTx again.
    """
    eng = make_engine()
    eng.cfg["scan"].pop("auto_peek_cooldown_s")
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
