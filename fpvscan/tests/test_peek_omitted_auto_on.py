"""Omitted auto_peek must still steal the sweep for a confident analog hit."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_omitted_auto_peek_is_on():
    """#29 pinned shipped YAML auto_peek true. helpers.py sets True
    explicitly; #36 did too. Dropping the key must still LOCK. A
    leftover False skips the picture on a confirmed analog channel
    and the operator only ever sees occupancy ticks.
    """
    eng = make_engine()
    eng.cfg["scan"].pop("auto_peek")
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=14.0, confidence=0.9,
        standard="PAL",
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
