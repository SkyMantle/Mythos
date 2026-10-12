"""Omitted auto_peek_secs must dwell long enough to show a picture."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_omitted_auto_peek_secs_is_eight():
    """#18 pinned shipped YAML 5 s vs the engine fallback 8. #53 used
    an explicit 5. Dropping the key and then defaulting to 0 expires
    the peek before `_do_lock` runs — the operator never sees the bird
    and Sweep resumes on a confirmed analog channel.
    """
    eng = make_engine()
    eng.cfg["scan"].pop("auto_peek_secs")
    eng.cfg["scan"]["auto_peek_min_conf"] = 0.0
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12.0, confidence=0.9,
    )
    with patch("fpvscan.engine.time.time", return_value=1_000.0):
        eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
    assert eng.state.auto_until == 1_008.0
