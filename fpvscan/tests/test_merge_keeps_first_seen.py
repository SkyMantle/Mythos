"""A re-hit must keep the original first_seen, not jump to now."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_merge_keeps_first_seen_from_the_older_hit():
    """#9 / #52 kept the stronger occupancy (freq/bw/snr). first_seen
    is the other half of `_merge`: the console ages a row off that
    stamp. Overwriting it on every overlapping sweep step makes a
    live VTx look brand-new each pass and the operator cannot tell a
    persistent bird from a one-shot spur.
    """
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = False
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12.0,
        first_seen=100.0, last_seen=100.0, hits=1,
    )
    with patch("fpvscan.engine.time.time", return_value=250.0):
        eng._merge(Detection(
            freq_hz=5800.4e6, bandwidth_hz=9e6, snr_db=10.0,
            first_seen=250.0, last_seen=250.0,
        ))
    kept = next(iter(eng.state.detections.values()))
    assert kept.first_seen == 100.0
    assert kept.hits == 2
    assert kept.freq_hz == 5800e6
    assert kept.snr_db == 12.0
