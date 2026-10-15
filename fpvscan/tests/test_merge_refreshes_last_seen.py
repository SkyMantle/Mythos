"""A re-hit must refresh last_seen so a live VTx does not look stale."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_merge_refreshes_last_seen_on_a_rehit():
    """#61 kept first_seen from the older occupancy. last_seen is the
    other half: applyState ages the hit list off that stamp. Copying
    the old last_seen onto the new Detection makes a bird that is
    still in the air look like it vanished at first_seen, and the
    operator treats a live analog channel as a one-shot spur.
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
    assert kept.last_seen == 250.0
    assert kept.first_seen == 100.0
    assert kept.hits == 2
