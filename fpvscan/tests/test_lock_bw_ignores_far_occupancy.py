"""_lock_bw must not borrow a neighbor's occupancy past MERGE_TOL."""
from __future__ import annotations

from fpvscan.engine import Detection, Engine
from tests.helpers import make_engine


def test_occupancy_beyond_merge_tol_does_not_set_lock_bw():
    """#33 only grepped the `6e6` window. #18 used a co-located
    detection. A Band-F neighbor is 20 MHz away; if the lookup widens
    to that spacing, LOCK channelizes two VTx into one picture. This
    is the window contract, not the +2·MERGE_TOL pad (#12).
    """
    eng = make_engine()
    default = 10e6
    eng.state.detections[5807] = Detection(
        freq_hz=5800e6 + 7e6, bandwidth_hz=20e6, snr_db=18.0,
    )
    assert 7e6 > Engine.MERGE_TOL_HZ
    assert eng._lock_bw(5800e6, default) == default
    assert eng._lock_bw(5800e6, default) != 20e6
    assert eng._lock_bw(5800e6, 3e6) == 8e6
