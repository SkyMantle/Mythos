"""Occupancy inside MERGE_TOL is the LOCK channelizer width, not YAML default."""
from __future__ import annotations

from fpvscan.engine import Detection, Engine
from tests.helpers import make_engine


def test_occupancy_just_inside_merge_tol_sets_lock_bw():
    """#18 used a co-located detection. #53 ignored one 7 MHz away
    (beyond MERGE_TOL). The window is `< 6e6`, so a 5.9 MHz center
    error — typical of a 10 MHz analog blob on a 35 Msps hop — must
    still pick up the measured occupancy. Falling back to the YAML
    8–10 MHz default either clips the discriminator or, with the
    +2·MERGE_TOL pad (#12), is the wrong failure mode to pin here:
    we only require the nearby occupancy to win over the default.
    """
    eng = make_engine()
    default = 8e6
    occ_bw = 12e6
    eng.state.detections[5806] = Detection(
        freq_hz=5800e6 + 5.9e6, bandwidth_hz=occ_bw, snr_db=16.0,
    )
    assert 5.9e6 < Engine.MERGE_TOL_HZ
    got = eng._lock_bw(5800e6, default)
    assert got != default
    assert got >= occ_bw
