"""MERGE_TOL must glue a 5 MHz re-hit and still list a 7 MHz neighbor."""
from __future__ import annotations

from fpvscan.engine import Detection, Engine
from tests.helpers import make_engine


def test_merge_collapses_a_five_mhz_rehit():
    """A 5 MHz walk is inside MERGE_TOL_HZ (6 MHz) — typical of the
    same analog blob on the next overlapped hop. Leaving those as
    two rows makes the operator click-lock 5 MHz off the peak.
    """
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = False
    assert Engine.MERGE_TOL_HZ == 6e6
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=18.0, hits=1))
    eng._merge(Detection(
        freq_hz=5800e6 + 5e6, bandwidth_hz=11e6, snr_db=16.0, hits=1))
    assert len(eng.state.detections) == 1
    d = next(iter(eng.state.detections.values()))
    assert d.hits == 2
    assert d.freq_hz == 5800e6


def test_merge_keeps_a_seven_mhz_neighbor():
    """#53 ignored occupancy 7 MHz away for `_lock_bw` (`< 6e6`).
    `_merge` uses `<= MERGE_TOL_HZ`. 7 MHz is outside that window —
    two analog birds that close (F4 vs a 5807 MHz spur, or a tight
    pair) must stay two clickable Hz. A leftover 20 MHz MERGE_TOL
    (Band-F neighbor step) glues them into one row and lock() parks
    on the stronger centre, mixing both pictures.
    """
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = False
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=18.0, hits=1))
    eng._merge(Detection(
        freq_hz=5800e6 + 7e6, bandwidth_hz=11e6, snr_db=16.0, hits=1))
    assert len(eng.state.detections) == 2
    freqs = sorted(d.freq_hz for d in eng.state.detections.values())
    assert freqs == [5800e6, 5807e6]
    assert all(d.hits == 1 for d in eng.state.detections.values())
