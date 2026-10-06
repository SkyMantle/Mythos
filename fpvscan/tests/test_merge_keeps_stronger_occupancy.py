"""A weaker re-inspect must not move the occupancy we already trusted."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_merge_keeps_stronger_occupancy_on_weaker_second_hit():
    """#9 kept the stronger *second* hit. The inverse is the usual
    case: the first inspect is on-channel, the next sweep step sees
    the same VTx 3 MHz over and 11 dB down. `_merge` copies the old
    freq/bw/snr onto the new Detection ('тримаємось кращої оцінки').
    Dropping that copy lets `_lock_bw` / click-to-hold follow the
    weaker centre, and LOCK channelizes into the neighbor.
    """
    eng = make_engine()
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=22.0,
        standard="PAL", confidence=0.9, hits=1))
    eng._merge(Detection(
        freq_hz=5803e6, bandwidth_hz=7e6, snr_db=11.0,
        standard="?", confidence=0.4, hits=1))
    dets = list(eng.state.detections.values())
    assert len(dets) == 1
    d = dets[0]
    assert d.freq_hz == 5800e6
    assert d.bandwidth_hz == 12e6
    assert d.snr_db == 22.0
    assert d.hits == 2
    assert list(eng.state.detections) == [5800]
