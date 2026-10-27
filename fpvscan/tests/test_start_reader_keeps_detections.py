"""A LOCK hop must still see occupancy so `_lock_bw` does not fall to YAML."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_start_reader_keeps_leftover_detections():
    """#66 lock / #68 expiry / #70 `_maybe_peek` keep
    `detections` so `_lock_bw` still sees the measured
    occupancy. #74 `_start_reader` already keeps leftover
    `_lock_n`. The hop itself is the other half: it reseeds
    DecodeState after `retune_and_read` (#50) and must not
    `detections.clear()`. A leftover wipe on every channel
    change decodes at the YAML floor and the heartbeat
    looks like the VTx vanished — the same ghost as a
    click-to-hold that cleared the list.
    """
    eng = make_engine()
    leftover = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=18.0,
        standard="NTSC", confidence=0.80, channel="F4", band="5G8",
        first_seen=11.0, last_seen=22.0, hits=2,
    )
    eng.state.detections[5800] = leftover
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert 5800 in eng.state.detections
        kept = eng.state.detections[5800]
        assert kept is leftover
        assert kept.hits == 2
        assert kept.bandwidth_hz == 10e6
        assert eng._lock_bw(5800e6, 8e6) == 10e6 + 2 * eng.MERGE_TOL_HZ
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
