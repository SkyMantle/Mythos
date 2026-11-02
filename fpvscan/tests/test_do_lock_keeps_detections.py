"""A LOCK field must still see occupancy so `_lock_bw` does not fall to YAML."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_do_lock_keeps_leftover_detections():
    """#66 lock / #68 expiry / #70 `_maybe_peek` keep
    `detections` so `_lock_bw` still sees the measured
    occupancy. #76 `_start_reader` keeps the list on a
    channel change; #76 `_do_sweep` keeps it on an 8 MHz
    hop. Clear is the wipe. `_do_lock` is the missing half:
    a field on the same bird must not `detections.clear()`.
    A leftover wipe on every decode drops occupancy so the
    next field (and `_lock_bw`) falls to the YAML floor —
    adjacent Raceband video mixes into the picture, and the
    heartbeat looks like the VTx vanished.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    leftover = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=18.0,
        standard="NTSC", confidence=0.80, channel="F4", band="5G8",
        first_seen=11.0, last_seen=22.0, hits=2,
    )
    eng.state.detections[5800] = leftover
    try:
        eng._do_lock()
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
