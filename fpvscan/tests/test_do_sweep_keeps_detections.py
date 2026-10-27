"""A sweep hop must not wipe the hit list — confirm_hits lives across steps."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_detections():
    """#39 Sweep *command* keeps `detections` (Clear is the wipe).
    #68 `_do_sweep` nulls `_lock_tuned` on every hop so a
    leftover hold cannot skip `_start_reader`. The hop must
    not also `detections.clear()`. A leftover wipe on each
    8 MHz step resets `hits` so `confirm_hits=2` never
    lights the bird — or a second inspect of the same VTx
    looks new and `_lock_bw` falls to the YAML floor. The
    command/hop split is the same as tune: Сканувати keeps
    the list; only Очистити drops it.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    leftover = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=18.0,
        standard="NTSC", confidence=0.80, channel="F4", band="5G8",
        first_seen=11.0, last_seen=22.0, hits=2,
    )
    eng.state.detections[5800] = leftover
    try:
        eng._do_sweep()
        assert 5800 in eng.state.detections
        kept = eng.state.detections[5800]
        assert kept is leftover
        assert kept.hits == 2
        assert kept.freq_hz == 5800e6
        assert kept.snr_db == 18.0
        assert kept.first_seen == 11.0
        assert eng._lock_tuned is None
    finally:
        eng._stop_reader()
