"""Console «Очистити» must wipe hits, not tear down an active LOCK."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_clear_keeps_lock_target_and_mode():
    """clear() resets detections / peek memory / sweep index. If it
    also flips mode to SWEEP or nulls lock_target, the operator
    hitting Очистити to drop a stale Wi-Fi hit also kills the
    picture and the recorder (lock/sweep stop the clip). That looks
    like the click-to-hold bug, but from the other button.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14)
    eng._peeked[5800] = 1.0
    eng._sweep_i = 4
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.detections == {}
    assert eng._peeked == {}
    assert eng._sweep_i == 0
