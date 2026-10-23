"""Очистити must not reset the live LOCK waterfall cadence."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_clear_command_keeps_lock_n():
    """#63 kept `_lock_tuned`; #64 kept `_ring`; #69 kept
    `_lock_dec` / `_lock_state` so Очистити does not reseed
    genlock. `_lock_n` is the waterfall half of the same
    hold: `_do_lock` emits when `_lock_n % spectrum_every
    == 1` (#58). Zeroing it on Clear fires a 4096×4 FFT on
    the next field even though the operator only wanted the
    hit list emptied. #48 is the occupancy-BW collapse after
    Clear; this is the cadence side.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng._lock_tuned = 5800e6
    eng._lock_n = 7
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14.0, hits=2,
    )
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._lock_tuned == 5800e6
    assert eng._lock_n == 7
    assert eng.state.detections == {}
