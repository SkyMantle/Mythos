"""Clear must not force a LOCK reader restart."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_clear_keeps_lock_tuned():
    """#37 kept mode and lock_target so Очистити does not tear down
    the picture. `_lock_tuned` is the other half: `_do_lock` starts a
    new reader whenever it disagrees with want. Nulling it on Clear
    joins the IQ thread, dumps 10 ms of PLL, and reseeds DecodeState
    — the live field rolls even though the operator only wanted the
    hit list emptied. #48 is the occupancy-BW collapse after Clear;
    this is the reader/genlock side of the same button.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng._lock_tuned = 5800e6
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14.0, hits=2,
    )
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._lock_tuned == 5800e6
    assert eng.state.detections == {}
