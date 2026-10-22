"""Auto-peek must not wipe the occupancy that `_lock_bw` reads."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_keeps_detections_for_lock_bw():
    """#66 lock() keeps the hit list so a click-to-hold still has
    occupancy. #68 expiry keeps it; Clear (#37) is the wipe.
    `_maybe_peek` is the shipped path onto LOCK (`auto_peek: true`).
    Clearing `detections` when a confirm glances at the bird would
    drop the measured occupancy `_lock_bw` uses: the 5 s picture
    would decode at the YAML floor, and the heartbeat would look
    like the VTx vanished until the next confirm_hits cycle.
    """
    eng = make_engine()
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng.state.detections[5800] = det
    eng.state.mode = "SWEEP"
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
    assert 5800 in eng.state.detections
    assert eng.state.detections[5800] is det
    # Occupancy still wins over the YAML floor. Do not pin the
    # +2·MERGE_TOL pad (#12).
    assert eng._lock_bw(5800e6, 8e6) != 8e6
    snap = [d["channel"] for d in eng.snapshot()["detections"]]
    assert "F4" in snap
