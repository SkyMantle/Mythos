"""Click-to-hold must not wipe the occupancy that `_lock_bw` reads."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_lock_keeps_detections_for_lock_bw():
    """#39 Sweep keeps the hit list so Сканувати does not drop the
    only clickable Hz. Clear (#37) is the wipe. lock() already nulls
    `_lock_tuned` / AFC (#21) and cancels auto (#63). Clearing
    `detections` on a click-to-hold would also drop the measured
    occupancy `_lock_bw` uses: the hold would then decode at the YAML
    floor instead of the 5.8 bird that just confirmed, and the
    console heartbeat would look like the VTx vanished until the
    next confirm_hits cycle.
    """
    eng = make_engine()
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng.state.detections[5800] = det
    eng.state.mode = "SWEEP"
    eng.command("lock", freq_hz=5800e6)
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert 5800 in eng.state.detections
    assert eng.state.detections[5800] is det
    # Occupancy still wins over the YAML floor. Do not pin the
    # +2·MERGE_TOL pad (#12).
    assert eng._lock_bw(5800e6, 8e6) != 8e6
    snap = [d["channel"] for d in eng.snapshot()["detections"]]
    assert "F4" in snap
