"""«Сканувати» must leave the confirmed hit list alone."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_sweep_keeps_detections_and_drops_lock():
    """Clear wipes detections. Sweep must not. Operators hit Сканувати
    to resume the 400 MHz–6 GHz pass after a peek; if that also
    clears the list they lose the only Hz they can click back onto,
    and the next confirm_hits=2 cycle looks like the VTx vanished.
    """
    eng = make_engine()
    det = Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14,
                    standard="NTSC", confidence=0.9, channel="F4",
                    band="5G8", hits=2)
    eng.state.detections[5800] = det
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng.command("sweep")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert 5800 in eng.state.detections
    assert eng.state.detections[5800].freq_hz == 5800e6
    assert eng.state.detections[5800].channel == "F4"
