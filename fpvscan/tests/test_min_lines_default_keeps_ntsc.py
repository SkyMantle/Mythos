"""Engine min_lines default must keep a one-field NTSC render."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


@dataclass
class _Field:
    luma: np.ndarray
    line_rate: float = 15734.0
    lines: int = 232
    standard: str = "NTSC"
    locked: bool = True


def _drain_frames(eng):
    out = []
    while True:
        try:
            ev = eng.events.get_nowait()
        except Exception:
            break
        if ev.get("type") == "frame":
            out.append(ev)
    return out


def test_default_min_lines_accepts_a_232_line_ntsc_field():
    """#4 lowered the gate from 250 to 200 because one-field NTSC is
    ~230–240 active lines. YAML omits `min_lines`, so the engine
    fallback is the contract. A stale 250 drops a valid NTSC field
    every frame — snow + vertical roll, and the Photo click never
    sees a frame (#52 only kept `_snap` on a 10-line reject).
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    del eng.cfg["video"]["min_lines"]
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    field = _Field(luma=np.full((232, 64), 80, dtype=np.uint8))
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=field):
            eng._do_lock()
        assert _drain_frames(eng), "232-line NTSC field was dropped"
    finally:
        eng._stop_reader()


def test_default_min_lines_still_drops_a_partial_decode():
    """180 lines is a torn capture, not a short standard. The 200
    floor must still reject it so auto-peek does not publish snow.
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    del eng.cfg["video"]["min_lines"]
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    field = _Field(luma=np.zeros((180, 64), dtype=np.uint8), lines=180)
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=field):
            eng._do_lock()
        assert _drain_frames(eng) == []
    finally:
        eng._stop_reader()
