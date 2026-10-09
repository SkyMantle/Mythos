"""Temporal average must freeze the background and pass motion through."""
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
    line_rate: float = 15625.0
    lines: int = 8
    standard: str = "PAL"
    locked: bool = True


def test_motion_adaptive_average_passes_a_moving_edge():
    """Plain EMA (`a = 1/average` everywhere) smears a 100→200 jump to
    ~133 and doubles interlaced text. #4 weights only static pixels:
    `w = min(|cur-acc|/thresh, 1)`, so a 100-level jump at thresh=24
    is `a=1` (keep current) while the unchanged half stays at 100.
    Reverting to uniform EMA is the motion-smear regression.
    """
    h, w = 8, 8
    static = np.full((h, w), 100, dtype=np.uint8)
    moving = static.copy()
    moving[:, w // 2:] = 200
    frames = [_Field(luma=static.copy()), _Field(luma=moving.copy())]

    def _decode(*_a, **_k):
        return frames.pop(0)

    eng = make_engine(average=3, motion_thresh=24.0, min_lines=8,
                      idle_ms=0, capture_ms=5, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", side_effect=_decode):
            eng._do_lock()
            eng._do_lock()
    finally:
        eng._stop_reader()

    acc = eng._acc
    assert acc is not None
    left = float(acc[:, : w // 2].mean())
    right = float(acc[:, w // 2:].mean())
    assert abs(left - 100.0) < 1.0
    assert abs(right - 200.0) < 1.0
    # Uniform EMA at average=3 would land the moving half near 133.
    assert abs(right - (100.0 * 2 / 3 + 200.0 / 3)) > 40.0
