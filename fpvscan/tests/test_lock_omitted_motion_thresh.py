"""Omitted motion_thresh must still treat a 12-level step as half motion."""
from __future__ import annotations

from dataclasses import dataclass

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


def test_omitted_motion_thresh_is_24(monkeypatch):
    """#55 pinned the algorithm with explicit thresh=24: a 100-level
    jump is full motion. helpers.py always passes 24. Dropping the key
    must keep 24 so a 12-level step is `w=0.5`. With average=3 that
    lands at 108, not 112. A leftover 0 floors to 1 (`max(1, thr)`),
    treats the step as full motion, and the static-background SNR
    gain from #4 never happens on a partial config.
    """
    h, w = 8, 8
    seeded = np.full((h, w), 100.0, dtype=np.float32)
    cur = np.full((h, w), 112, dtype=np.uint8)

    eng = make_engine(average=3, min_lines=8, idle_ms=0, capture_ms=5,
                      afc=False)
    eng.cfg["video"].pop("motion_thresh")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    eng._acc = seeded.copy()
    monkeypatch.setattr("fpvscan.engine.cvbs.decode",
                        lambda *_a, **_k: _Field(luma=cur.copy()))
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()

    acc = float(eng._acc.mean())
    # thresh=24, |Δ|=12 → w=0.5, a=1/3+0.5·2/3=2/3 → 100/3 + 112·2/3 = 108
    assert abs(acc - 108.0) < 0.6
    # leftover thr floors to 1 → w=1 → acc=112
    assert abs(acc - 112.0) > 2.0
