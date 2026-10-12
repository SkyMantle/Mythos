"""Omitted LOCK average / idle_ms must not blur the picture or spin the CPU."""
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
    lines: int = 232
    standard: str = "NTSC"
    locked: bool = True


def test_omitted_average_does_not_start_ema():
    """#18 pinned shipped YAML average=3. #55 exercised motion-adaptive
    mixing with an explicit k. The `.get` fallback is 0.0 — no EMA.
    Defaulting a partial config to 3 (or any leftover) smears motion
    and splits interlace on every live frame.
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    eng.cfg["video"].pop("average")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    field = _Field(luma=np.full((232, 64), 80, dtype=np.uint8))
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=field), \
             patch("fpvscan.engine.cvbs.encode", return_value=b"img"):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert eng._acc is None


def test_omitted_idle_ms_is_120(monkeypatch):
    """#18 pinned shipped YAML 10 ms vs engine 120. #43 slept an
    explicit 40 ms. Dropping the key and then defaulting to 0 busy-loops
    the engine thread, starves the USB reader, and overflows look like
    a dead VTx. helpers.py uses 0 so other tests stay fast — this pops
    it.
    """
    slept: list[float] = []
    monkeypatch.setattr("fpvscan.engine.time.sleep",
                        lambda s: slept.append(float(s)))
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *_a, **_k: None)
    eng = make_engine(capture_ms=5, afc=False)
    eng.cfg["video"].pop("idle_ms")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert slept == [0.12]
