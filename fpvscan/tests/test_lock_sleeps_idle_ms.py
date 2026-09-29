"""LOCK must yield idle_ms even when the ring is short or a grab ran."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


class _ShortRing:
    def snapshot(self, n: int):
        return np.zeros(8, dtype=np.complex64), 0


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_short_ring_sleeps_idle_ms_instead_of_spinning(monkeypatch):
    """Until the USB reader fills capture_ms, snapshot is short.
    Returning without sleep busy-loops the engine thread, starves
    the reader, and USB overflow looks like a dead VTx. idle_ms is
    the yield; helpers default it to 0 so other tests stay fast.
    """
    slept: list[float] = []
    monkeypatch.setattr("fpvscan.engine.time.sleep",
                        lambda s: slept.append(float(s)))
    eng = make_engine(src=MockSource(), capture_ms=8, idle_ms=40, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _ShortRing()
    eng._lock_tuned = 5800e6
    eng._lock_n = 0
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert slept == [0.04]
    assert eng._lock_n == 0


def test_lock_sleeps_idle_ms_after_a_grab(monkeypatch):
    """Shipped idle_ms=10 is the Pi 5 duty cycle. Dropping the sleep
    after a successful (or empty) decode pegs Cortex-A76, the USB
    reader misses sync_rx, and overflows increment while the console
    still shows LOCK.
    """
    slept: list[float] = []
    monkeypatch.setattr("fpvscan.engine.time.sleep",
                        lambda s: slept.append(float(s)))
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *a, **k: None)
    eng = make_engine(src=MockSource(), capture_ms=5, idle_ms=40, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert slept == [0.04]
