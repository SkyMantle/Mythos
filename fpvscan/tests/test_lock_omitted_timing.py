"""Omitted LOCK timing keys must keep a field-length grab and a 3× ring."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _SpyRing:
    def __init__(self):
        self.asked = None

    def snapshot(self, n: int):
        self.asked = int(n)
        return np.zeros(int(n), dtype=np.complex64), 0


def test_do_lock_capture_defaults_to_120_ms_when_omitted():
    """#35 / #53 locked the 20 ms decode floor and a 50 ms sim.
    Engine's fallback is 120 ms — shipped YAML is 80. Dropping the
    key and then defaulting to 0 asks the ring for nothing and LOCK
    never emits a frame. helpers.py uses 5 ms so this pops it.
    """
    ring = _SpyRing()
    eng = make_engine(idle_ms=0, afc=False)
    eng.cfg["video"].pop("capture_ms")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = ring
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=None):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert ring.asked == int(20e6 * 0.120)


def test_do_lock_ring_defaults_to_max_of_300ms_and_3x_capture():
    """#26 primed 10 ms into the ring. The ring itself is
    `max(0.3, 3·capture)` when YAML omits `ring_seconds` (shipped
    is 0.5). A 0 s fallback leaves only the prime and every
    snapshot is short — LOCK idles forever looking unlocked.
    """
    seen: list[float] = []
    eng = make_engine(idle_ms=0, capture_ms=80, afc=False)
    eng.cfg["video"].pop("ring_seconds")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6

    def spy(want, fs, ring_seconds):
        seen.append(float(ring_seconds))
        eng._ring = _SpyRing()
        eng._lock_tuned = want

    eng._start_reader = spy
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=None):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert seen == [0.3]
