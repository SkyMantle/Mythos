"""LOCK waterfall must stay on the physical LO, not the digital AFC shift."""
from __future__ import annotations

from queue import Queue

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_spectrum_center_is_tune_plus_lo_offset_not_afc():
    """AFC is a digital mixer offset. If the spectrum event used f+_afc
    the ribbon would jump every correction while the radio stayed put,
    and the operator would nudge the wrong way.
    """
    fs = 20e6
    off = 2e6
    events = Queue()
    src = MockSource(fs=fs)
    # spectrum_every=1 never emits: 1 % 1 == 0. Shipped/default 8
    # fires on the first LOCK because _lock_n becomes 1.
    eng = make_engine(src=src, events=events, capture_ms=8, idle_ms=0,
                      afc=False, lo_offset_hz=off, sample_rate=fs,
                      spectrum_every=8)
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6 + off
    eng._afc = 1.5e6
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()

    specs = []
    while not events.empty():
        ev = events.get_nowait()
        if ev["type"] == "spectrum":
            specs.append(ev["data"])
    assert specs, "first LOCK must emit a spectrum"
    assert specs[0]["center_hz"] == 5800e6 + off
    assert specs[0]["center_hz"] != 5800e6 + off + eng._afc
    assert specs[0]["span_hz"] == fs
