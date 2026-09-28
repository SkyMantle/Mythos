"""LOCK channelize is the cheap boxcar; FIR is the inspect-only path."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_channelize_uses_the_fast_boxcar():
    """INSPECT passes fast=False so a neighbor VTx does not leak a
    line-rate. LOCK copies that FIR onto every 80 ms grab and the
    Pi 5 misses the idle_ms budget — the console looks frozen while
    the radio is fine. Omitting the kwarg keeps channelize's
    default boxcar (fast=True).
    """
    fs = 20e6
    src = MockSource(fs=fs)
    eng = make_engine(src=src, capture_ms=8, idle_ms=0, afc=False,
                      sample_rate=fs)
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6

    seen: list[bool] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.demod.channelize

    def wrap(iq, rate, offset_hz, out_bw_hz, fast=True):
        seen.append(bool(fast))
        return real(iq, rate, offset_hz, out_bw_hz, fast=fast)

    eng_mod.demod.channelize = wrap
    try:
        eng._do_lock()
    finally:
        eng_mod.demod.channelize = real
        eng._stop_reader()

    assert seen, "LOCK never channelized"
    assert seen[0] is True
