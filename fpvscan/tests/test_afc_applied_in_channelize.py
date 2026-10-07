"""Digital AFC must mix in the channelizer, not only skip a physical hop."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_channelize_offset_is_minus_lo_plus_digital_afc():
    """#18 checked mixer sign with AFC off (`−off`). #52 checked that
    400 kHz of leftover `_afc` does not `_start_reader`. Neither proves
    the correction actually reaches `channelize`. Dropping `+ self._afc`
    from the mixer leaves LOCK on the physical LO while the VTx drifts
    — snow on a confirmed analog channel, no extra retune to blame.
    """
    fs = 40e6
    off = 8e6
    afc = 400e3
    src = MockSource(fs=fs)
    eng = make_engine(
        src=src, lo_offset_hz=off, channel_bw_hz=10e6, sample_rate=fs,
        capture_ms=8, idle_ms=0, afc=False,
    )
    eng.cfg["scan"]["sample_rate"] = fs
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6 + off
    eng._afc = afc

    seen: list[float] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.demod.channelize

    def wrap(iq, rate, offset_hz, out_bw_hz, fast=True):
        seen.append(float(offset_hz))
        return real(iq, rate, offset_hz, out_bw_hz, fast=fast)

    eng_mod.demod.channelize = wrap
    try:
        eng._do_lock()
    finally:
        eng_mod.demod.channelize = real
        eng._stop_reader()

    assert seen, "channelize never ran"
    assert seen[0] == -off + afc
    assert src.retune_calls == 0
    assert eng._lock_tuned == 5800e6 + off
