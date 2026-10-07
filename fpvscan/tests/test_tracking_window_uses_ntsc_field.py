"""NTSC LOCK tracking must size the ring grab for 262.5-line fields."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.cvbs import FIELD_LINES, DecodeState
from tests.helpers import MockSource, make_engine


class _RecRing:
    def __init__(self):
        self.n = None

    def snapshot(self, n: int):
        self.n = int(n)
        return np.zeros(int(n), dtype=np.complex64), 0


def test_tracking_window_uses_ntsc_field_lines_not_pal():
    """#46 proved `* _lock_dec * 1.7` on a PAL state. Hardcoding
    FIELD_LINES['PAL'] (312.5) for every lock asks NTSC for an extra
    50-line tail of the *next* field — V-sync of the following picture
    lands in the grab, genlock tears, and the console looks like the
    NTSC-as-PAL bug even when classify was right. '?' (287.5) is also
    wrong here.
    """
    fs = 40e6
    dec = 5
    ch_bw = 8e6
    assert max(1, int(fs / ch_bw)) == dec
    period = (fs / dec) / 15734.264
    margin = 1.7
    n_ntsc = int(period * FIELD_LINES["NTSC"] * dec * margin)
    n_pal = int(period * FIELD_LINES["PAL"] * dec * margin)
    n_unk = int(period * FIELD_LINES["?"] * dec * margin)
    n_full = int(fs * 0.120)
    n_floor = int(fs * 0.02)
    assert n_floor < n_ntsc < n_full
    assert n_ntsc != n_pal
    assert n_ntsc != n_unk

    src = MockSource(fs=fs)
    eng = make_engine(
        src=src, capture_ms=120, idle_ms=0, afc=False,
        channel_bw_hz=ch_bw, sample_rate=fs,
    )
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    ring = _RecRing()
    eng._ring = ring
    eng._lock_tuned = 5800e6
    eng._lock_dec = dec
    eng._lock_state = DecodeState(period=period, standard="NTSC", lost=0)
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()

    assert ring.n == n_ntsc
    assert ring.n != n_pal
    assert ring.n != n_unk
    assert ring.n != n_full
    assert ring.n != n_floor
