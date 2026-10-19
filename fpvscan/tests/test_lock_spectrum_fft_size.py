"""LOCK waterfall must stay on a cheap 4096×4 PSD, not the sweep FFT."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_spectrum_is_4096_by_4_not_scan_fft():
    """#58 Sweep omitted `fft_size=4096` / `averages=8` (`need=32768`).
    #18 LOCK spectrum center is f+lo_offset, span is raw fs. The
    waterfall itself is a hardcoded `psd_db(iq, 4096, 4)` plus
    `downsample_for_display(..., 384)` (#33 canvas width). Wiring
    shipped scan knobs (8192×8) into every LOCK field blows the
    Pi 5 idle_ms budget and the waterfall stalls while the picture
    still ticks. A leftover 8192 here is not a sharper ribbon —
    it is a dropped frame.
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    eng.cfg["scan"]["fft_size"] = 8192
    eng.cfg["scan"]["averages"] = 8
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    eng._lock_n = 0
    try:
        with patch("fpvscan.engine.spectrum.psd_db",
                   return_value=np.zeros(4096, dtype=np.float32)) as psd, \
             patch("fpvscan.engine.spectrum.downsample_for_display",
                   return_value=[0.0] * 8) as down, \
             patch("fpvscan.engine.spectrum.noise_floor_db",
                   return_value=-90.0), \
             patch("fpvscan.engine.cvbs.decode", return_value=None):
            eng._do_lock()
        assert psd.called, "first LOCK must emit a spectrum"
        args, kwargs = psd.call_args
        assert args[1] == 4096
        assert args[2] == 4
        assert kwargs == {}
        assert down.call_args.args[1] == 384
    finally:
        eng._stop_reader()
