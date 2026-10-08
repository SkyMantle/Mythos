"""AFC residual is the leftover after the LO-offset mixer, not the raw ADC."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_afc_does_not_peg_to_the_physical_lo_offset():
    """#5 accumulated AFC on a 200 kHz tone with lo_offset=0 — measuring
    before channelize still works in that setup. With a fitting 8 MHz
    offset the VTx sits at −off in the ADC; freq_error before the mixer
    is ~−8 MHz and digital AFC slams to the limit / recenters every
    field. After the mixer the residual is ~0 and the reader stays put.
    """
    fs = 40e6
    off = 8e6
    src = MockSource(fs=fs, tone_hz=-off)
    eng = make_engine(
        src=src, lo_offset_hz=off, channel_bw_hz=10e6, sample_rate=fs,
        capture_ms=8, idle_ms=0, afc=True, afc_limit_hz=8e6,
        afc_deadband_hz=50e3, afc_gain=0.5, afc_recenter_frac=0.75,
    )
    eng.cfg["scan"]["sample_rate"] = fs
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        for _ in range(4):
            eng._do_lock()
        assert src.retune_calls == 1, (
            f"AFC measured on the pre-mixer ADC would peg to −{off/1e6:.0f} MHz "
            f"and recenter the reader (retune_calls={src.retune_calls})"
        )
        assert abs(eng._afc) < 1e6, (
            f"digital AFC absorbed the LO offset: _afc={eng._afc:.0f} Hz"
        )
        assert eng._lock_tuned == 5800e6 + off
    finally:
        eng._stop_reader()


def test_afc_gain_takes_half_the_error_per_field(monkeypatch):
    """Shipped afc_gain=0.5. Applying the full freq_error in one step
    overshoots a free-running VTx and walks the mixer into the
    recenter trip, resetting DecodeState every few fields.
    """
    src = MockSource()
    eng = make_engine(
        src=src, afc=True, afc_gain=0.5, afc_deadband_hz=1.0,
        afc_limit_hz=8e6, idle_ms=0, capture_ms=8,
    )
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6

    monkeypatch.setattr("fpvscan.engine.demod.freq_error_hz", lambda *_a, **_k: 400e3)
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()

    assert eng._afc == 200e3, (
        f"afc_gain=0.5 must take half of 400 kHz, got _afc={eng._afc}"
    )
    assert src.retune_calls == 0
