"""Digital AFC must stay inside the analog LPF, not the YAML 20 MHz cap."""
from __future__ import annotations

import inspect

import numpy as np

from fpvscan.engine import Engine
from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_afc_yaml_limit_is_clamped_to_nyquist_headroom():
    """Shipped `afc_limit_hz` is 20 MHz. At 35 Msps with a 10 MHz
    channel (dec>1) the mixer + half the window must stay inside
    0.45·fs. Without the clamp, AFC walks past the RFIC LPF and the
    picture folds onto the opposite skirt.
    """
    fs = 35e6
    ch_bw = 10e6
    src = MockSource(fs=fs)
    eng = make_engine(
        src=src, capture_ms=8, idle_ms=0, afc=True,
        channel_bw_hz=ch_bw, sample_rate=fs,
        afc_limit_hz=20e6, afc_deadband_hz=1.0,
        afc_gain=1.0, afc_recenter_frac=1.1,
        lo_offset_hz=0.0,
    )
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6

    import fpvscan.engine as eng_mod
    real = eng_mod.demod.freq_error_hz

    def fat_offset(iq, rate):
        return 15e6

    eng_mod.demod.freq_error_hz = fat_offset
    try:
        eng._do_lock()
    finally:
        eng_mod.demod.freq_error_hz = real
        eng._stop_reader()

    safe = max(0.0, fs * 0.45 - ch_bw / 2)
    assert 0 < eng._afc <= safe + 1.0
    assert eng._afc < 20e6
    assert abs(eng._afc - safe) < 1.0


def test_do_lock_computes_a_nyquist_safe_afc_limit():
    src = inspect.getsource(Engine._do_lock)
    assert "fs * 0.45" in src
    assert "ch_bw / 2" in src
    assert "safe_lim" in src
