"""LOCK picture scale: FM deviation and deemphasis order."""
from __future__ import annotations

import numpy as np

from fpvscan.engine import Detection, Engine
from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_demod_deviation_is_measured_channel_over_four():
    """INSPECT classifies at occ.bw/5. LOCK pictures at ch_bw/4, and
    ch_bw is the measured occupancy plus merge pad — not YAML
    channel_bw_hz and not video.deviation_hz. Using the inspect scale
    here washes the sync tip into the picture; using the YAML 10 MHz
    guess clips a 20 MHz analog channel.
    """
    fs = 40e6
    src = MockSource(fs=fs)
    eng = make_engine(src=src, capture_ms=8, idle_ms=0, afc=False,
                      channel_bw_hz=10e6, sample_rate=fs)
    eng.cfg["video"]["sample_rate"] = fs
    eng.cfg["video"]["deviation_hz"] = 10e6
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=20e6, snr_db=12.0,
    )
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6

    seen: list[float] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.demod.fm_demod

    def wrap(iq, rate, deviation_hz=4e6):
        seen.append(float(deviation_hz))
        return real(iq, rate, deviation_hz)

    eng_mod.demod.fm_demod = wrap
    try:
        eng._do_lock()
    finally:
        eng_mod.demod.fm_demod = real
        eng._stop_reader()

    assert seen, "LOCK never FM-demodulated"
    ch_bw = eng._lock_bw(5800e6, 10e6)
    assert ch_bw == 20e6 + 2 * Engine.MERGE_TOL_HZ
    assert seen[0] == ch_bw / 4
    assert seen[0] != 20e6 / 5
    assert seen[0] != 10e6


def test_lock_deemphasizes_after_fm_demod():
    """Deemphasis before the discriminator smears instantaneous frequency
    and the CVBS decoder loses line sync. Order is demod then 0.5 µs.
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

    order: list[str] = []
    import fpvscan.engine as eng_mod

    real_fm = eng_mod.demod.fm_demod
    real_de = eng_mod.demod.deemphasis

    def wrap_fm(iq, rate, deviation_hz=4e6):
        order.append("fm")
        return real_fm(iq, rate, deviation_hz)

    def wrap_de(v, rate, tau=0.5e-6):
        order.append("de")
        return real_de(v, rate, tau)

    eng_mod.demod.fm_demod = wrap_fm
    eng_mod.demod.deemphasis = wrap_de
    try:
        eng._do_lock()
    finally:
        eng_mod.demod.fm_demod = real_fm
        eng_mod.demod.deemphasis = real_de
        eng._stop_reader()

    assert order[:2] == ["fm", "de"]
