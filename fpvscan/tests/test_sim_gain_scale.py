"""SimSource gain must keep the (gain-40)/40 voltage scale.

Selftest and every sim LOCK fixture were tuned against gain=40 → scale 1.
A 'fix' to 20*log10 (power) would drop default-scene SNR by 20 dB and
make classify_video reject the F4 emitter.
"""
from __future__ import annotations

import numpy as np

from fpvscan.sdr.sim import Emitter, SimSource


def _cw(gain_db: float) -> np.ndarray:
    src = SimSource(
        emitters=[Emitter(5800e6, -20.0, "cw", label="tone")],
        noise_db=-200.0,
        seed=1,
    )
    src.open()
    src.set_sample_rate(1e6)
    src.set_center_freq(5800e6)
    src.set_gain(gain_db)
    return src.read(4096)


def test_sim_gain_40_is_unity_and_zero_is_minus_20db_voltage():
    x40 = _cw(40.0)
    x0 = _cw(0.0)
    rms40 = float(np.sqrt(np.mean(np.abs(x40) ** 2)))
    rms0 = float(np.sqrt(np.mean(np.abs(x0) ** 2)))
    assert rms40 > 0
    # 10 ** ((40-0)/40) = 10. A 20*log10 scale would give 100.
    ratio = rms40 / rms0
    assert abs(ratio - 10.0) < 0.5
    assert abs(ratio - 100.0) > 50
