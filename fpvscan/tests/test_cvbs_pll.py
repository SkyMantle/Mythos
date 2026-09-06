"""Tracking PLL must refine the line period only after several fields."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.dsp.cvbs import DecodeState, FIELD_LINES
from fpvscan.dsp import demod


def _pulse_train(n: int, t0: float, period: float, pulse: int = 8) -> np.ndarray:
    v = np.full(n, 0.70, dtype=np.float32)
    i = t0
    while i + pulse < n:
        s = int(i)
        v[s:s + pulse] = 0.0
        i += period
    return v


def test_period_pll_updates_after_four_fields():
    period = 200.0
    field = period * FIELD_LINES["PAL"]
    n_fields = 5
    abs_start = n_fields * field
    n = int(80 * period)
    # Real vsync is 40 samples later than the prediction (local_t0=0).
    v = _pulse_train(n, t0=40.0, period=period)
    state = DecodeState(sign=1.0, period=period, standard="PAL",
                        abs_t0=0.0, lost=0)
    tracked = cvbs._attempt_tracked(v, fs=4e6, width=16, max_lines=40,
                                    state=state, abs_start=abs_start)
    assert tracked is not None
    assert state.period is not None
    assert state.period > period


def test_period_pll_holds_when_fewer_than_four_fields():
    period = 200.0
    field = period * FIELD_LINES["PAL"]
    abs_start = 2 * field
    n = int(80 * period)
    v = _pulse_train(n, t0=40.0, period=period)
    state = DecodeState(sign=1.0, period=period, standard="PAL",
                        abs_t0=0.0, lost=0)
    tracked = cvbs._attempt_tracked(v, fs=4e6, width=16, max_lines=40,
                                    state=state, abs_start=abs_start)
    assert tracked is not None
    assert state.period == period


def test_classify_rejected_noise_explains_why():
    fs = 1e6
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, int(fs * 0.05)).astype(np.float32)
    score = demod.classify_video(noise, fs)
    assert not score.is_video
    assert score.reason
    assert "впевненість" in score.reason


def test_channelize_short_iq_skips_decimation():
    """Fewer samples than the decimation factor must not yield an empty band."""
    iq = np.ones(4, dtype=np.complex64)
    ch, fs2 = demod.channelize(iq, 20e6, 0.0, out_bw_hz=1e6)
    assert fs2 == 20e6
    assert len(ch) == 4
    assert ch.dtype == np.complex64
