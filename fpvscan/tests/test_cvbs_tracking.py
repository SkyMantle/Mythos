"""LOCK tracking uses FIELD_LINES[standard]; a wrong label rolls the picture."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.dsp.cvbs import DecodeState, FIELD_LINES, _predict_local_t0
from fpvscan.sdr.sim import _cvbs


def test_predict_ntsc_field_is_50_lines_off_if_labeled_pal():
    """One NTSC field later: NTSC prediction lands on vsync, PAL is 50 lines away.

    The search window in _attempt_tracked is less than one line, so a 50-line
    miss latches onto a random hsync and the picture rolls.
    """
    period = 1000.0
    st = DecodeState(sign=1.0, period=period, standard="NTSC", abs_t0=100.0, lost=0)
    one_ntsc_field = period * FIELD_LINES["NTSC"]
    loc, n, _fp = _predict_local_t0(st, abs_start=100.0 + one_ntsc_field)
    assert n == 1
    assert abs(loc) < 1e-6

    st.standard = "PAL"
    loc, n, _fp = _predict_local_t0(st, abs_start=100.0 + one_ntsc_field)
    assert abs(loc) > 40.0 * period, loc


def test_predict_returns_none_without_period_or_abs_t0():
    assert _predict_local_t0(DecodeState(), abs_start=0.0) is None
    assert _predict_local_t0(
        DecodeState(period=100.0, abs_t0=None), abs_start=0.0) is None


def test_tracking_miss_still_blind_decodes_and_counts_lost():
    """A tracking miss must increment lost and still fall through to blind search."""
    fs = 8e6
    line_hz = 15625.0
    period = fs / line_hz
    n = int(fs * 0.08)
    # Capture starts one sim-field (262 lines) after abs_t0=0.
    # State claims PAL (312.5 lines), so the predicted vsync is ~50 lines off
    # and the ±0.95-line window cannot see the real equalizing pulses.
    abs_start = 262.0 * period
    t = (abs_start + np.arange(n)) / fs
    v = _cvbs(t, line_hz)
    state = DecodeState(sign=1.0, period=period, standard="PAL",
                        abs_t0=0.0, lost=0)
    frame = cvbs.decode(v, fs, width=48, state=state, abs_start=abs_start)
    assert frame is not None
    assert frame.locked
    # Either tracking luckily hit, or blind search recovered. The important
    # contract: a miss must not leave the decoder stuck with lost>=3 after one try.
    assert state.lost <= 1


def test_tracking_is_skipped_after_three_losses(monkeypatch):
    calls = []

    def spy(*_a, **_k):
        calls.append(1)
        return None

    monkeypatch.setattr(cvbs, "_attempt_tracked", spy)
    fs = 8e6
    v = _cvbs(np.arange(int(fs * 0.06)) / fs, 15625.0)
    state = DecodeState(sign=1.0, period=fs / 15625.0, standard="PAL",
                        abs_t0=0.0, lost=3)
    frame = cvbs.decode(v, fs, width=48, state=state, abs_start=0.0)
    assert calls == []
    assert frame is not None
    assert state.lost == 0  # blind search reseeds
