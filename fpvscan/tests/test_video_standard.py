#!/usr/bin/env python3
"""NTSC must not be labeled PAL: LOCK tracking uses FIELD_LINES per standard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.dsp import cvbs, demod
from fpvscan.dsp.cvbs import DecodeState, FIELD_LINES, _predict_local_t0
from fpvscan.dsp.demod import LINE_NTSC, LINE_PAL, identify_standard
from fpvscan.sdr.sim import C_LINE_NTSC, C_LINE_PAL, Emitter, SimSource


def test_identify_standard_picks_nearest():
    assert identify_standard(LINE_PAL) == "PAL"
    assert identify_standard(LINE_NTSC) == "NTSC"
    # Decoder (120 Hz) and classifier (150 Hz) tolerances both exceed the
    # 109 Hz PAL–NTSC gap, so first-match PAL would swallow every NTSC signal.
    assert identify_standard(LINE_NTSC, tol_hz=150.0) == "NTSC"
    assert identify_standard(LINE_PAL, tol_hz=150.0) == "PAL"
    assert identify_standard(14000.0) == "?"


def _line_sync_wave(fs: float, line_hz: float, seconds: float = 0.12) -> np.ndarray:
    """Line-rate sync pulses so classify_video sees the fundamental + harmonics."""
    n = int(fs * seconds)
    t = np.arange(n, dtype=np.float64) / fs
    phase = np.mod(t * line_hz, 1.0)
    return np.where(phase < 0.08, 0.0, 0.72).astype(np.float32)


def test_classifier_ntsc_not_pal_at_default_tolerance():
    fs = 1_000_000.0
    pal = demod.classify_video(_line_sync_wave(fs, LINE_PAL), fs, tol_hz=150.0)
    ntsc = demod.classify_video(_line_sync_wave(fs, LINE_NTSC), fs, tol_hz=150.0)
    assert pal.is_video and pal.standard == "PAL", pal
    assert ntsc.is_video and ntsc.standard == "NTSC", ntsc


def test_predict_ntsc_field_is_50_lines_off_if_labeled_pal():
    """One NTSC field later: NTSC prediction lands on vsync, PAL is 50 lines away."""
    period = 1000.0
    st = DecodeState(sign=1.0, period=period, standard="NTSC", abs_t0=100.0, lost=0)
    one_ntsc_field = period * FIELD_LINES["NTSC"]
    loc, n, _fp = _predict_local_t0(st, abs_start=100.0 + one_ntsc_field)
    assert n == 1
    assert abs(loc) < 1e-6

    st.standard = "PAL"
    loc, n, _fp = _predict_local_t0(st, abs_start=100.0 + one_ntsc_field)
    # 312.5 − 262.5 = 50 lines; search window is < 1 line, so tracking misses.
    assert abs(loc) > 40.0 * period, loc


def _sim_one_frame(line_rate: float):
    fs = 20e6
    src = SimSource(
        emitters=[Emitter(5800e6, -18.0, "fpv", 8e6, line_rate, "t")],
        noise_db=-85, seed=3,
    )
    src.open()
    src.set_sample_rate(fs)
    src.set_center_freq(5800e6)
    src.read(int(fs * 0.04))
    iq = src.read(int(fs * 0.08))
    base = demod.deemphasis(demod.fm_demod(iq, fs, 4e6), fs)
    return cvbs.decode(base, fs, width=160)


def test_sim_ntsc_decodes_as_ntsc():
    fr = _sim_one_frame(C_LINE_NTSC)
    assert fr is not None
    assert fr.standard == "NTSC", fr.standard


def test_sim_pal_decodes_as_pal():
    fr = _sim_one_frame(C_LINE_PAL)
    assert fr is not None
    assert fr.standard == "PAL", fr.standard


def _ntsc_baseband(fs: float, n: int, abs_start: float = 0.0) -> np.ndarray:
    """True 262.5-line NTSC fields (the sim uses 262, which is not enough here)."""
    t = (abs_start + np.arange(n, dtype=np.float64)) / fs
    line_phase = np.mod(t * LINE_NTSC, 1.0)
    field_line = np.mod(t * LINE_NTSC, FIELD_LINES["NTSC"])
    v = np.full(n, 0.72, dtype=np.float32)
    v[line_phase < 0.07] = 0.0
    vs = field_line < 3.0
    v[vs] = np.where(np.mod(field_line[vs], 0.5) < 0.40, 0.0, 0.30)
    # Active luma varies slowly so consecutive fields look alike.
    v[(field_line >= 20) & (line_phase > 0.15)] = 0.40 + 0.40 * (
        np.mod(line_phase[(field_line >= 20) & (line_phase > 0.15)] * 8, 1.0) > 0.5
    ).astype(np.float32)
    return v


def test_ntsc_tracking_holds_across_one_field():
    fs = 8e6
    period = fs / LINE_NTSC
    field = period * FIELD_LINES["NTSC"]
    n = int(field * 1.6)
    state = DecodeState()
    fr0 = cvbs.decode(_ntsc_baseband(fs, n, 0.0), fs, width=96,
                      state=state, abs_start=0.0)
    assert fr0 is not None and fr0.standard == "NTSC", (
        None if fr0 is None else fr0.standard)
    assert state.standard == "NTSC"

    fr1 = cvbs.decode(_ntsc_baseband(fs, n, field), fs, width=96,
                      state=state, abs_start=field)
    assert fr1 is not None
    assert state.lost == 0
    assert state.standard == "NTSC"
    h, w = min(24, fr0.luma.shape[0], fr1.luma.shape[0]), min(
        fr0.luma.shape[1], fr1.luma.shape[1])
    corr = np.corrcoef(fr0.luma[:h, :w].ravel().astype(np.float32),
                       fr1.luma[:h, :w].ravel().astype(np.float32))[0, 1]
    assert corr > 0.9, f"tracking jumped, corr={corr:.3f} lost={state.lost}"


if __name__ == "__main__":
    test_identify_standard_picks_nearest()
    test_classifier_ntsc_not_pal_at_default_tolerance()
    test_predict_ntsc_field_is_50_lines_off_if_labeled_pal()
    test_sim_ntsc_decodes_as_ntsc()
    test_sim_pal_decodes_as_pal()
    test_ntsc_tracking_holds_across_one_field()
    print("OK")
