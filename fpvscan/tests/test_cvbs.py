"""CVBS decode is the LOCK picture path: polarity, geometry, and field tracking."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.sdr.sim import _cvbs


def _baseband(fs: float = 8e6, seconds: float = 0.08, line_hz: float = 15625.0):
    t = np.arange(int(fs * seconds)) / fs
    return _cvbs(t, line_hz), fs


def test_decode_pal_bars_locks_and_has_rows():
    v, fs = _baseband()
    frame = cvbs.decode(v, fs, width=64)
    assert frame is not None
    assert frame.standard == "PAL"
    assert abs(frame.line_rate - 15625) < 5
    assert frame.lines >= 64
    assert frame.locked
    assert frame.luma.shape[1] == 64
    assert frame.luma.dtype == np.uint8
    # gradient bars: left of active picture darker than right
    assert frame.luma[:, 2].mean() < frame.luma[:, -2].mean()


def test_decode_inverted_polarity():
    v, fs = _baseband()
    frame = cvbs.decode(-v, fs, width=64)
    assert frame is not None
    assert frame.locked
    assert abs(frame.line_rate - 15625) < 5


def test_decode_too_short_returns_none():
    fs = 8e6
    v = np.zeros(int(fs * 0.01), dtype=np.float32)
    assert cvbs.decode(v, fs, width=64) is None


def test_decode_state_tracks_across_consecutive_fields():
    fs = 8e6
    line_hz = 15625.0
    # two back-to-back 80 ms captures, second starts where the first ended
    n = int(fs * 0.08)
    t0 = np.arange(n) / fs
    t1 = (n + np.arange(n)) / fs
    state = cvbs.DecodeState()
    f0 = cvbs.decode(_cvbs(t0, line_hz), fs, width=48, state=state, abs_start=0.0)
    assert f0 is not None
    assert state.period is not None
    assert state.lost == 0
    period_before = state.period
    f1 = cvbs.decode(_cvbs(t1, line_hz), fs, width=48, state=state, abs_start=float(n))
    assert f1 is not None
    assert state.lost == 0
    # tracking must not throw away the period and fall back to a blind search reset
    assert state.period is not None
    assert abs(state.period - period_before) / period_before < 0.05


def test_one_field_clamp_stops_before_next_vsync():
    """A 80 ms capture holds several fields; without the clamp, max_lines=288
    would paint through the next equalizing pulses and tear the picture."""
    v, fs = _baseband(seconds=0.08)
    frame = cvbs.decode(v, fs, width=48, max_lines=288)
    assert frame is not None
    assert 200 <= frame.lines <= 270
    assert frame.luma.shape[0] == frame.lines


def test_genlock_snaps_to_actual_sync_edges():
    """A 0.4-sample period error accumulates to ~16 samples over 40 lines;
    TBC must follow the real falling edges instead of the global grid."""
    period = 512.0
    n_lines = 40
    t0 = 100.0
    true = t0 + np.arange(n_lines) * (period + 0.4)
    v = np.full(int(true[-1] + period + 20), 0.5, dtype=np.float32)
    for s in true:
        i = int(np.floor(s))
        v[i:i + 10] = 0.0
    starts = cvbs._genlock_starts(v, t0, period, n_lines, thr=0.18)
    predicted = t0 + (n_lines - 1) * period
    assert abs(starts[-1] - true[-1]) < 1.5
    assert abs(predicted - true[-1]) > 5


def test_auto_levels_stretches_washed_out_active_region():
    period = 100.0
    n_lines, width = 30, 40
    starts = 20.0 + np.arange(n_lines) * period
    v = np.full(int(starts[-1] + period + 5), 0.40, dtype=np.float32)
    for s in starts:
        i = int(s)
        v[i:i + 5] = 0.0
        ramp = 0.40 + 0.10 * np.linspace(0.0, 1.0, 70, dtype=np.float32)
        v[i + 20:i + 90] = ramp
    a0, a1 = 0.20, 0.90
    stretched = cvbs._render(v, starts, period, a0, a1, width,
                             auto_levels=True, sharpen=0.0)
    washed = cvbs._render(v, starts, period, a0, a1, width,
                          auto_levels=False, sharpen=0.0)
    assert stretched.max() == 255
    assert washed.max() < 120
    assert stretched.std() > 2 * washed.std()


def test_sharpen_raises_horizontal_edge_energy():
    v, fs = _baseband(seconds=0.06)
    plain = cvbs.decode(v, fs, width=64, sharpen=0.0, auto_levels=True)
    sharp = cvbs.decode(v, fs, width=64, sharpen=0.8, auto_levels=True)
    assert plain is not None and sharp is not None
    g0 = np.abs(np.diff(plain.luma.astype(np.float32), axis=1)).mean()
    g1 = np.abs(np.diff(sharp.luma.astype(np.float32), axis=1)).mean()
    assert g1 > g0 * 1.15


def test_encode_webp_and_jpeg_are_nonempty():
    v, fs = _baseband(seconds=0.06)
    frame = cvbs.decode(v, fs, width=32)
    assert frame is not None
    webp = cvbs.encode(frame, "webp", 70, height=48)
    jpeg = cvbs.to_jpeg(frame, quality=50, height=48)
    png = cvbs.encode(frame, "png", height=48)
    assert webp[:4] == b"RIFF"
    assert jpeg[:2] == b"\xff\xd8"
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(webp) > 32
