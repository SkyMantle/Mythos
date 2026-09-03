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
