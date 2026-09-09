"""LOCK picture integrity: track window, min_lines vs average, DC strip, rec fps."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from fpvscan.dsp.cvbs import DecodeState, Frame, FIELD_LINES
from fpvscan.dsp import demod
from tests.helpers import MockSource, make_engine


class _RecRing:
    """Ring stand-in that records the snapshot length LOCK asked for."""

    def __init__(self):
        self.n = None

    def snapshot(self, n):
        self.n = int(n)
        return np.zeros(int(n), dtype=np.complex64), 0.0

    def write(self, iq):
        pass


def _lock_with_ring(eng, ring, *, period, standard, lost, dec=1):
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = ring
    eng._lock_tuned = 5800e6
    eng._lock_dec = dec
    eng._lock_state = DecodeState(period=period, standard=standard, lost=lost)
    # Snapshot length is what we assert; skip the CVBS path (zeros are not a field).
    with patch("fpvscan.engine.cvbs.decode", return_value=None):
        try:
            eng._do_lock()
        finally:
            eng._stop_reader()
    return ring.n


def test_tracking_miss_asks_for_full_capture_not_short_window():
    """After a miss, a short n_track window can skip the next vsync entirely."""
    fs = 20e6
    period = fs / 15625.0
    capture_ms = 80
    n_full = int(fs * capture_ms / 1000)
    n_track = int(period * FIELD_LINES["PAL"] * 1 * 1.7)
    assert n_track < n_full

    src = MockSource(fs=fs)
    tracked = make_engine(src=src, capture_ms=capture_ms, idle_ms=0, afc=False)
    n_ok = _lock_with_ring(
        tracked, _RecRing(), period=period, standard="PAL", lost=0, dec=1
    )
    missed = make_engine(src=MockSource(fs=fs), capture_ms=capture_ms, idle_ms=0, afc=False)
    n_miss = _lock_with_ring(
        missed, _RecRing(), period=period, standard="PAL", lost=1, dec=1
    )
    assert n_ok == n_track
    assert n_miss == n_full
    assert n_miss > n_ok


def test_short_frame_does_not_pollute_motion_average(monkeypatch):
    """min_lines must reject before _acc is updated, or one torn field smears LOCK."""
    src = MockSource()
    eng = make_engine(
        src=src, capture_ms=30, idle_ms=0, afc=False, average=4.0, min_lines=200, width=64
    )
    short = Frame(
        luma=np.full((40, 64), 255, dtype=np.uint8),
        line_rate=15625.0,
        lines=40,
        standard="PAL",
        locked=True,
    )
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *a, **k: short)
    monkeypatch.setattr("fpvscan.engine.cvbs.encode", lambda *a, **k: b"x")

    class _FullRing:
        def snapshot(self, n):
            return np.zeros(int(n), dtype=np.complex64), 0.0

        def write(self, iq):
            pass

    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    eng._lock_dec = 1
    eng._lock_state = DecodeState()
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert eng._acc is None


def test_lock_strips_dc_before_channelize(monkeypatch):
    """ADC offset / LO leakage sits at DC. Leaving it in pulls AFC and washes luma."""
    src = MockSource()
    src._iq = lambda n: np.full(int(n), 0.4 + 0.1j, dtype=np.complex64)
    eng = make_engine(src=src, capture_ms=20, idle_ms=0, afc=False)
    seen = []
    real = demod.channelize

    def spy(iq, fs, offset_hz, out_bw_hz, fast=True):
        seen.append(complex(np.mean(iq)))
        return real(iq, fs, offset_hz, out_bw_hz, fast=fast)

    monkeypatch.setattr("fpvscan.engine.demod.channelize", spy)

    class _FullRing:
        def snapshot(self, n):
            return np.full(int(n), 0.4 + 0.1j, dtype=np.complex64), 0.0

        def write(self, iq):
            pass

    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert seen, "channelize was not called"
    assert abs(seen[0]) < 1e-3


def test_rec_start_uses_shipped_fps_height_preset():
    """Engine default rec_fps is 5. Shipped config is 24 — a missed key makes files crawl."""
    src = MockSource()
    eng = make_engine(src=src)
    eng.cfg["video"]["rec_fps"] = 24
    eng.cfg["video"]["rec_height"] = 288
    eng.cfg["video"]["rec_preset"] = "veryfast"
    eng.cfg["video"]["width"] = 640
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    captured = {}

    class _FakeRec:
        def __init__(self, path, width, height, fps=5, crf=24, preset="veryfast", exe=None):
            captured.update(width=width, height=height, fps=fps, preset=preset, exe=exe)

        def start(self):
            pass

    with patch("fpvscan.engine.VideoRecorder", _FakeRec):
        with patch("fpvscan.engine.paths.stamped", return_value="rec_x.mp4"):
            with patch("fpvscan.engine.paths.ensure", side_effect=lambda p: p):
                eng._rec_start()
    assert captured["fps"] == 24
    assert captured["height"] == 288
    assert captured["width"] == 640
    assert captured["preset"] == "veryfast"
    assert eng._rec is not None
