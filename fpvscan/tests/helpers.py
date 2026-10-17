"""Shared fixtures for engine tests. No hardware, no wall-clock waits."""
from __future__ import annotations

from queue import Queue

import numpy as np

from fpvscan.engine import Engine


class MockSource:
    """IQ source that records retune/gain/bias calls and yields a tone or zeros."""

    name = "mock"

    def __init__(self, fs: float = 20e6, tone_hz: float = 0.0):
        self._fs = float(fs)
        self._fc = 0.0
        self.tone_hz = float(tone_hz)
        self.retune_calls = 0
        self.set_sample_rate_calls = 0
        self.gain = None
        self.bias = None
        self.opened = False

    @property
    def sample_rate(self):
        return self._fs

    @property
    def center_freq(self):
        return self._fc

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def set_gain(self, db):
        self.gain = float(db)

    def set_sample_rate(self, hz):
        self.set_sample_rate_calls += 1
        self._fs = float(hz)

    def set_center_freq(self, hz):
        self._fc = float(hz)

    def set_bias_tee(self, on):
        self.bias = bool(on)
        return True

    def _iq(self, n: int) -> np.ndarray:
        n = int(n)
        if self.tone_hz == 0.0:
            return np.zeros(n, dtype=np.complex64)
        t = np.arange(n, dtype=np.float64) / self._fs
        return np.exp(2j * np.pi * self.tone_hz * t).astype(np.complex64)

    def read(self, n: int) -> np.ndarray:
        return self._iq(n)

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.retune_calls += 1
        self._fc = float(hz)
        return self._iq(n)


def engine_cfg(**video_kw) -> dict:
    video = {
        "sample_rate": 20e6,
        "channel_bw_hz": 8e6,
        "lo_offset_hz": 0.0,
        "capture_ms": 5,
        "idle_ms": 0,
        "ring_seconds": 0.05,
        "afc": False,
        "afc_gain": 0.5,
        "afc_limit_hz": 8e6,
        "afc_deadband_hz": 50e3,
        "afc_recenter_frac": 0.75,
        "width": 64,
        "spectrum_every": 10_000,
        "min_lines": 200,
        "average": 0.0,
        "motion_thresh": 24.0,
        "auto_levels": True,
        "sharpen": 0.0,
    }
    video.update(video_kw)
    return {
        "sdr": {"gain_db": 30, "bias_tee_gain_offset_db": 15},
        "scan": {
            "sample_rate": 20e6,
            "start_hz": 400e6,
            "stop_hz": 6000e6,
            "channel_bw_hz": 20e6,
            "step_hz": 0,
            "confirm_hits": 2,
            "priority_bands": False,
            "auto_peek": True,
            "auto_peek_min_conf": 0.6,
            "auto_peek_secs": 8,
            "auto_peek_cooldown_s": 60,
        },
        "video": video,
    }


def make_engine(src=None, events=None, **video_kw) -> Engine:
    src = src if src is not None else MockSource()
    ev = events if events is not None else Queue()
    return Engine(src, engine_cfg(**video_kw), ev)
