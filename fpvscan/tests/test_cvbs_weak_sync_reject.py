"""CVBS must refuse a frame when line sync is missing or incoherent.

A 20 ms floor already exists; these gates fire *after* that, on captures
that look long enough to decode. Letting them through paints snow into
LOCK and auto-peek, then the operator treats garbage as a VTx.
"""
from __future__ import annotations

import numpy as np
import pytest

from fpvscan.dsp import cvbs
from fpvscan.sdr.sim import _cvbs


def _baseband(fs: float = 8e6, seconds: float = 0.05, line_hz: float = 15625.0):
    t = np.arange(int(fs * seconds)) / fs
    return _cvbs(t, line_hz), fs


def test_constant_signal_longer_than_20ms_is_rejected():
    """hi-lo < 1e-9 after the 20 ms floor — DC/LO leak, not video."""
    fs = 8e6
    v = np.full(int(fs * 0.05), 0.30, dtype=np.float32)
    assert cvbs.decode(v, fs, width=32) is None


def test_incoherent_noise_is_rejected_by_interval_score():
    """best_s <= 0.5 after trying both polarities. Gaussian noise has
    falling edges but they do not agree on a line period.
    """
    fs = 8e6
    rng = np.random.default_rng(1)
    v = rng.normal(0.5, 0.25, int(fs * 0.05)).astype(np.float32)
    assert cvbs.decode(v, fs, width=32) is None


def test_too_few_sync_edges_is_rejected():
    """< 24 falling edges → fail. A 25 ms burst of two pulses must not
    become a 'frame' that pollutes auto-peek.
    """
    fs = 8e6
    v = np.full(int(fs * 0.025), 0.7, dtype=np.float32)
    v[100:180] = 0.0
    v[10_000:10_080] = 0.0
    assert cvbs.decode(v, fs, width=32) is None


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_horizontal_sync_without_vsync_is_unlocked():
    """Regular 4.7 µs pulses at 15 625 Hz but no broad equalizing run.
    Console 'Синхро є/нема' is Frame.locked, not 'did we get rows'.
    """
    fs = 8e6
    line_hz = 15625.0
    t = np.arange(int(fs * 0.05)) / fs
    x = t % (1.0 / line_hz)
    v = np.full_like(t, 0.70, dtype=np.float32)
    v[x < 4.7e-6] = 0.0
    frame = cvbs.decode(v, fs, width=32, auto_levels=False)
    assert frame is not None
    assert frame.locked is False
    assert abs(frame.line_rate - line_hz) < 20
    assert frame.lines >= 32


def test_clean_pal_sim_still_locks():
    v, fs = _baseband()
    frame = cvbs.decode(v, fs, width=32)
    assert frame is not None
    assert frame.locked is True
    assert frame.standard == "PAL"
