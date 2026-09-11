"""LOCK must keep CVBS phase across decimation and never starve line-rate."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.cvbs import DecodeState
from fpvscan.engine import Detection, Engine
from tests.helpers import MockSource, make_engine


class _RecRing:
    """Stand-in IQ ring that records snapshot length / abs_start."""

    def __init__(self, abs_start: int = 0):
        self.n = None
        self.abs_start = int(abs_start)

    def snapshot(self, n: int):
        self.n = int(n)
        return np.zeros(int(n), dtype=np.complex64), self.abs_start


def test_decode_abs_start_is_iq_position_over_decimation():
    """Genlock extrapolates field phase from abs_start. If LOCK passed
    the raw IQ index (or forgot the +1), tracking would jump by `dec`
    samples each frame and the picture would roll on a live channel.
    """
    fs = 20e6
    src = MockSource(fs=fs)
    eng = make_engine(src=src, capture_ms=30, idle_ms=0, afc=False,
                      channel_bw_hz=8e6, sample_rate=fs)
    eng.cfg["scan"]["sample_rate"] = fs
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    ring = _RecRing(abs_start=10_000)
    eng._ring = ring
    eng._lock_tuned = 5800e6

    seen: list[float] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.cvbs.decode

    def wrap(base, rate, width=640, max_lines=288, state=None,
             abs_start=0.0, auto_levels=True, sharpen=0.0):
        seen.append(float(abs_start))
        return None

    eng_mod.cvbs.decode = wrap
    try:
        eng._do_lock()
    finally:
        eng_mod.cvbs.decode = real
        eng._stop_reader()

    assert seen, "decode never ran"
    dec = max(1, int(fs / 8e6))
    assert dec == 2
    assert seen[0] == ring.abs_start / dec + 1


def test_tracking_snapshot_never_shorter_than_20ms():
    """Line-rate needs tens of PAL/NTSC periods. capture_ms in YAML can
    be shorter than that; the tracking window must still floor at 20 ms
    or a locked channel goes snow on the next field.
    """
    fs = 20e6
    capture_ms = 5
    n_full = int(fs * capture_ms / 1000)
    n_floor = int(fs * 0.02)
    assert n_full < n_floor

    src = MockSource(fs=fs)
    eng = make_engine(src=src, capture_ms=capture_ms, idle_ms=0, afc=False,
                      sample_rate=fs)
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    ring = _RecRing()
    eng._ring = ring
    eng._lock_tuned = 5800e6
    eng._lock_dec = 1
    eng._lock_state = DecodeState(
        period=fs / 15625.0, standard="PAL", lost=0,
    )
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()

    assert ring.n == n_floor
    assert ring.n > n_full


def test_lock_channelize_uses_measured_occupancy_not_yaml_bw():
    """YAML channel_bw_hz is a guess. A 20 MHz analog blob channelized
    at the shipped 10 MHz would slice the deviation and LOCK would show
    torn lines on a confirmed hit.
    """
    fs = 40e6
    src = MockSource(fs=fs)
    eng = make_engine(src=src, capture_ms=8, idle_ms=0, afc=False,
                      channel_bw_hz=8e6, sample_rate=fs)
    eng.cfg["scan"]["sample_rate"] = fs
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=20e6, snr_db=12.0,
    )
    ring = _RecRing()
    eng._ring = ring
    eng._lock_tuned = 5800e6

    seen_bw: list[float] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.demod.channelize

    def wrap(iq, rate, offset_hz, out_bw_hz, fast=True):
        seen_bw.append(float(out_bw_hz))
        return real(iq, rate, offset_hz, out_bw_hz, fast=fast)

    eng_mod.demod.channelize = wrap
    try:
        eng._do_lock()
    finally:
        eng_mod.demod.channelize = real
        eng._stop_reader()

    assert seen_bw, "channelize never ran"
    expect = eng._lock_bw(5800e6, 8e6)
    assert expect == 20e6 + 2 * Engine.MERGE_TOL_HZ
    assert seen_bw[0] == expect
    assert seen_bw[0] != 8e6
