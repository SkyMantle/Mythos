"""Tracking snapshot length is in raw IQ samples, not channelized ones."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.config import load
from fpvscan.dsp.cvbs import FIELD_LINES, DecodeState
from tests.helpers import MockSource, make_engine


ROOT_CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


class _RecRing:
    def __init__(self):
        self.n = None

    def snapshot(self, n: int):
        self.n = int(n)
        return np.zeros(int(n), dtype=np.complex64), 0


def test_shipped_yaml_omits_track_window_margin():
    """Engine defaults to 1.7 fields. Pinning a 1.0 in YAML would make
    a 1-sample phase error miss V-sync and roll the picture; omitting
    the key keeps the headroom.
    """
    assert "track_window_margin" not in ROOT_CFG["video"]


def test_tracking_window_multiplies_period_by_dec_and_margin():
    """`DecodeState.period` is measured after channelize. The ring
    snapshot is raw IQ. Forgetting `* _lock_dec` asks for one-fifth
    of a field at 40 Msps / 8 MHz, the 20 ms floor then always wins,
    and tracking never sees a full PAL field — LOCK looks unlocked
    on a confirmed analog bird. The 1.7 margin is the only slack
    that still finds V-sync when the predicted t0 is a few lines off.
    """
    fs = 40e6
    dec = 5
    ch_bw = 8e6
    assert max(1, int(fs / ch_bw)) == dec
    period = (fs / dec) / 15625.0
    margin = 1.7
    n_track = int(period * FIELD_LINES["PAL"] * dec * margin)
    n_full = int(fs * 0.120)
    n_floor = int(fs * 0.02)
    n_forgot_dec = int(period * FIELD_LINES["PAL"] * margin)
    assert n_floor < n_track < n_full
    assert n_forgot_dec < n_floor

    src = MockSource(fs=fs)
    eng = make_engine(
        src=src, capture_ms=120, idle_ms=0, afc=False,
        channel_bw_hz=ch_bw, sample_rate=fs,
    )
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    ring = _RecRing()
    eng._ring = ring
    eng._lock_tuned = 5800e6
    eng._lock_dec = dec
    eng._lock_state = DecodeState(period=period, standard="PAL", lost=0)
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()

    assert ring.n == n_track
    assert ring.n != n_forgot_dec
    assert ring.n != n_full
    assert ring.n != n_floor
