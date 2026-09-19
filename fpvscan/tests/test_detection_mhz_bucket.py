"""Detections and auto-peek cooldown share one MHz bucket.

The console hit map uses Math.round(freq/1e6). The engine stores
int(freq/1e6). A refactor that keys by Hz (or by round()) desyncs
Clear / cooldown / the hit list.
"""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_merge_stores_truncated_mhz_key():
    eng = make_engine()
    eng.cfg["scan"]["confirm_hits"] = 99
    eng._merge(Detection(
        freq_hz=5800.9e6, bandwidth_hz=10e6, snr_db=12.0, hits=1,
        confidence=0.9))
    assert 5800 in eng.state.detections
    assert 5801 not in eng.state.detections
    assert eng.state.detections[5800].freq_hz == 5800.9e6


def test_peek_cooldown_uses_the_same_mhz_bucket():
    """5800.2 and 5800.8 are the same VTx. Peeking both would restart
    LOCK mid-clip and look like a drop.
    """
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = True
    eng.cfg["scan"]["auto_peek_min_conf"] = 0.0
    first = Detection(
        freq_hz=5800.2e6, bandwidth_hz=10e6, snr_db=12.0,
        confidence=0.9)
    eng._maybe_peek(first)
    assert eng.state.mode == "LOCK"
    assert 5800 in eng._peeked

    eng.state.mode = "SWEEP"
    eng.state.lock_target = None
    eng.state.auto = False
    second = Detection(
        freq_hz=5800.8e6, bandwidth_hz=10e6, snr_db=14.0,
        confidence=0.9)
    eng._maybe_peek(second)
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
