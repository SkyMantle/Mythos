"""Omitted AFC keys must still track a free-running VTx without slamming."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def _lock_ready(eng):
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6


def test_omitted_afc_is_on():
    """#54 pinned shipped YAML `afc: true`. helpers.py always passes
    False so other tests stay still. Dropping the key must still call
    `freq_error_hz` — defaulting False leaves LOCK on the physical LO
    while a free-running VTx walks out of the channelizer.
    """
    called: list[float] = []

    def spy(*_a, **_k):
        called.append(1.0)
        return 0.0

    eng = make_engine(idle_ms=0, capture_ms=5)
    eng.cfg["video"].pop("afc")
    _lock_ready(eng)
    try:
        with patch("fpvscan.engine.demod.freq_error_hz", side_effect=spy), \
             patch("fpvscan.engine.cvbs.decode", return_value=None):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert called, "omitted afc=False would skip freq_error_hz"


def test_omitted_afc_deadband_is_150khz(monkeypatch):
    """#24 / #54 pinned shipped YAML 100 kHz. The `.get` fallback is
    150 kHz. A leftover 0 AFC-s on every sample of IQ noise, walks
    `_afc` into the recenter trip, and resets DecodeState each field.
    helpers.py uses 50 kHz so other tests see small errors — this pops
    it.
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=True, afc_gain=0.5)
    eng.cfg["video"].pop("afc_deadband_hz")
    _lock_ready(eng)
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *_a, **_k: None)
    monkeypatch.setattr("fpvscan.engine.demod.freq_error_hz",
                        lambda *_a, **_k: 100e3)
    try:
        eng._do_lock()
        assert eng._afc == 0.0
        monkeypatch.setattr("fpvscan.engine.demod.freq_error_hz",
                            lambda *_a, **_k: 200e3)
        eng._do_lock()
        assert eng._afc == 100e3
    finally:
        eng._stop_reader()


def test_omitted_afc_limit_is_8mhz(monkeypatch):
    """#21 clamped an explicit limit to Nyquist headroom. #54 pinned
    shipped YAML 20 MHz. The `.get` fallback is 8 MHz. At 40 Msps /
    8 MHz channel, safe_lim is 14 MHz, so a leftover 20 MHz would
    clamp to 14 MHz and a leftover 0 takes the lim<=0 recenter path
    every field. Disable recenter here so the clamp is visible.
    """
    src = MockSource(fs=40e6)
    eng = make_engine(
        src=src, sample_rate=40e6, idle_ms=0, capture_ms=5,
        afc=True, afc_gain=0.5, afc_recenter_frac=2.0,
    )
    eng.cfg["scan"]["sample_rate"] = 40e6
    eng.cfg["video"].pop("afc_limit_hz")
    _lock_ready(eng)
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *_a, **_k: None)
    monkeypatch.setattr("fpvscan.engine.demod.freq_error_hz",
                        lambda *_a, **_k: 30e6)
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert eng._afc == 8e6


def test_omitted_afc_recenter_frac_is_075(monkeypatch):
    """#52 skipped a 400 kHz leftover below an explicit recenter.
    Dropping the key must keep 0.75. A leftover 0 recenters on any
    nonzero `_afc`, calling `_start_reader` and wiping DecodeState
    every field — the pre-#4 roll.
    """
    src = MockSource()
    eng = make_engine(src=src, idle_ms=0, capture_ms=5, afc=True)
    eng.cfg["video"].pop("afc_recenter_frac")
    _lock_ready(eng)
    # At 20 Msps / 8 MHz channel, safe_lim = 5 MHz. 2.5 MHz is 0.5·lim
    # (below 0.75) but any leftover 0-frac would trip.
    eng._afc = 2.5e6
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *_a, **_k: None)
    monkeypatch.setattr("fpvscan.engine.demod.freq_error_hz",
                        lambda *_a, **_k: 0.0)
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert src.retune_calls == 0
    assert eng._afc == 2.5e6
    assert eng._lock_tuned == 5800e6
