"""Omitted LOCK mixer / occupancy / spectrum keys keep the channel centered."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def _drain_spectrum(eng):
    out = []
    while True:
        try:
            ev = eng.events.get_nowait()
        except Exception:
            break
        if ev.get("type") == "spectrum":
            out.append(ev)
    return out


def test_omitted_lo_offset_is_zero_and_channel_bw_is_20mhz():
    """#17 / #32 forwarded an explicit LO offset and clamped a too-wide
    one. #18 used measured occupancy for `_lock_bw`. With no detection
    and omitted keys the fallbacks are lo_offset=0 and channel_bw=20 MHz
    (then `_lock_bw` floors at max(8, 20)=20). A leftover 8 MHz offset
    from the YAML comment mixes the picture onto the LO leak; a 0 Hz
    channel_bw would still floor at 8 MHz but a required-key read
    KeyErrors LOCK on a partial config.
    """
    seen: dict = {}

    def fake_ch(iq, fs, offset_hz, out_bw_hz, fast=True):
        seen["offset_hz"] = float(offset_hz)
        seen["out_bw_hz"] = float(out_bw_hz)
        return np.zeros(64, dtype=np.complex64), 1e6

    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    eng.cfg["video"].pop("lo_offset_hz")
    eng.cfg["video"].pop("channel_bw_hz")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.demod.channelize", side_effect=fake_ch), \
             patch("fpvscan.engine.cvbs.decode", return_value=None):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert seen["offset_hz"] == 0.0
    assert seen["out_bw_hz"] == 20e6


def test_omitted_spectrum_every_is_eight_not_one():
    """#29 pinned shipped YAML spectrum_every=8 and `_lock_n % every == 1`.
    The same expression with every=1 never emits (1%1==0). Dropping the
    key must keep 8 so the first LOCK spectrum still paints. Starting
    `_lock_n` at 8 → 9: 9%8==1 emits, 9%1==0 would not.
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    eng.cfg["video"].pop("spectrum_every")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    eng._lock_n = 8
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=None), \
             patch("fpvscan.engine.spectrum.psd_db",
                   return_value=np.zeros(4096, dtype=np.float32)), \
             patch("fpvscan.engine.spectrum.downsample_for_display",
                   return_value=[0.0] * 8), \
             patch("fpvscan.engine.spectrum.noise_floor_db", return_value=-90.0):
            eng._do_lock()
        assert _drain_spectrum(eng), "omitted spectrum_every=1 would never emit"
    finally:
        eng._stop_reader()
