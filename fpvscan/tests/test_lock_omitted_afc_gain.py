"""Omitted afc_gain must still take half the residual per field."""
from __future__ import annotations

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_omitted_afc_gain_is_half(monkeypatch):
    """#54 applied explicit 0.5 to a stubbed 400 kHz error. helpers.py
    always passes 0.5 so other tests stay still. Dropping the key must
    still take half. A leftover 0 never walks `_afc` and a leftover 1.0
    slams the full residual into the mixer — overshoot, then recenter,
    then DecodeState wipe on a free-running VTx.
    """
    eng = make_engine(idle_ms=0, capture_ms=5, afc=True, afc_deadband_hz=50e3)
    eng.cfg["video"].pop("afc_gain")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    monkeypatch.setattr("fpvscan.engine.cvbs.decode", lambda *_a, **_k: None)
    monkeypatch.setattr("fpvscan.engine.demod.freq_error_hz",
                        lambda *_a, **_k: 400e3)
    try:
        eng._do_lock()
    finally:
        eng._stop_reader()
    assert eng._afc == 200e3
