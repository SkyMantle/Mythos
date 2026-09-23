"""USB timeout must restart the stream; IO must reopen the device.

Instantiating BladeRF() loads libbladeRF. These tests build a bare
instance and never touch USB.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from fpvscan.sdr import bladerf as B


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def _bare(lib=None):
    rad = B.BladeRF.__new__(B.BladeRF)
    rad.lib = lib if lib is not None else types.SimpleNamespace()
    rad._dev = object()
    rad.ch = 0
    rad.agc = False
    rad.gain_db = 0.0
    rad.bias_tee = False
    rad.device = ""
    rad._fc = 0.0
    rad._fs = 40e6
    rad._streaming = True
    rad._quick = {}
    rad._raw = None
    rad.use_meta = False
    rad.timeout_ms = 3500
    rad.timeouts = 0
    rad.io_errors = 0
    rad.overflows = 0
    rad.clip_frac = 0.0
    return rad


class _RxLib:
    def __init__(self, first_rc: int):
        self.first_rc = first_rc
        self.calls = 0

    def bladerf_sync_rx(self, *_a, **_k):
        self.calls += 1
        return self.first_rc if self.calls == 1 else 0

    def bladerf_strerror(self, rc):
        return b"err"


def test_timeout_restarts_the_stream_not_the_device():
    """A stalled sync_rx is a USB glitch. _recover sleeps 0.4 s+ and
    walks gain/fs/freq; doing that on every timeout makes a 400 MHz–6 GHz
    pass unusable. Timeout must bump `timeouts` and call _config_stream.
    """
    lib = _RxLib(B.ERR_TIMEOUT)
    rad = _bare(lib)
    calls = {"recover": 0, "reconfig": 0}
    rad._recover = lambda: calls.__setitem__("recover", calls["recover"] + 1)
    rad._config_stream = lambda: calls.__setitem__(
        "reconfig", calls["reconfig"] + 1)
    out = rad.read(32)
    assert rad.timeouts == 1
    assert calls["reconfig"] == 1
    assert calls["recover"] == 0
    assert lib.calls == 2
    assert out.dtype == np.complex64
    assert out.shape == (32,)


def test_io_error_reopens_the_device():
    """ERR_IO means the handle is dead. Restarting only the stream
    keeps calling sync_rx on a null device (access violation).
    """
    lib = _RxLib(B.ERR_IO)
    rad = _bare(lib)
    calls = {"recover": 0, "reconfig": 0}
    rad._recover = lambda: calls.__setitem__("recover", calls["recover"] + 1)
    rad._config_stream = lambda: calls.__setitem__(
        "reconfig", calls["reconfig"] + 1)
    out = rad.read(16)
    assert calls["recover"] == 1
    assert calls["reconfig"] == 0
    assert rad.timeouts == 0
    assert out.shape == (16,)


def test_set_frequency_io_also_recovers():
    """Sweep walks RFIC bands; the USB control channel sometimes drops
    mid-pass. set_center_freq must recover and retry, not raise on the
    first -5 and kill the engine thread.
    """
    body = SRC[SRC.index("def set_center_freq"):SRC.index("def set_sample_rate")]
    assert "if rc == ERR_IO:" in body
    assert "self._recover()" in body
    retry = body.split("if rc == ERR_IO:", 1)[1]
    assert "bladerf_set_frequency" in retry
