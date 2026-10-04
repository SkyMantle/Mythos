"""set_sample_rate must publish the board's actual rate, not the request."""
from __future__ import annotations

import inspect
import types

from fpvscan.engine import Engine
from fpvscan.sdr import bladerf as B


def _bare():
    rad = B.BladeRF.__new__(B.BladeRF)
    rad.lib = types.SimpleNamespace()
    rad._dev = object()
    rad.ch = 0
    rad._fc = 0.0
    rad._fs = 40e6
    rad._streaming = False
    rad.bandwidth_ratio = 0.9
    rad._bw = 0.0
    return rad


def _ck_ok(rc, what):
    if rc < 0:
        raise B.BladeRFError(what)
    return rc


def test_sample_rate_property_is_actual_value_not_request(monkeypatch):
    """bladeRF A4 often returns 34999872 for a 35e6 ask. Engine copies
    `src.sample_rate` into scan/video (merged #2). If the driver stores
    the request instead of `actual.value`, LOCK still sees a 128 Hz
    mismatch after the copy and the 1 Hz guard never sees the real Δ.
    """
    monkeypatch.setattr(B, "_ck", _ck_ok)
    rad = _bare()
    bw_seen = []

    def set_rate(_dev, _ch, hz, actual_ptr):
        assert int(hz) == 35_000_000
        actual_ptr._obj.value = 34_999_872
        return 0

    def set_bw(_dev, _ch, bw, bw_ptr):
        bw_seen.append(int(bw))
        bw_ptr._obj.value = int(bw)
        return 0

    rad.lib.bladerf_set_sample_rate = set_rate
    rad.lib.bladerf_set_bandwidth = set_bw
    rad._config_stream = lambda: None

    rad.set_sample_rate(35e6)
    assert rad.sample_rate == 34_999_872.0
    assert bw_seen == [int(34_999_872 * 0.9)]


def test_set_sample_rate_source_assigns_actual():
    src = inspect.getsource(B.BladeRF.set_sample_rate)
    assert "self._fs = float(actual.value)" in src
    assert "self._fs = float(hz)" not in src


def test_engine_run_copies_rate_after_set():
    """Copying cfg before `set_sample_rate` freezes the YAML ask.
    The board has not spoken yet, so #2's actual-rate write is a no-op.
    """
    body = inspect.getsource(Engine._run)
    set_at = body.index("self.src.set_sample_rate(fs)")
    assert set_at < body.index('self.cfg["scan"]["sample_rate"] = fs')
    assert set_at < body.index('self.cfg["video"]["sample_rate"] = fs')
