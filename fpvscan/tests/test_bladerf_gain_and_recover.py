"""bladeRF gain clamp, range fallback, bias query, and USB recover backoff.

Instantiating BladeRF() loads libbladeRF. These tests build a bare
instance and never touch USB.
"""
from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest

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
    rad._fs = 0.0
    rad._streaming = False
    rad._quick = {}
    return rad


class _Lib:
    def __init__(self):
        self.gain = None
        self.mode = None

    def bladerf_set_gain_mode(self, _dev, _ch, mode):
        self.mode = mode
        return 0

    def bladerf_set_gain(self, _dev, _ch, db):
        self.gain = int(db)
        return 0

    def bladerf_strerror(self, rc):
        return b"ok"


def test_set_gain_clamps_to_board_range():
    """YAML 35 dB + bias-tee offset, or a CLI fat-finger, can exceed
    the xA4 max. Unclamped values error on some lib versions and
    silently wrap on others.
    """
    lib = _Lib()
    rad = _bare(lib)
    rad.gain_range = lambda: (0, 30)
    rad.set_gain(50)
    assert lib.gain == 30
    assert rad.gain_db == 30
    assert lib.mode == B.GAIN_MGC
    rad.set_gain(-20)
    assert lib.gain == 0
    assert rad.gain_db == 0


def test_agc_skips_the_numeric_gain_write():
    lib = _Lib()
    rad = _bare(lib)
    rad.agc = True
    rad.gain_range = lambda: (0, 30)
    rad.set_gain(50)
    assert lib.gain is None
    assert lib.mode == B.GAIN_SLOWATTACK_AGC
    assert rad.gain_db == 50


def test_gain_range_falls_back_to_minus15_60():
    """bringup.py / record.py iterate range(g_lo+5, g_hi+1, 10).
    A failed get_gain_range must not return (0, 0) or the sweep is empty.
    """
    rad = _bare()

    def boom(*_a, **_k):
        raise RuntimeError("old lib")

    rad.lib.bladerf_get_gain_range = boom
    assert rad.gain_range() == (-15, 60)


def test_get_bias_tee_is_none_without_symbol_or_device():
    rad = _bare(types.SimpleNamespace())
    assert rad.get_bias_tee() is None
    rad._dev = None
    rad.lib = types.SimpleNamespace(bladerf_get_bias_tee=lambda *a: 0)
    assert rad.get_bias_tee() is None


def test_set_bias_tee_missing_symbol_returns_false():
    rad = _bare(types.SimpleNamespace())
    assert rad.set_bias_tee(True) is False
    assert rad.bias_tee is False


def test_quick_retune_without_a_profile_is_false():
    rad = _bare()
    assert rad.quick_retune(5800e6) is False
    assert rad._fc == 0.0


def test_prime_quick_tune_stops_on_first_error():
    """xA4 builds that lack profiles fail on the first frequency.
    Continuing would walk the whole 400–6000 MHz plan on a dead API.
    """
    rad = _bare()
    seen = []

    def boom(hz):
        seen.append(int(hz))
        raise B.BladeRFError("no quick tune")

    rad.set_center_freq = boom
    assert rad.prime_quick_tune([5800e6, 5840e6, 5880e6]) == 0
    assert seen == [5_800_000_000]


def test_recover_backoff_grows_four_tenths():
    """USB re-enumeration needs hundreds of ms, not a tight retry loop
    that loses the board for good. 0.4, 0.8, 1.2… × 6 attempts.
    """
    sig = inspect.signature(B.BladeRF._recover)
    assert sig.parameters["retries"].default == 6
    body = SRC[SRC.index("def _recover"):SRC.index("def info")]
    assert "time.sleep(0.4 * (i + 1))" in body
    assert "self._streaming = False" in body
    assert "self.set_gain(g)" in body


def test_load_lib_with_no_candidates_mentions_the_apt_packages():
    prev, prev_path = B._LIB, B._LIB_PATH
    B._LIB = None
    B._LIB_PATH = None
    try:
        with pytest.raises(B.BladeRFError) as ei:
            B.load_lib("/definitely/missing/bladeRF.dll")
        msg = str(ei.value)
        assert "не завантажується" in msg or "не знайдено" in msg
    finally:
        B._LIB, B._LIB_PATH = prev, prev_path
    assert "libbladerf2" in B._not_installed_msg() or "bladeRF.dll" in B._not_installed_msg()
