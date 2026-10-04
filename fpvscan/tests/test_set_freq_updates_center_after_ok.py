"""set_center_freq must publish the new LO only after the lib call succeeds."""
from __future__ import annotations

import inspect
import types

import pytest

from fpvscan.sdr import bladerf as B


def _bare():
    rad = B.BladeRF.__new__(B.BladeRF)
    rad.lib = types.SimpleNamespace()
    rad._dev = object()
    rad.ch = 0
    rad._fc = 5740e6
    rad._fs = 40e6
    rad.device = ""
    return rad


def _ck_ok(rc, what):
    if rc < 0:
        raise B.BladeRFError(what)
    return rc


def test_successful_set_center_writes_fc(monkeypatch):
    """Sweep / snapshot / occupancy all trust `src.center_freq`.
    A hop that moves the RFIC without `_fc = hz` attributes the next
    blob to the previous step — click locks a neighbor.
    """
    monkeypatch.setattr(B, "_ck", _ck_ok)
    rad = _bare()
    seen = []

    def set_freq(_dev, _ch, hz):
        seen.append(int(hz))
        return 0

    rad.lib.bladerf_set_frequency = set_freq
    rad.set_center_freq(5800e6)
    assert seen == [5_800_000_000]
    assert rad.center_freq == 5800e6


def test_failed_set_center_leaves_fc_alone(monkeypatch):
    """A failed hop that still writes `_fc` makes Engine believe it
    is on the new MHz while the analog filter and LO stayed put.
    """
    monkeypatch.setattr(B, "_ck", _ck_ok)
    rad = _bare()
    rad.lib.bladerf_set_frequency = lambda *_a: -1  # not ERR_IO, no recover
    with pytest.raises(B.BladeRFError):
        rad.set_center_freq(5800e6)
    assert rad.center_freq == 5740e6


def test_set_center_source_assigns_fc_only_after_ck():
    src = inspect.getsource(B.BladeRF.set_center_freq)
    assert src.index("_ck(rc") < src.index("self._fc = float(hz)")
