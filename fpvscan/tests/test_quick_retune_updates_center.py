"""quick_retune must publish the new LO on the Python side."""
from __future__ import annotations

import inspect
import types

from fpvscan.sdr import bladerf as B


def _bare():
    rad = B.BladeRF.__new__(B.BladeRF)
    rad.lib = types.SimpleNamespace()
    rad._dev = object()
    rad.ch = 0
    rad._fc = 0.0
    rad._fs = 40e6
    rad._quick = {}
    return rad


def test_successful_quick_retune_writes_fc():
    """Sweep / `_grab` trusts `src.center_freq` after a hop.
    `schedule_retune` moving the RFIC without `_fc = hz` leaves
    Engine and `snapshot()['tuned_hz']` on the previous step —
    occupancy is attributed to the wrong MHz and a click locks
    a neighbor (or empty air).
    """
    rad = _bare()
    rad._quick[5_800_000_000] = object()
    seen = []

    def schedule(_dev, _ch, _when, hz, _buf):
        seen.append(int(hz))
        return 0

    rad.lib.bladerf_schedule_retune = schedule
    assert rad.quick_retune(5800e6) is True
    assert seen == [5_800_000_000]
    assert rad._fc == 5800e6


def test_failed_quick_retune_leaves_fc_alone():
    rad = _bare()
    rad._fc = 5740e6
    rad._quick[5_800_000_000] = object()
    rad.lib.bladerf_schedule_retune = lambda *_a: -1
    assert rad.quick_retune(5800e6) is False
    assert rad._fc == 5740e6


def test_quick_retune_source_assigns_fc_only_after_ok():
    src = inspect.getsource(B.BladeRF.quick_retune)
    assert src.index("if rc < 0:") < src.index("self._fc = float(hz)")
    assert "return False" in src
