"""SWEEP must pass YAML occupancy knobs into find_occupied.

The engine fallback for min_bw_hz is 4 MHz. A real bench VTx measured
4.3 MHz and vanished whenever the YAML 2 MHz floor was ignored.
"""
from __future__ import annotations

from unittest.mock import patch

from tests.helpers import make_engine


def test_sweep_forwards_yaml_occupancy_knobs():
    """Hardcoding threshold/min_bw/dc_notch would silently drop a weak VTx."""
    seen = {}
    eng = make_engine()
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6
    eng.cfg["scan"]["threshold_db"] = 5.0
    eng.cfg["scan"]["min_bw_hz"] = 2e6
    eng.cfg["scan"]["dc_notch_hz"] = 200e3

    def capture(_psd, _center, _fs, **kw):
        seen.update(kw)
        return []

    with patch("fpvscan.engine.spectrum.find_occupied", side_effect=capture):
        eng._do_sweep()

    assert seen["threshold_db"] == 5.0
    assert seen["min_bw_hz"] == 2e6
    assert seen["dc_notch_hz"] == 200e3


def test_analog_bw_floor_is_below_the_bench_4mhz_vtx():
    """Comment in _do_sweep: a 4.3 MHz occupier was rejected by a 4 MHz
    floor. The analog gate must stay at 2.5–35 MHz, not the old 4 MHz.
    """
    import inspect

    from fpvscan.engine import Engine

    src = inspect.getsource(Engine._do_sweep)
    assert "2.5e6" in src
    assert "35e6" in src
