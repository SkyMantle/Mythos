"""Explicit sweep step must not fall back to the auto formula."""
from __future__ import annotations

from tests.helpers import make_engine


def test_explicit_step_hz_is_used_instead_of_auto():
    """`step_hz: 0` means auto. Operators set a real step to overlap a
    known VTx. If the `or auto` short-circuit regresses, a 10 MHz step
    silently becomes ~8 MHz (at 20 Msps) and the scan grid is wrong.
    """
    eng = make_engine()
    eng.cfg["scan"]["step_hz"] = 10e6
    eng.cfg["scan"]["priority_bands"] = False
    plan = eng._sweep_plan()
    assert len(plan) >= 2
    assert plan[1] - plan[0] == 10e6
    auto = max(20e6 * 0.25, 20e6 * 0.9 - 20e6 / 2)
    assert auto != 10e6
    assert plan[1] - plan[0] != auto
