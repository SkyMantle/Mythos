"""Shipped auto-step (step_hz=0) at 35 Msps / 10 MHz — the production sweep geometry."""
from __future__ import annotations

from pathlib import Path
from queue import Queue

from fpvscan.config import load
from fpvscan.engine import Engine
from tests.helpers import MockSource


ROOT = Path(__file__).resolve().parents[1]
CFG = load(ROOT / "config.yaml")


def test_shipped_auto_step_keeps_a_channel_above_the_analog_gate():
    """step_hz=0 is the shipped path. Helpers in #5 used 20 Msps / 20 MHz
    and got an 8 MHz hop. On the bench YAML (35 Msps / 10 MHz) the hop is
    26.5 MHz. Shrinking it toward `fs` parks a 10–20 MHz VTx on the
    stitch: each hop sees only a sliver and `_do_sweep` drops it at the
    2.5 MHz analog gate before INSPECT ever runs.
    """
    scan = CFG["scan"]
    fs = float(scan["sample_rate"])
    ch = float(scan["channel_bw_hz"])
    assert scan["step_hz"] == 0
    assert fs == 35e6
    assert ch == 10e6

    src = MockSource(fs=fs)
    eng = Engine(src, CFG, Queue())
    plan = eng._sweep_plan()
    step = plan[1] - plan[0]
    want = max(fs * 0.25, fs * 0.9 - ch / 2)
    assert step == want
    assert step == 26.5e6

    overlap = fs * 0.9 - step
    assert overlap == ch / 2
    # Channel of the configured width sitting on a hop stitch is still
    # wider than the 2.5 MHz analog gate in each hop.
    visible = (overlap + ch) / 2
    assert visible > 2.5e6
    assert plan[0] == float(scan["start_hz"]) + fs / 2
    assert plan[-1] < float(scan["stop_hz"])
