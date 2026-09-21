"""LOCK channelizer width must use the same neighbor window as merge."""
from __future__ import annotations

from pathlib import Path

from fpvscan.engine import Engine


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "engine.py").read_text(encoding="utf-8")


def test_lock_bw_neighbor_window_is_the_merge_tolerance():
    """Merge collapses hits within MERGE_TOL_HZ (6 MHz). _lock_bw looks
    up the measured occupancy with a bare `6e6`. If those drift apart, a
    confirmed Raceband neighbor is decoded with the YAML 10 MHz default
    instead of occupancy+2·MERGE_TOL — the picture is two VTx at once.
    """
    assert Engine.MERGE_TOL_HZ == 6e6
    body = SRC[SRC.index("def _lock_bw"):SRC.index("def _start_reader")]
    assert "if abs(d.freq_hz - freq_hz) < 6e6:" in body
    assert "return max(8e6, d.bandwidth_hz +2 * self.MERGE_TOL_HZ)" in body
