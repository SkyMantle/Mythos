"""Shipped YAML must keep digital AFC and auto-levels on — they are the #4 path."""
from __future__ import annotations

from pathlib import Path

from fpvscan.config import load


CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_shipped_afc_is_on_with_half_gain():
    """#24 pinned deadband 100 kHz, not the master switch. `afc: false`
    leaves LOCK on the physical LO while a free-running VTx walks out
    of the channelizer — snow on a confirmed analog hit, no extra
    retune to blame. afc_gain=1.0 overshoots into recenter.
    """
    v = CFG["video"]
    assert v["afc"] is True
    assert float(v["afc_gain"]) == 0.5


def test_shipped_auto_levels_is_on():
    """#18 pinned average/sharpen. auto_levels false maps 0.30–1.0
    instead of the 2nd/99th percentiles, so a weak VTx field stays
    grey and the console looks unlocked.
    """
    assert CFG["video"]["auto_levels"] is True
