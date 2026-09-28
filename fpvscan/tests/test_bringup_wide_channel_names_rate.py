"""bringup.py must name -s when occupancy is wider than the passband."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_occupancy_wider_than_passband_names_sample_rate():
    """A 27 MHz analog VTx at 20 Msps fills the whole window. Raising
    gain_db or threshold_db does not recover the sidebands — the
    decoder then looks dead. The hint has to be `-s`, not another
    trip around the gain sweep.
    """
    assert "top.bandwidth_hz > fs * 0.75" in SRC
    assert "підніми -s" in SRC
    assert "зараз" in SRC[SRC.index("top.bandwidth_hz > fs * 0.75"):]
