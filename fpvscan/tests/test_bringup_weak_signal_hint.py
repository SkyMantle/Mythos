"""bringup.py must name the antenna/RX1/freq when the peak is in the noise."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_peak_below_minus_45_names_antenna_rx1_and_freq():
    """A −50 dBFS peak after the gain sweep is almost never 'raise
    gain again'. The VTx is off, the antenna is in TX, or -f is
    the wrong megahertz. If this hint drops, the operator walks
    gain_db into ADC clip on an empty band and then chases a
    decoder bug.
    """
    assert "level(iq)[0] < -45" in SRC
    assert "сигналу майже немає" in SRC
    assert "антена" in SRC
    assert "RX1" in SRC
    assert "частота" in SRC
