"""doctor.py's FPGA-miss hint must name the xA4 bitstream, not just «not configured»."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "doctor.py").read_text(encoding="utf-8")


def test_unconfigured_fpga_names_the_xa4_bitstream():
    """open() already refuses an empty FPGA. doctor is what the
    operator runs first: a generic «не залита» without
    `bladeRF-cli -l hostedxA4.rbf` / `apt install
    bladerf-fpga-hostedxa4` sends them into a Windows Nuand
    reinstall that does not flash the Pi bitstream.
    """
    assert "bladerf_is_fpga_configured" in SRC
    assert "bladeRF-cli -l hostedxA4.rbf" in SRC
    assert "bladerf-fpga-hostedxa4" in SRC
    assert "не залита" in SRC
