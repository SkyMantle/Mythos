"""BladeRF.open must name the xA4 bitstream when FPGA is blank."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_open_fpga_miss_names_hostedxa4():
    """doctor.py already prints the load command. Engine.start() hits
    BladeRF.open first; if that raise() drops hostedxA4.rbf the
    journal only says «FPGA не завантажена» and the operator apt-
    installs libbladerf2 again instead of the bitstream package.
    """
    start = SRC.index("def open(self)")
    end = SRC.index("def _config_stream")
    body = SRC[start:end]
    assert "FPGA не завантажена" in body
    assert "hostedxA4.rbf" in body
    assert "bladerf-fpga-hostedxa4" in body
