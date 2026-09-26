"""bringup.py must not open the bladeRF with no frequency."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_without_f_scan_or_file_errors():
    """`python scripts/bringup.py` with no args used to fall into
    BladeRF() and tune whatever leftover Hz the board had. On a
    stand that looks like a dead VTx. The guard must demand -f,
    --scan, or --file before opening the radio.
    """
    assert "if a.file:" in SRC
    assert "if not a.freq and not a.scan:" in SRC
    assert "ap.error" in SRC
    assert "-f" in SRC and "--scan" in SRC and "--file" in SRC
