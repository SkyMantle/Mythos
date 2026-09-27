"""bringup/record print BladeRF.info() keys; a rename is a bench crash."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")
BRINGUP = (Path(__file__).resolve().parents[1]
           / "scripts" / "bringup.py").read_text(encoding="utf-8")
RECORD = (Path(__file__).resolve().parents[1]
          / "scripts" / "record.py").read_text(encoding="utf-8")


def test_info_keys_match_what_the_bench_scripts_print():
    """`info()` is the first thing bringup.py and record.py print after
    open(). Renaming `fpga_kle` to `fpga_size` (or dropping `serial`)
    is a KeyError on a live xA4 before any IQ is captured — the
    operator thinks the board failed to open.
    """
    body = SRC[SRC.index("def info(self)"):SRC.index("def gain_range")]
    for key in ('"board"', '"serial"', '"fpga_kle"', '"gain_range"'):
        assert key in body
    for script in (BRINGUP, RECORD):
        assert "i['board']" in script
        assert "i['serial']" in script
        assert "i['fpga_kle']" in script
        assert "i['gain_range']" in script
