"""record.py must release the exclusive bladeRF handle after the grab."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "record.py").read_text(encoding="utf-8")


def test_record_closes_before_writing_the_cf32():
    """`BladeRF.open` takes the board exclusively. Recording 3 s at
    40 Msps is ~960 MB; writing that after a forgotten `close()`
    keeps doctor / bringup / `run.py` on 'device busy' for the
    whole dump. Close must sit after `read(n)` and before
    `write_capture`.
    """
    grab = SRC.index("iq = src.read(n)")
    close = SRC.index("src.close()")
    dump = SRC.index("write_capture(a.out")
    assert grab < close < dump
