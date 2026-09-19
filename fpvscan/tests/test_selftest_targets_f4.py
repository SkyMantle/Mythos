"""scripts/selftest.py is the no-hardware CI. It must look at the
default-scene F4 emitter, not a leftover 5.8 GHz constant that no
longer matches SimSource.
"""
from __future__ import annotations

from pathlib import Path

from fpvscan.sdr.sim import default_scene


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "selftest.py").read_text(encoding="utf-8")


def test_selftest_looks_at_the_default_5g8_emitter():
    f4 = next(e for e in default_scene() if "F4" in e.label or e.freq_hz == 5800e6)
    assert f4.freq_hz == 5800e6
    assert "5800e6" in SRC
    assert "retune_and_read(5800e6" in SRC
    assert "find_occupied(psd, 5800e6" in SRC
