"""Headless must say when the engine never read the radio."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "headless.py").read_text(encoding="utf-8")


def test_zero_spectra_names_the_radio_not_the_browser():
    """headless exists to split 'USB is dead' from 'the console WS
    is stuck'. A silent 45 s run with n_spec==0 looks like a decoder
    bug. The line has to say the engine never produced a spectrum,
    so the operator stays off uvicorn and index.html.
    """
    assert "if n_spec == 0:" in SRC
    assert "Жодного спектра" in SRC
    assert "нижче веба" in SRC
    assert "Рушій не читає з плати" in SRC[SRC.index("if n_spec == 0:"):]
