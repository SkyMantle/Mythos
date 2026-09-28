"""A failed CVBS decode must name rate, clip, and SNR — not ctypes."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_decode_fail_names_rate_clip_and_snr():
    """`fr is None` after a real grab is almost never a missing
    bladeRF.dll. Operators who get a generic traceback reinstall
    libbladeRF while the channel is clipped or the sample rate is
    below Carson. The three-line checklist is the whole point of
    bringup vs the web console.
    """
    assert "if fr is None:" in SRC
    assert "кадр не зібрався" in SRC
    fail = SRC[SRC.index("if fr is None:"):SRC.index("print(f\"  {fr.standard}")]
    assert "підніми -s" in fail
    assert "насиченні" in fail
    assert "С/Ш" in fail
