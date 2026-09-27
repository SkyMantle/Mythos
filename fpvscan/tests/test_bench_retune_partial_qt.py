"""bench_retune.py must tell the operator to leave quick_tune off."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bench_retune.py").read_text(encoding="utf-8")


def test_partial_profiles_name_quick_tune_false():
    """xA4 + some libbladeRF builds prime only a subset of the 400 MHz–
    6 GHz grid. A missing profile falls back to a full PLL grab in
    the middle of a sweep and looks like a random USB stall. The
    script must still say `quick_tune: false` — leaving it true
    after a partial prime is how a stand that 'worked in doctor'
    hangs on the first 5.8 hop.
    """
    assert "if ok < len(freqs):" in SRC
    assert "quick_tune: false" in SRC
    assert "не для всіх частот" in SRC
    assert "xA4" in SRC
