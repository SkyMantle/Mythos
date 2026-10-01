"""bench_retune.py must open the radio in metadata/quick-tune mode."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bench_retune.py").read_text(encoding="utf-8")


def test_bench_opens_bladerf_with_use_meta():
    """Section 2 times `quick_retune`. That path needs the SC16
    META stream (`use_meta=True`) so `schedule_retune(..., RETUNE_NOW)`
    actually runs. The BladeRF default is `use_meta=False`. Opening
    without it makes prime_quick_tune / quick_retune look broken on
    a healthy xA4, and the operator freezes `quick_tune: false` in
    config.yaml — every sweep step then pays a full PLL settle.
    """
    assert "BladeRF(device=a.device, lib_path=a.lib, use_meta=True)" in SRC
    assert "prime_quick_tune" in SRC
    assert "quick_retune" in SRC
