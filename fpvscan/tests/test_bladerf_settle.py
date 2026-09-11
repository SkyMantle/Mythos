"""bladeRF retune settle is microseconds, not the base-class n//4 dump."""
from __future__ import annotations

from pathlib import Path

from fpvscan.sdr import bladerf as B


def test_retune_dumps_settle_us_not_a_quarter_of_the_capture():
    """Base SdrSource dumps n//4 to let a generic PLL settle. bladeRF
    already knows settle_us (shipped 500 µs). Using n//4 on a 80 ms
    LOCK grab would throw away 20 ms of analog video every retune.
    """
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert "skip = max(2048, int(self._fs * self.settle_us * 1e-6))" in src
    assert "self.read(n // 4)" not in src


def test_quick_tune_fast_path_settles_20_microseconds():
    """After a stored profile reload, 20 µs is enough. A millisecond
    sleep here would dominate the 400 MHz–6 GHz sweep.
    """
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert "self.read(max(1024, int(self._fs * 20e-6)))" in src
