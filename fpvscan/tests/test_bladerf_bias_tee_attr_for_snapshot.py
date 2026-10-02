"""Console bias-tee state is `src.bias_tee`, not a private flag."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLADE = (ROOT / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")
ENGINE = (ROOT / "fpvscan" / "engine.py").read_text(encoding="utf-8")


def test_snapshot_reads_public_bias_tee_attribute():
    """`snapshot()['bias_tee']` is `getattr(src, 'bias_tee', False)`.
    BladeRF must keep that public attribute in sync with the last
    successful `set_bias_tee`. A private `_bias` (or only
    `get_bias_tee()`) makes every heartbeat paint the button off, and
    the operator clicks it 'on' — a second 4.5 V request onto an LNA
    that is already powered, then a gain-offset double-cut that
    buries the VTx under the noise floor.
    """
    assert 'getattr(self.src, "bias_tee"' in ENGINE
    assert "self.bias_tee = bias_tee" in BLADE
    assert "self.bias_tee = bool(on)" in BLADE
    assert "self.bias_tee = False" in BLADE
