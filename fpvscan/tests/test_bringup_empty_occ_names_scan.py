"""bringup occupancy miss must send the operator to --scan, not a higher threshold."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_empty_occupancy_names_scan_and_rx1():
    """`find_occupied` at 6 dB / 1 MHz is already looser than the
    engine. An empty list means the LO is not on the bird, not
    that the threshold is 1 dB too high. The hint must still
    name `--scan` and RX1 — otherwise the stand session becomes
    'raise threshold_db' and a 20 MHz analog VTx stays invisible.
    """
    start = SRC.index('print("\\n=== 2. спектр ===")')
    end = SRC.index('print("\\n=== 3. рядкова частота ===")')
    body = SRC[start:end]
    assert "if not occ:" in body
    assert "6 дБ" in body
    assert "--scan" in body
    assert "RX1" in body
    assert "return" in body
