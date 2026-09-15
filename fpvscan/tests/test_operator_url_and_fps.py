"""Operator-facing URL and console FPS window."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_prints_loopback_when_bound_to_all_interfaces():
    """config.yaml binds 0.0.0.0. Printing that host as the URL sends
    the operator's laptop browser to an address it cannot open; run.py
    must rewrite wildcard binds to 127.0.0.1 in the banner.
    """
    src = (ROOT / "run.py").read_text(encoding="utf-8")
    assert 'shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host' in src
    assert "http://{shown}:{port}" in src


def test_console_fps_uses_a_three_second_window():
    """showFrame pushes timestamps and displays length/3. If the window
    stays 3000 ms but the divisor becomes 1, the OSD reads 3× the real
    rate and a stalled LOCK looks healthy. Both ends of the contract
    have to move together.
    """
    html = (ROOT / "fpvscan" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    assert "Date.now() - 3000" in html
    assert "fpsHist.length / 3" in html
