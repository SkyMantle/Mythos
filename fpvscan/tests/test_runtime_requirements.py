"""Runtime deps the decode path imports; dropping one fails on the Pi, not CI."""
from __future__ import annotations

from pathlib import Path


REQ = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
    encoding="utf-8")


def test_decode_and_console_packages_are_listed():
    """scipy is channelize(fast=False) and deemphasis; Pillow is the WebP
    stream; PyYAML is config.load. A Pi image that only pip-installed
    numpy boots, sweeps, and then ImportErrors on the first LOCK frame.
    """
    assert "numpy>=1.26" in REQ
    assert "scipy>=1.11" in REQ
    assert "Pillow>=10.0" in REQ
    assert "fastapi>=0.110" in REQ
    assert "uvicorn[standard]>=0.29" in REQ
    assert "PyYAML>=6.0" in REQ
