"""Pi boot order: the unit must start after the network and keep USB alive."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE = (ROOT / "deploy" / "fpvscan.service").read_text(encoding="utf-8")
INSTALL = (ROOT / "deploy" / "install_pi.sh").read_text(encoding="utf-8")


def test_unit_starts_after_network_and_is_niced():
    """The console is reached over ZeroTier. Starting before
    network-online leaves the operator hitting a dead port after reboot.
    Nice=-5 keeps the USB reader ahead of apt/unattended-upgrades on the
    Pi; PYTHONUNBUFFERED is how USB errors show up in journalctl instead
    of sitting in a libc buffer until the unit is killed.
    """
    assert "After=network-online.target" in SERVICE
    assert "Wants=network-online.target" in SERVICE
    assert "Type=simple" in SERVICE
    assert "WantedBy=multi-user.target" in SERVICE
    assert "Nice=-5" in SERVICE
    assert "Environment=PYTHONUNBUFFERED=1" in SERVICE


def test_install_pulls_libbladerf_and_the_xa4_fpga():
    """libbladerf2 without the hosted xA4 bitstream enumerates the board
    and then Engine.open raises 'FPGA не завантажена'. set -euo is what
    stops the unit being enabled on a half-installed image.
    """
    assert "set -euo pipefail" in INSTALL
    assert "libbladerf2" in INSTALL
    assert "bladerf-fpga-hostedxa4" in INSTALL
    assert "bladeRF-cli -e info" in INSTALL
