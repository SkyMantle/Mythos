"""Pi deploy: a USB drop must restart the scanner, and the service
must run the venv interpreter as a user that can open the bladeRF.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE = (ROOT / "deploy" / "fpvscan.service").read_text(encoding="utf-8")
INSTALL = (ROOT / "deploy" / "install_pi.sh").read_text(encoding="utf-8")


def test_systemd_unit_restarts_and_opens_the_radio():
    """Restart=no is how a single USB glitch leaves the Pi serving a
    dead console until someone ssh's in. plugdev is the udev group
    libbladeRF needs; without it open() fails as user fpv.
    """
    assert "Restart=always" in SERVICE
    assert "RestartSec=3" in SERVICE
    assert "User=fpv" in SERVICE
    assert "SupplementaryGroups=plugdev" in SERVICE
    assert "WorkingDirectory=/opt/fpvscan" in SERVICE
    assert ("ExecStart=/opt/fpvscan/.venv/bin/python "
            "/opt/fpvscan/run.py -c /opt/fpvscan/config.yaml") in SERVICE


def test_install_pi_uses_venv_and_enables_the_unit():
    """system python + a forgotten pip install boots a unit that
    ImportErrors in a loop. rsync --exclude .venv keeps the laptop
    venv off the aarch64 image.
    """
    assert "useradd -r -s /usr/sbin/nologin -G plugdev fpv" in INSTALL
    assert "rsync -a --exclude .venv" in INSTALL
    assert "pip install -q -r $APP/requirements.txt" in INSTALL
    assert "cp deploy/fpvscan.service /etc/systemd/system/" in INSTALL
    assert "systemctl enable --now fpvscan" in INSTALL
