"""Pi imaging must install the unit file before systemd starts it."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "deploy" / "install_pi.sh").read_text(encoding="utf-8")


def test_install_copies_and_reloads_before_enable_now():
    """`enable --now` without a preceding `cp` + `daemon-reload`
    starts whatever unit is already on the image (or fails with
    'Unit fpvscan.service not found'). A re-image then boots the
    old ExecStart / old WorkingDirectory and the operator gets a
    dead console with a green systemctl status. Copy, reload, then
    enable — in that order.
    """
    assert "cp deploy/fpvscan.service /etc/systemd/system/" in INSTALL
    assert "systemctl daemon-reload" in INSTALL
    assert "systemctl enable --now fpvscan" in INSTALL
    assert INSTALL.index("cp deploy/fpvscan.service") < INSTALL.index(
        "systemctl daemon-reload")
    assert INSTALL.index("systemctl daemon-reload") < INSTALL.index(
        "systemctl enable --now fpvscan")
