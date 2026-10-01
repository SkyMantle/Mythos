"""Pi imaging: the app tree must be owned by fpv before the venv is created."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "deploy" / "install_pi.sh").read_text(encoding="utf-8")


def test_install_chowns_app_before_creating_the_venv():
    """`set -euo` plus `sudo -u fpv python3 -m venv` fails immediately
    when /opt/fpvscan is still root-owned after rsync. The unit then
    never reaches `systemctl enable --now`, and the operator images a
    Pi with no console. chown must run first; creating the venv as
    root and chowning later leaves root-owned .venv files that pip
    as fpv cannot upgrade.
    """
    assert "set -euo pipefail" in INSTALL
    assert "chown -R fpv:fpv $APP" in INSTALL
    assert "sudo -u fpv python3 -m venv $APP/.venv" in INSTALL
    assert INSTALL.index("chown -R fpv:fpv $APP") < INSTALL.index(
        "sudo -u fpv python3 -m venv $APP/.venv")
    assert INSTALL.index("sudo -u fpv python3 -m venv") < INSTALL.index(
        "pip install")
