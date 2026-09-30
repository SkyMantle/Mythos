"""The Pi install hint must name the shipped web port, and re-runs
must survive an already-created fpv user.
"""
from __future__ import annotations

from pathlib import Path

from fpvscan.config import load


ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "deploy" / "install_pi.sh").read_text(encoding="utf-8")
CFG = load(ROOT / "config.yaml")


def test_install_console_url_matches_shipped_port():
    """ZeroTier bookmarks the URL printed at the end of
    install_pi.sh. If that port drifts from config.yaml `web.port`
    the unit listens on 8080 and the operator hits a dead :80.
    """
    port = int(CFG["web"]["port"])
    assert f":{port}" in INSTALL
    assert "Веб-консоль:" in INSTALL


def test_install_useradd_does_not_abort_on_rerun():
    """The script is `set -euo pipefail`. A second imaging pass
    (or a retry after a pip failure) hits 'user already exists'
    and never reaches `systemctl enable --now`. `|| true` is
    what lets the rest of the script finish.
    """
    assert "set -euo pipefail" in INSTALL
    assert "useradd -r -s /usr/sbin/nologin -G plugdev fpv" in INSTALL
    assert "2>/dev/null || true" in INSTALL
