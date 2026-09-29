"""install_pi.sh must not abort when the bladeRF is unplugged."""
from __future__ import annotations

from pathlib import Path


INSTALL = (Path(__file__).resolve().parents[1]
           / "deploy" / "install_pi.sh").read_text(encoding="utf-8")


def test_install_continues_when_the_board_is_missing():
    """The script is `set -euo pipefail`. A bare `bladeRF-cli -e info`
    on a Pi that is being imaged before the radio is plugged in
    exits 1, skips venv+systemd, and the next boot has no unit.
    The `|| echo` is what lets the image finish; the operator still
    sees the USB hint. `print rxvga1` is the same trap (`|| true`).
    """
    assert "set -euo pipefail" in INSTALL
    assert "bladeRF-cli -e info ||" in INSTALL
    assert "кабель USB3" in INSTALL
    assert 'bladeRF-cli -e "print rxvga1" >/dev/null 2>&1 || true' in INSTALL
