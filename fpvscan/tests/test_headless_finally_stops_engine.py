"""Headless must release the radio on --secs expiry and Ctrl-C."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "headless.py").read_text(encoding="utf-8")


def test_headless_stops_the_engine_in_finally():
    """bladeRF is exclusive. `--secs` and KeyboardInterrupt used to
    fall out of the loop and leave Engine._run + the IQ reader holding
    the USB handle, so the next `run.py` / doctor / bringup failed
    with 'device busy' until a reboot. `finally: eng.stop()` is what
    joins the worker and closes the source on every exit path.
    """
    assert "eng.start()" in SRC
    assert "except KeyboardInterrupt" in SRC
    finally_at = SRC.index("finally:")
    tail = SRC[finally_at:]
    assert "eng.stop()" in tail
    # A stop only on the happy path (after the while) is not enough —
    # KeyboardInterrupt skips it. The call must sit inside finally.
    assert SRC.index("except KeyboardInterrupt") < finally_at
