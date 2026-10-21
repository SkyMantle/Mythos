"""Очистити must not eat a pending Photo."""
from __future__ import annotations

from tests.helpers import make_engine


def test_clear_command_keeps_pending_photo():
    """#65 snapshot in SWEEP stays pending; #37 Clear wipes the
    hit list / cooldown / `_sweep_i` and (#65) leaves auto alone.
    Clearing `_snap` on Очистити would drop a Photo armed during
    the hold — the next field would decode but never write the
    WebP. The flag is consumed only when a LOCK frame exists (#9).
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._snap = True
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._snap is True
    assert eng.events.empty()
