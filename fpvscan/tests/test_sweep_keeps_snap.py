"""«Сканувати» must not eat a pending Photo."""
from __future__ import annotations

from tests.helpers import make_engine


def test_sweep_command_keeps_pending_photo():
    """#65 snapshot in SWEEP sets `_snap` without a mode==LOCK
    guard. Sweep already `_stop_reader()`s (#24) and clears
    lock_target. Clearing `_snap` on Сканувати would drop a Photo
    armed during the hold — resume-scan then click-to-hold would
    not write the WebP. The flag is consumed only when a LOCK
    frame exists (#9).
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._snap = True
    eng.command("sweep")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng._snap is True
    assert eng.events.empty()
