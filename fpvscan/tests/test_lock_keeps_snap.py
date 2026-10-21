"""Click-to-hold must not eat a pending Photo."""
from __future__ import annotations

from tests.helpers import make_engine


def test_lock_command_keeps_pending_photo():
    """#65 snapshot in SWEEP stays pending; #66 lock() keeps the
    hit list / pass counter and (#21) nulls `_lock_tuned` / AFC.
    Clearing `_snap` on a hit-row click would drop a Photo armed
    during the previous peek — the hold would decode a picture
    but never write the WebP, and the operator would mash Photo
    again after they already asked for a still.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._snap = True
    eng.command("lock", freq_hz=5800e6)
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._snap is True
    assert eng.events.empty()
