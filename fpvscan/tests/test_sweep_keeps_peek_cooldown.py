"""«Сканувати» must not reset auto-peek cooldown."""
from __future__ import annotations

from tests.helpers import make_engine


def test_sweep_command_keeps_peeked():
    """#37 Clear wipes `_peeked` so the same bird can be shown again.
    #39 Sweep keeps the hit list. Cooldown is the third piece of
    scan memory: `_maybe_peek` skips a MHz bucket for
    `auto_peek_cooldown_s` (60 s omitted). Clearing it on Sweep
    means every Сканувати after a 5 s peek immediately LOCK-peeks
    the same analog hit — the 400 MHz–6 GHz walk never resumes.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng._peeked[5800] = 1.0
    eng._peeked[5725] = 2.0
    eng.command("sweep")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng._peeked[5800] == 1.0
    assert eng._peeked[5725] == 2.0
