"""A manual lock is a hold, not a rewind of the sweep plan or cooldown."""
from __future__ import annotations

from tests.helpers import make_engine


def test_lock_command_keeps_sweep_index():
    """#49 Sweep from LOCK keeps `_sweep_i`; Clear is the rewind.
    lock() already nulls `_lock_tuned` so the new channel retunes.
    Resetting the plan index on a click-to-hold restarts the
    400 MHz–6 GHz walk after the operator later hits Сканувати —
    a hop-on VTx they already waited past is missed for a full pass.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._sweep_i = 17
    eng.command("lock", freq_hz=5800e6)
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._sweep_i == 17


def test_lock_command_keeps_peek_cooldown():
    """Clicking a hit must not forget that this MHz already peeked.
    After the hold, Sweep (#39) resumes the pass; leftover
    `_peeked.clear()` on lock would auto-peek the same bird the
    moment the operator lets go.
    """
    eng = make_engine()
    eng._peeked[5800] = 1.0
    eng.command("lock", freq_hz=5725e6)
    eng._drain_commands()
    assert eng._peeked == {5800: 1.0}
