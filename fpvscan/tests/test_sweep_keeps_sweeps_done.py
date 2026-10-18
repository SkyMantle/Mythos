"""«Сканувати» must not rewind the console pass counter."""
from __future__ import annotations

from tests.helpers import make_engine


def test_sweep_command_keeps_sweeps_done():
    """#49 Sweep from LOCK keeps `_sweep_i` so resume does not
    restart at 400 MHz. `sweeps_done` is the other half of that
    walk: `snapshot()["sweeps_done"]` is the console pass
    counter. Zeroing it on Сканувати after a hold makes a 3-pass
    walk look like pass 0 — the operator thinks the radio just
    booted and waits for a full 400 MHz–6 GHz cycle that already
    finished. Clear rewinds `_sweep_i`; it also leaves this
    counter alone.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.sweeps_done = 3
    eng.command("sweep")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng.state.sweeps_done == 3

    eng.command("clear")
    eng._drain_commands()
    assert eng.state.sweeps_done == 3
