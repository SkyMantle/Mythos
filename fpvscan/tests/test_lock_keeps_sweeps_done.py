"""Click-to-hold must not rewind the console pass counter."""
from __future__ import annotations

from tests.helpers import make_engine


def test_lock_command_keeps_sweeps_done():
    """#65 Sweep / Clear keep `sweeps_done`. #64 lock() already keeps
    `_sweep_i` so resume does not restart at 400 MHz. The pass
    counter is the other half of that walk: `snapshot()["sweeps_done"]`
    is `#s-sweeps` on the console. Zeroing it on a hit-row click
    after a finished 400 MHz–6 GHz walk makes pass 3 look like boot,
    and the operator waits for a cycle that already finished.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng.state.sweeps_done = 3
    eng.command("lock", freq_hz=5800e6)
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.sweeps_done == 3
    assert eng.snapshot()["sweeps_done"] == 3
