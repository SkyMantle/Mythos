"""Sweep from LOCK must resume the plan, not rewind it."""
from __future__ import annotations

from tests.helpers import make_engine


def test_sweep_command_preserves_sweep_index():
    """Clear is the rewind (`_sweep_i = 0`). Sweep from LOCK is
    'go look for the next bird'. Resetting the index on Sweep
    restarts the 400 MHz–6 GHz walk and can miss a hop-on / hop-off
    VTx for a full pass (~seconds) after the operator already
    waited through most of the band.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng._sweep_i = 17
    eng._handle_command("sweep", {})
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng.state.auto is False
    assert eng._sweep_i == 17
