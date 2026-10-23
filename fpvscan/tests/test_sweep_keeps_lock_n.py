"""Сканувати must not reset the LOCK waterfall cadence."""
from __future__ import annotations

from tests.helpers import make_engine


def test_sweep_command_keeps_lock_n():
    """#24 Sweep joins the IQ reader; #65 keeps `sweeps_done`.
    `_lock_n` is not the pass counter — it is how often LOCK
    paints the 4096×4 waterfall (#58 / #66). Zeroing it on
    Сканувати would make the next glance/click emit on the
    first field (`1%8==1`) and spend idle_ms on a FFT the
    walk did not need. Leftover cadence is current main.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._lock_n = 7
    eng.command("sweep")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng._lock_n == 7
