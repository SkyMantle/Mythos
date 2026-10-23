"""Click-to-hold must not reset the LOCK waterfall cadence."""
from __future__ import annotations

from tests.helpers import make_engine


def test_lock_command_keeps_lock_n():
    """#63 lock() cancels auto and #21 nulls `_lock_tuned` / AFC
    so the hold retunes. #66 kept detections / `sweeps_done`.
    `_lock_n` is the waterfall half of the same click: leftover
    `spectrum_every=8` cadence (#58) must survive so the first
    field is not a free 4096×4 FFT (`1%8==1`). Zeroing it on
    every hit-row click would spend Pi idle_ms on a paint the
    operator did not ask for. Leftover skip after a hold that
    stopped mid-cycle is current main.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._lock_n = 7
    eng.command("lock", freq_hz=5800e6)
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.auto is False
    assert eng._lock_n == 7
