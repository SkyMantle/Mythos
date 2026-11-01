"""A sweep hop must consume a click-to-hold queued mid-walk."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_drains_leftover_lock_before_grab():
    """#47 `Engine.command` only `put`s. #80 `_start_reader`
    keeps leftover `_cmd` so AFC recenter does not eat a
    Photo / Сканувати. `_do_sweep` is the other half: it
    *does* `_drain_commands` at the start of each 8 MHz
    step (and again after incrementing `_sweep_i`) and
    returns once `mode == LOCK`. A leftover hop that
    grabs first, or that never drains, makes click-to-hold
    wait out the rest of the 400 MHz–6 GHz walk — the
    operator clicked a bird and the picture never comes.
    Drain-before-grab is the contract: `retune_calls`
    stays 0 and `_sweep_i` is not advanced. `fixed_freq`
    keeps the plan to one step so a missed drain is a
    single grab, not a full pass.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng.command("lock", freq_hz=1280e6)
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 0
        assert eng._sweep_i == 0
        assert eng.state.mode == "LOCK"
        assert eng.state.lock_target == 1280e6
        assert eng.state.auto is False
        assert eng.snapshot()["mode"] == "LOCK"
        assert eng.snapshot()["lock_target"] == 1280e6
        assert eng._cmd.empty()
        assert eng.state.mode != "SWEEP"
        assert eng._lock_tuned is None
        assert eng._reader_thread is None
    finally:
        eng._stop_reader()
