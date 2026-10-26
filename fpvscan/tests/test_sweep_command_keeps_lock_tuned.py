"""Сканувати must not null leftover `_lock_tuned` — only `_do_sweep` hops do."""
from __future__ import annotations

from tests.helpers import make_engine


def test_sweep_command_keeps_lock_tuned():
    """#24 Sweep joins the IQ reader so `_do_sweep` owns USB.
    #68 `_do_sweep` nulls `_lock_tuned` on every hop so a
    leftover hold cannot skip `_start_reader`. #63 Clear
    *keeps* it so Очистити does not reseed DecodeState.
    The command boundary is the missing half: `_handle_command`
    only `_stop_reader()`s. Nulling tune on Сканувати would
    collide with Clear/LOCK reuse semantics and hide a hop
    that forgot to clear. Leftover `_lock_tuned` after the
    join is current main — the next hop (or a peek whose
    ring is gone) still retunes because `_ring is None`.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng._lock_tuned = 5800e6
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._ring is not None
        eng.command("sweep")
        eng._drain_commands()
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        assert eng.state.auto is False
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()
