"""Photo during SWEEP must stay pending for the next LOCK frame."""
from __future__ import annotations

from tests.helpers import make_engine


def test_snapshot_in_sweep_stays_pending():
    """#44 rec_start in SWEEP is rejected (`спершу стань на канал`)
    because there is no frame to mux. Photo is the other capture
    button: the command only sets `_snap`, and `_do_lock` consumes
    it when a frame exists (#9 / #49). Checking `mode == LOCK` on
    snapshot would silently drop Photo during a scan — the next
    peek/lock would not save. The flag must survive Sweep, and
    must not emit an error notice.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng.command("snapshot")
    eng._drain_commands()
    assert eng._snap is True
    assert eng.state.mode == "SWEEP"
    assert eng.events.empty()
