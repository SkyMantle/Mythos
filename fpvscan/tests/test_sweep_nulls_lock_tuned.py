"""A sweep hop must null leftover `_lock_tuned` before the next grab."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_nulls_lock_tuned_before_hop():
    """#21 lock() nulls `_lock_tuned` so a channel switch retunes.
    #66 `_maybe_peek` does the same. #63 Clear keeps it so
    Очистити does not join the reader. Sweep command (#24) only
    `_stop_reader()`s — leftover `_lock_tuned` from the previous
    hold can still match `want` if the same MHz peeks again after
    a short join (#10) left the ring alive. `_do_sweep` must null
    it on every hop so `_do_lock` cannot skip `_start_reader` and
    decode stale IQ from the last hold.
    """
    eng = make_engine()
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6
    eng.state.mode = "SWEEP"
    eng._lock_tuned = 5800e6
    eng._do_sweep()
    assert eng.state.mode == "SWEEP"
    assert eng._lock_tuned is None
    assert eng.state.tuned_hz != 0.0
    assert eng.state.sweep_pos_hz == eng.state.tuned_hz
