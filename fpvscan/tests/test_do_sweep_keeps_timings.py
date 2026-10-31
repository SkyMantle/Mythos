"""A sweep hop must not wipe the console timing ribbon after a glance."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_timings():
    """#80 `_start_reader` keeps `_timings` across AFC recenter /
    the 1 Hz fs restart. `snapshot()["timings_ms"]` is the rolling
    stage ribbon the operator uses to see Pi idle_ms pressure.
    Commands / expiry do not write it. The sweep hop is the
    missing half: after a glance the walk resumes one 8 MHz
    step at a time. A leftover `_timings.clear()` on every hop
    flashes the ribbon to empty — it looks like the pipeline
    stalled even though the next peek is still armed. The hop
    already nulls `_lock_tuned` (#68); it must not write the
    averages. `fixed_freq` keeps this to one grab.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    timings = {"channelize": 12.0, "decode": 40.0}
    eng._timings = timings
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._timings is timings
        assert eng._timings == {"channelize": 12.0, "decode": 40.0}
        assert eng.snapshot()["timings_ms"] == {"channelize": 12.0, "decode": 40.0}
        assert eng._timings != {}
    finally:
        eng._stop_reader()
