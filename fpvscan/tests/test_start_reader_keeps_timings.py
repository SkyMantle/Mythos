"""A LOCK hop / AFC recenter must not wipe the console timing ribbon."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_timings():
    """#79 hop keeps `_acc`. `snapshot()["timings_ms"]` is the
    rolling stage averages (`channelize` / `demod_fm` / `decode`
    / `encode`) the operator uses to see Pi idle_ms pressure.
    A leftover `_timings.clear()` on AFC recenter or the 1 Hz
    fs restart (#2) flashes the ribbon to empty on every VTx
    drift past `afc_recenter_frac` — it looks like the pipeline
    stalled even though the hold is still live. The hop already
    reseeds DecodeState (#50); it must not write the averages.
    """
    eng = make_engine()
    timings = {"channelize": 12.0, "decode": 40.0}
    eng._timings = timings
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._timings is timings
        assert eng._timings == {"channelize": 12.0, "decode": 40.0}
        assert eng.snapshot()["timings_ms"] == {"channelize": 12.0, "decode": 40.0}
        assert eng._timings != {}
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
