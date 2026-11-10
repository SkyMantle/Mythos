"""A sweep hop must not spawn or replace the engine worker thread."""
from __future__ import annotations

import threading

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_worker_thread():
    """`start()` *does* `self._thread = Thread(target=_run)` so
    one worker owns the USB handle. `stop()` *does* join
    that handle. #93 a later LOCK field keeps `_thread` so
    a second `_run` does not race `src.read`; the start
    gate does not look at `_thread`. The missing half: an
    8 MHz hop must not `self.start()` / `self._thread =
    Thread(target=self._run)` (copy-paste from `start()`)
    and must not `_thread = None` (copy-paste from
    `__init__`). `_do_sweep` never consults `_thread`; it
    *is* what `_run` is running on. A leftover new worker
    still looks like a live walk (`mode=SWEEP`, reader
    down, `retune_calls==1`) while a second `_run` opens
    the same handle, races `_grab` / `src.read`, and Stop
    cannot join the original worker. Direct `_do_sweep`
    must keep the sentinel identity, must not start it,
    and must not spawn an IQ reader. `fixed_freq` keeps
    this to one grab; zeros IQ so occupancy/inspect stay
    out of the path.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    sentinel = threading.Thread(target=lambda: None)
    eng._thread = sentinel
    try:
        leftover = eng._thread
        assert leftover is sentinel
        assert leftover.is_alive() is False
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng._thread is leftover
        assert eng._thread is sentinel
        assert leftover.is_alive() is False
        assert leftover.is_alive() is not True
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop.set()
        spawned = eng._thread
        if spawned is not None and spawned is not sentinel and spawned.is_alive():
            spawned.join(timeout=3)
        eng._stop_reader()
