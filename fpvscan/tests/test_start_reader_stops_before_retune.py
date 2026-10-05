"""Channel switch must join the old IQ reader before the next USB hop."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_start_reader_stops_old_thread_before_retune():
    """LOCK `_start_reader` is the only path that hops while a reader
    may still be in `sync_rx`. Sweep joins in the command handler
    (#24); a channel click / AFC recenter does not. Starting the new
    `retune_and_read` while the old thread still owns the handle is
    the same dual-access that drops the picture after a short join.
    """
    src = inspect.getsource(Engine._start_reader)
    assert src.index("self._stop_reader()") < src.index("retune_and_read")


def test_start_reader_resets_decode_state_after_the_hop():
    """Keeping the previous `DecodeState` / `_lock_dec` across a hop
    feeds the new VTx the old line period. Tracking then asks for the
    wrong field length and `_lock_dec != dec` may not fire if the
    measured BW happens to match.
    """
    src = inspect.getsource(Engine._start_reader)
    assert src.index("retune_and_read") < src.index("DecodeState()")
    assert "self._lock_dec = None" in src
    assert src.index("retune_and_read") < src.index("self._lock_tuned = want")
