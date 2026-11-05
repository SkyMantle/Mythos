"""A LOCK field must not copy Sweep's leftover `_lock_tuned = None`."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_lock_tuned_on_later_field():
    """#68 `_do_sweep` *does* `_lock_tuned = None` on every 8 MHz hop
    (and again after `_inspect`) so the next glance retunes.
    #74 commands / peek-expiry keep the leftover. #50 `_start_reader`
    *does* write `want` on a channel change. The missing half: a
    later field on the same bird must not `_lock_tuned = None`
    before the `self._ring is None or self._lock_tuned != want`
    gate. A leftover null on every decode (copy-paste from Sweep)
    makes `_lock_tuned != want` every field, `_start_reader` kills
    the IQ thread, reseeds `DecodeState()`, and the picture rolls
    even though the VTx never moved. First field starts the reader
    (hop *does* write `want`); leftover is the identity after that.
    `retune_calls` stays 1 — the hop already paid for the first
    `retune_and_read`.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        leftover = eng._lock_tuned
        assert leftover == 5800e6
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng._lock_tuned == leftover
        assert eng._lock_tuned == 5800e6
        assert eng._lock_tuned is not None
        assert src.retune_calls == 1
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
