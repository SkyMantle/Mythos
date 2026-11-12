"""A sweep hop must not replace the cooldown dict `_maybe_peek` already writes."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_peeked_identity():
    """#64 Sweep *command* keeps `_peeked` contents (Clear is
    the wipe). #77 a hop keeps the *contents* (must not
    `_peeked.clear()` on each 8 MHz step). #83 a later LOCK
    field keeps contents; this run's LOCK half keeps dict
    identity. The missing hop analog: an 8 MHz hop must not
    `self._peeked = {}` / `self._peeked = dict(self._peeked)`
    (copy-paste from `__init__` or a "fresh cooldown").
    After a contents-preserving shallow copy #77 still
    passes, but `_maybe_peek` / Clear no longer share the
    dict the hop left behind — the next confirm_hits cycle
    writes cooldown onto a table nobody else holds, so the
    same analog hit LOCK-peeks again instead of waiting
    `auto_peek_cooldown_s` and the 400 MHz–6 GHz walk never
    resumes. `fixed_freq` keeps this to one grab; zeros IQ
    so occupancy/inspect stay out of the path. Leftover is
    the same dict after the hop, and a later write must
    land on *that* dict.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng._peeked[5800] = 1.0
    eng._peeked[1280] = 2.0
    leftover = eng._peeked
    try:
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng._peeked is leftover
        assert leftover[5800] == 1.0
        assert leftover[1280] == 2.0
        assert leftover == {5800: 1.0, 1280: 2.0}
        eng._peeked[3333] = 9.0
        assert leftover[3333] == 9.0
        assert 3333 in leftover
        assert eng._peeked[3333] == leftover[3333]
        assert eng._lock_tuned is None
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
