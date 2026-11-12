"""A LOCK field must not replace the cooldown dict `_maybe_peek` already writes."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_peeked_identity_on_later_field():
    """#64 Sweep / lock keep `_peeked` contents. #68 expiry /
    #70 peek keep other MHz. #77 `_start_reader` keeps the
    table on a channel change. #83 a later field keeps the
    *contents* (must not `_peeked.clear()` — the 8 s glance
    would re-peek the same analog hit). The missing half:
    a later field on the same bird must not
    `self._peeked = {}` / `self._peeked = dict(self._peeked)`
    (copy-paste from `__init__` or a "fresh cooldown").
    `_do_lock` never consults the dict object identity;
    `_maybe_peek` *does* write `self._peeked[key] = now`
    and Clear *does* `self._peeked.clear()`. A leftover
    replacement still looks like a live hold (`mode=LOCK`,
    `lock_target`, `_ring`, `_lock_tuned`, reader alive,
    `retune_calls==1`, the 0.4 Hz fs-tolerance case still
    passes, #83's contents check still passes on a shallow
    copy) while the next confirm_hits cycle writes cooldown
    onto a dict Clear / `_maybe_peek` no longer share —
    the 400 MHz–6 GHz walk re-peeks the same bird instead
    of waiting `auto_peek_cooldown_s`. First field starts
    the reader; leftover is the same dict after that, and
    a later write must land on *that* dict.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._peeked[5800] = 1.0
    eng._peeked[1280] = 2.0
    leftover = eng._peeked
    try:
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng._peeked is leftover
        eng._do_lock()
        assert eng._peeked is leftover
        assert leftover[5800] == 1.0
        assert leftover[1280] == 2.0
        assert leftover == {5800: 1.0, 1280: 2.0}
        eng._peeked[3333] = 9.0
        assert leftover[3333] == 9.0
        assert 3333 in leftover
        assert eng._peeked[3333] == leftover[3333]
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        assert eng.state.mode == "LOCK"
        assert eng.state.lock_target == 5800e6
    finally:
        eng._stop_reader()
