"""A LOCK field must not drop the channel the operator is holding."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_lock_target_on_later_field():
    """#21 lock() writes `lock_target`; #63 click-to-hold cancels
    auto so the leftover peek timer cannot steal the hold. #79
    `_start_reader` keeps `lock_target` on AFC recenter / the 1 Hz
    fs restart. #81 `_do_sweep` keeps the glance `_maybe_peek`
    just armed. #85 a field keeps `auto` / `auto_until` so a
    glance does not become a hold. The missing half: a later
    field on a *manual* hold (`auto=False`, default
    `auto_until=0`) must not `lock_target=None`. Peek expiry
    *does* null it (with `mode="SWEEP"`) when `auto and
    time.time() >= auto_until`. A leftover expiry that drops
    the `auto and` guard fires on field 1 (`time.time() >= 0`)
    even though this is a click-to-hold — `_run` then sees
    `mode==LOCK and not lock_target` and falls through to
    `_do_sweep`. The picture drops, a live recording stops with
    the lock-command path, and the 400 MHz–6 GHz walk resumes
    while the operator is still watching. First field starts
    the reader (hop *does* write `_lock_tuned`); leftover is
    the same `lock_target` after that.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    assert eng.state.auto is False
    assert eng.state.auto_until == 0.0
    try:
        eng._do_lock()
        leftover = eng.state.lock_target
        assert leftover == 5800e6
        assert leftover is not None
        assert eng.state.mode == "LOCK"
        eng._do_lock()
        assert eng.state.lock_target == leftover
        assert eng.state.lock_target == 5800e6
        assert eng.state.lock_target is not None
        assert eng.snapshot()["lock_target"] == 5800e6
        assert eng.state.mode == "LOCK"
        assert eng.state.auto is False
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
