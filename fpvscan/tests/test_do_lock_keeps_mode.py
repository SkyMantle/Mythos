"""A LOCK field must not flip a click-to-hold back to Sweep."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_mode_on_later_field():
    """#21 lock() writes `mode=LOCK`; Sweep command *does*
    `mode="SWEEP"`. #79 `_start_reader` keeps mode on recenter.
    #81 `_do_sweep` keeps `mode=LOCK` after `_maybe_peek` arms
    a glance. #85 a field keeps auto so a glance does not
    become a hold. The missing half: a later field on a
    *manual* hold (`auto=False`, default `auto_until=0`) must
    not `mode="SWEEP"`. Peek expiry *does* write SWEEP when
    `auto and time.time() >= auto_until`. A leftover expiry
    that drops the `auto and` guard (or always copies the
    Sweep-command write) fires on field 1 — `_run` then sees
    `mode != "LOCK"` and calls `_do_sweep` even though
    `lock_target` is still the bird. The hold dies in one
    step, the IQ reader is stopped, and the console walks
    400 MHz–6 GHz while the operator never pressed
    Сканувати. First field starts the reader; leftover is
    `mode=LOCK` after that, not Sweep.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    assert eng.state.auto is False
    assert eng.state.auto_until == 0.0
    try:
        eng._do_lock()
        leftover = eng.state.mode
        assert leftover == "LOCK"
        assert leftover != "SWEEP"
        assert eng.state.lock_target == 5800e6
        eng._do_lock()
        assert eng.state.mode == leftover
        assert eng.state.mode == "LOCK"
        assert eng.state.mode != "SWEEP"
        assert eng.snapshot()["mode"] == "LOCK"
        assert eng.state.lock_target == 5800e6
        assert eng.state.auto is False
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
        assert eng._ring is not None
    finally:
        eng._stop_reader()
