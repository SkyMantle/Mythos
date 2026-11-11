"""A LOCK field must not replace the queue the web already put()s onto."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_command_queue_identity_on_later_field():
    """#47 `Engine.command` only `put`s on `_cmd`. `create_app`
    lock / sweep / snapshot / record / bias-tee / clear *do*
    `engine.command(...)` onto that Queue. `_run` *does*
    `_drain_commands()` between fields. #83 a later field
    keeps the *contents* (must not `_drain_commands()` —
    a Photo / Сканувати / neighbor lock queued mid-hold
    stays until `_run` sees it). #92 keeps the `events`
    Queue identity so `_emit("frame")` still reaches the
    pump. The missing half: a later field on the same bird
    must not `self._cmd = Queue()` (copy-paste from
    `__init__`). `_do_lock` never consults the Queue
    object identity; it does not drain either. A leftover
    replacement still looks like a live hold (`mode=LOCK`,
    `lock_target`, `_ring`, `_lock_tuned`, reader alive,
    `retune_calls==1`, the 0.4 Hz fs-tolerance case still
    passes) while a Photo / Сканувати / click-to-hold
    armed *after* the first field `put`s onto a queue
    `_run` no longer drains, and Sweep's `_stop_reader()`
    never runs. First field starts the reader; leftover is
    the same Queue after that, and a later `command()`
    lands on *that* Queue without being eaten.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        leftover = eng._cmd
        assert leftover.empty()
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng._cmd is leftover
        eng.command("snapshot")
        eng.command("lock", freq_hz=1280e6)
        eng._do_lock()
        assert eng._cmd is leftover
        assert leftover.empty() is False
        name, kw = leftover.get_nowait()
        assert name == "snapshot"
        assert kw == {}
        name, kw = leftover.get_nowait()
        assert name == "lock"
        assert kw["freq_hz"] == 1280e6
        assert leftover.empty()
        assert eng._snap is False
        assert eng.state.lock_target == 5800e6
        assert eng.state.lock_target != 1280e6
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        assert eng.state.mode == "LOCK"
    finally:
        eng._stop_reader()
