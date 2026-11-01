"""A LOCK field must not eat a Photo / Sweep / lock queued mid-hold."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_command_queue():
    """#47 `Engine.command` only `put`s on `_cmd`. #80
    `_start_reader` keeps the queue across AFC recenter /
    the 1 Hz fs restart. `_do_sweep` *does* drain at each
    8 MHz step — that is the hop's job in SWEEP. `_do_lock`
    is the missing half: a field decode + idle_ms runs on
    the same bird without going through the hop. A leftover
    `_drain_commands()` there eats a Photo armed mid-hold,
    a Сканувати that should resume the 400 MHz–6 GHz walk,
    or a click-to-hold on a neighbor — and Sweep's
    `_stop_reader()` would null `_ring` while this field
    still snapshots it. The field must not read the queue.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.command("snapshot")
    eng.command("lock", freq_hz=1280e6)
    try:
        eng._do_lock()
        assert not eng._cmd.empty()
        name, kw = eng._cmd.get_nowait()
        assert name == "snapshot"
        assert kw == {}
        name, kw = eng._cmd.get_nowait()
        assert name == "lock"
        assert kw["freq_hz"] == 1280e6
        assert eng._cmd.empty()
        assert eng._snap is False
        assert eng.state.lock_target == 5800e6
        assert eng.state.lock_target != 1280e6
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
