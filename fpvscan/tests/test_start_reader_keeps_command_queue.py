"""A LOCK hop must not eat a Photo / Sweep / lock queued during the hold."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_command_queue():
    """#47 `Engine.command` only `put`s on `_cmd` (uvicorn must
    not touch USB). `_do_sweep` *does* `_drain_commands` at the
    start of each 8 MHz step — that is the hop's job in SWEEP.
    `_start_reader` is the missing half: AFC recenter and the
    1 Hz fs restart (#2) retune on the same bird. A leftover
    `_drain_commands()` there eats a Photo armed mid-hold, a
    Сканувати that should resume the 400 MHz–6 GHz walk, or a
    click-to-hold on a neighbor. The hop reseeds DecodeState
    (#50) and must not read the command queue.
    """
    eng = make_engine()
    eng.command("snapshot")
    eng.command("lock", freq_hz=1280e6)
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert not eng._cmd.empty()
        name, kw = eng._cmd.get_nowait()
        assert name == "snapshot"
        assert kw == {}
        name, kw = eng._cmd.get_nowait()
        assert name == "lock"
        assert kw["freq_hz"] == 1280e6
        assert eng._cmd.empty()
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
