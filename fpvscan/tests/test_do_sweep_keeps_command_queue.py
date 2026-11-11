"""A sweep hop must not replace the queue the web already put()s onto."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_command_queue():
    """#47 `Engine.command` only `put`s on `_cmd`. `create_app`
    lock / sweep / snapshot / record / bias-tee / clear *do*
    `engine.command(...)` onto that Queue. `_do_sweep` *does*
    `_drain_commands()` at each 8 MHz step — that is the
    hop's job in SWEEP, so a click-to-hold mid-walk aborts
    before `_grab`. #83 a later LOCK field keeps the
    contents (must not drain). #96 a hop keeps `_thread`
    so a second `_run` does not race `_grab`. The missing
    half: an 8 MHz hop must not `self._cmd = Queue()`
    (copy-paste from `__init__`). After the drain the
    queue is empty either way, so emptiness alone cannot
    tell a replacement from a live drain — identity is
    the hop analog. A leftover new Queue still looks like
    a live walk (`mode=SWEEP`, reader down,
    `retune_calls==1`) while the next Photo / click-to-hold
    / Сканувати `put`s onto a queue `_run` no longer
    drains, so the bird never lights and Stop cannot
    abort the walk. `fixed_freq` keeps this to one grab;
    zeros IQ so occupancy/inspect stay out of the path.
    Leftover is the same Queue after the hop, and a later
    `command()` lands on *that* Queue so the next hop
    still drains it.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    leftover = eng._cmd
    try:
        assert leftover.empty()
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng._cmd is leftover
        assert leftover.empty()
        eng.command("snapshot")
        assert leftover.empty() is False
        assert leftover.qsize() == 1
        eng._do_sweep()
        assert eng._cmd is leftover
        assert leftover.empty()
        assert eng._snap is True
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        assert eng._reader_thread is None
        assert eng._ring is None
        assert src.retune_calls == 2
    finally:
        eng._stop_reader()
