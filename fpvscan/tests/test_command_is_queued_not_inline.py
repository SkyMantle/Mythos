"""HTTP / console commands must not touch the radio on the caller thread."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine
from tests.helpers import MockSource, make_engine


def test_command_only_enqueues():
    """`/api/lock` and `/api/bias_tee` run on the uvicorn thread.
    Calling `_handle_command` there races the LOCK IQ reader
    (`sync_rx`) and the sweep `_grab` — the same exclusive
    handle bug as a bias-tee toggle mid-LOCK. `command()` must
    only `put` on `_cmd`; `_drain_commands` owns the USB.
    """
    src = inspect.getsource(Engine.command)
    assert "self._cmd.put((name, kw))" in src
    assert "_handle_command" not in src
    assert "_do_lock" not in src
    assert "set_bias_tee" not in src
    assert "set_center_freq" not in src


def test_command_does_not_retune_until_drained():
    radio = MockSource()
    eng = make_engine(src=radio)
    eng.command("lock", freq_hz=5800e6)
    assert radio.retune_calls == 0
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng._cmd.qsize() == 1
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert radio.retune_calls == 0  # lock only records the target
