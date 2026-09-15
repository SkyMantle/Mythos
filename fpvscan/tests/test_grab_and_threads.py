"""Sweep grab and worker threads — USB races and hung Ctrl+C."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine
from tests.helpers import MockSource, make_engine


def test_grab_without_fast_path_uses_settled_retune():
    """File replay and the sim have no `retune_and_read_fast`. Sweep
    must fall back to `retune_and_read` (PLL dump included). Calling
    the fast name unconditionally AttributeErrors and the whole
    `_run` USB-retry loop lights up on a recording.
    """
    src = MockSource()
    assert not hasattr(src, "retune_and_read_fast")
    eng = make_engine(src=src)
    before = src.retune_calls
    iq = eng._grab(5800e6, 16)
    assert src.retune_calls == before + 1
    assert len(iq) == 16
    assert src.center_freq == 5800e6


def test_engine_and_reader_threads_are_daemons():
    """Non-daemon worker threads keep the process alive after uvicorn
    catches Ctrl+C, so the next `run.py` finds the port busy and
    exits without serving. Both the engine loop and the LOCK reader
    must be daemons.
    """
    start = inspect.getsource(Engine.start)
    reader = inspect.getsource(Engine._start_reader)
    assert "daemon=True" in start
    assert "daemon=True" in reader
