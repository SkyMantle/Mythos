"""Engine.stop must flip the flag before joining the worker."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_stop_sets_flag_before_join():
    """`run.py` / headless `finally: engine.stop()`. Joining first
    waits for a thread that is still in `_do_lock` / USB retry
    sleep, because the loop only exits when `_stop` is set. A
    no-timeout join there hangs the process on a stuck `sync_rx`
    and leaves the exclusive handle held after Ctrl-C.
    """
    src = inspect.getsource(Engine.stop)
    assert src.index("self._stop.set()") < src.index(".join(")
    assert "timeout=3" in src
