"""A second lock must retune; stale `_lock_tuned` keeps the old VTx picture."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


def test_lock_command_clears_tuned_so_channel_switch_retunes():
    """LOCK already sitting on F4, operator clicks A8. The command
    must null `_lock_tuned` (and AFC / motion average). Otherwise
    `_do_lock` sees a live ring at the old LO and the console shows
    the new MHz on the previous channel's video.
    """
    src = MockSource(fs=20e6)
    eng = make_engine(src=src, capture_ms=8, idle_ms=0, sample_rate=20e6)
    eng.cfg["video"]["sample_rate"] = 20e6
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._lock_tuned = 5800e6
    eng._afc = 1.2e6
    eng._acc = np.zeros((8, 8), dtype=np.float32)

    eng.command("lock", freq_hz=5725e6)
    eng._drain_commands()

    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5725e6
    assert eng._lock_tuned is None
    assert eng._afc == 0.0
    assert eng._acc is None

    try:
        eng._do_lock()
        assert src.retune_calls >= 1
        assert src.center_freq == 5725e6
        assert eng._lock_tuned == 5725e6
    finally:
        eng._stop_reader()
