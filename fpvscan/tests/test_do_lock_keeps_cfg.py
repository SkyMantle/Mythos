"""A LOCK field must not replace the YAML dict `_run` already mutated."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_cfg_on_later_field():
    """`_run` *does* write the board's actual sample rate into
    `self.cfg["scan"]["sample_rate"]` and
    `self.cfg["video"]["sample_rate"]` before the loop, then
    holds `scan = self.cfg["scan"]` locally. #92 a later field
    keeps `_stop` / `events` so the worker and the web pump
    still talk to the same objects. The missing half: a later
    field on the same bird must not `self.cfg = {}` /
    `self.cfg = dict(self.cfg)` / `self.cfg = self.cfg["video"]`
    (copy-paste from `__init__` or a "fresh YAML view").
    `_do_lock` reads `self.cfg["video"]` every field; the 1 Hz
    fs-tolerance restart compares that rate to `src.sample_rate`.
    A leftover replacement still looks like a live hold
    (`mode=LOCK`, `lock_target`, `_ring`, `_lock_tuned`,
    reader alive, `retune_calls==1` on the *first* field, the
    0.4 Hz fs-tolerance case still passes if YAML equals the
    mock) while the next field compares against 35e6 from a
    copied tree, restarts the IQ reader, and `snapshot()`
    confirm_hits / AFC knobs go back to whoever owns the old
    dict. First field starts the reader; leftover is the same
    top dict *and* the same nested video/scan dicts after that.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        leftover = eng.cfg
        leftover_video = eng.cfg["video"]
        leftover_scan = eng.cfg["scan"]
        assert leftover_video["sample_rate"] == 20e6
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng.cfg is leftover
        eng._do_lock()
        assert eng.cfg is leftover
        assert eng.cfg["video"] is leftover_video
        assert eng.cfg["scan"] is leftover_scan
        assert leftover_video["sample_rate"] == 20e6
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        assert eng.state.mode == "LOCK"
    finally:
        eng._stop_reader()
