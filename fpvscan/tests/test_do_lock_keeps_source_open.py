"""A LOCK field must not close the radio while the IQ reader is up."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_source_open_on_later_field():
    """#19 `_run` *does* `src.close()` in `finally` after
    `_stop_reader` / `_rec_stop`. Peek expiry / Sweep command
    *do* `_stop_reader()` but leave the handle open so the
    next `_grab` owns it. The missing half: a later field on
    the same bird must not `src.close()`. A leftover close
    (copy-paste from `_run` finally) zeros the USB handle
    while `_reader_loop` is still in `src.read`: the reader
    stores `_reader_err`, the next field consults it *before*
    `_start_reader`, and LOCK dies even though `_ring` /
    `_lock_tuned` still look live. First field starts the
    reader on an already-open source (`_do_lock` does not
    `open()`); leftover is `opened is True` after that.
    """
    src = MockSource()
    src.open()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        assert src.opened is True
        eng._do_lock()
        leftover = src.opened
        assert leftover is True
        assert leftover is not False
        assert eng._reader_thread is not None
        eng._do_lock()
        assert src.opened is leftover
        assert src.opened is True
        assert src.opened is not False
        assert eng._reader_thread is not None
        assert eng._reader_thread.is_alive()
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
