"""A LOCK field must not forget auto-peek cooldown."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_peeked():
    """#64 Sweep / lock keep `_peeked`. #68 expiry / #70 peek
    keep other MHz so the bird we just watched is not
    re-peeked. #77 `_start_reader` keeps the table on a
    channel change. `_do_lock` is the missing half: a field
    on the same bird must not `_peeked.clear()`. A leftover
    wipe on every decode forgets cooldown the moment the
    8 s glance expires — leftover `_peeked.clear()` before
    `_do_sweep` resumes would glance at the same analog hit
    instead of waiting `auto_peek_cooldown_s`, and the
    400 MHz–6 GHz walk never moves on.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._peeked[5800] = 1.0
    eng._peeked[1280] = 2.0
    try:
        eng._do_lock()
        assert eng._peeked[5800] == 1.0
        assert eng._peeked[1280] == 2.0
        assert eng._peeked == {5800: 1.0, 1280: 2.0}
        assert 5800 in eng._peeked
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
