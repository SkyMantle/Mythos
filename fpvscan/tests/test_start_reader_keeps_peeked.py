"""A LOCK hop must not forget auto-peek cooldown."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_peeked():
    """#64 Sweep / lock keep `_peeked`. #68 expiry / #70 peek
    keep other MHz so the bird we just watched is not re-peeked.
    #76 `_start_reader` already keeps leftover `detections` /
    `_snap`. Cooldown is the third piece of scan memory on the
    hop: it reseeds DecodeState after `retune_and_read` (#50)
    and must not `_peeked.clear()`. A leftover wipe on every
    channel change re-peeks the same analog hit the moment the
    operator lets go — leftover `_peeked.clear()` before writing
    the current key would glance at every other confirmed bird
    on the next confirm_hits cycle.
    """
    eng = make_engine()
    eng._peeked[5800] = 1.0
    eng._peeked[1280] = 2.0
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._peeked[5800] == 1.0
        assert eng._peeked[1280] == 2.0
        assert eng._peeked == {5800: 1.0, 1280: 2.0}
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
