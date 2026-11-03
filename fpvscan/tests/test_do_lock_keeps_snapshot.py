"""A LOCK field must not drop a pending Photo when no picture exists."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_pending_photo():
    """#69 commands/expiry keep `_snap`; #70 peek / #76 `_start_reader`
    keep it on a channel change; #77 `_do_sweep` keeps it on an 8 MHz
    hop. `_do_lock` *does* consume `_snap` when a live frame exists
    (`_save_photo`). The missing half: a field on the same bird with
    zeros IQ (no picture) must not `_snap = False`. A leftover wipe
    on every decode drops a still armed from Photo / Сфотографувати
    before the next successful field — the operator clicks, the
    notice never fires, and the flight is one picture with no
    still. The field already starts the reader when `_ring` is None
    and must not write the pending flag.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._snap = True
    try:
        eng._do_lock()
        assert eng._snap is True
        assert eng._snap is not False
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
