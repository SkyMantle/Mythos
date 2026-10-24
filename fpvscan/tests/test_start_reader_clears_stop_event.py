"""A new LOCK reader must not inherit a leftover stop flag."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_disarms_leftover_stop_event():
    """#24 Sweep / peek-expiry `_stop_reader` set `_reader_stop` and
    leave it set. #50 `_start_reader` joins before `retune_and_read`
    and reseeds DecodeState after the hop. The Event itself is the
    third leftover: `_reader_loop` is `while not _reader_stop`.
    Reusing that Event without `clear()` (or a fresh Event — current
    main allocates one) makes the new thread exit before the first
    `read`. The ring then holds only the prime, `_do_lock` sees
    `len(iq) < n` and sleeps idle_ms — LOCK looks dead after
    Сканувати or the 8 s timer. `_reader_err` after Sweep is still
    the bugfix track (#50 start clears it; `_stop_reader` does not).
    """
    eng = make_engine()
    leftover = eng._reader_stop
    leftover.set()
    assert leftover.is_set()
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert not eng._reader_stop.is_set()
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()
