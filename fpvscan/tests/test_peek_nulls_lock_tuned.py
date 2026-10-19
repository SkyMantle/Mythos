"""Auto-peek must null `_lock_tuned` so the IQ reader hops onto the bird."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_nulls_lock_tuned():
    """#21 lock() nulls `_lock_tuned` so a channel switch retunes.
    #63 Clear keeps it so Очистити does not join the reader.
    `_maybe_peek` is the third path onto LOCK: Sweep (#24) already
    `_stop_reader()`s (ring is None), but leftover `_lock_tuned`
    from the previous hold can still match `want` if the same MHz
    peeks again after a short join (#10) left the ring alive.
    `_do_lock` then skips `_start_reader` and decodes stale IQ from
    the last channel. Peek must force the hop, same as a click.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._lock_tuned = 5725e6
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
    assert eng._lock_tuned is None
