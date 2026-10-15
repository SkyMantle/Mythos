"""A non-wrapping snapshot is still stream order with abs_start = filled-n."""
from __future__ import annotations

import numpy as np

from fpvscan.iqbuffer import IQRingBuffer


def test_unwrapped_snapshot_is_the_tail_with_abs_start():
    """#52 covered the wrap concatenate and an oversized write. The
    common LOCK path is a ring that has not lapped yet: prime writes
    10 ms, then snapshot(capture_ms) takes the tail. Returning the
    head, or abs_start=0 when filled > n, makes cvbs.decode genlock
    from the wrong absolute index and the picture rolls every field
    until the ring wraps — then it tears for a different reason.
    """
    ring = IQRingBuffer(capacity=16)
    ring.write(np.arange(10, dtype=np.complex64))
    out, abs_start = ring.snapshot(4)
    np.testing.assert_array_equal(out.real, np.arange(6, 10, dtype=np.float32))
    assert abs_start == 6
    assert ring.filled == 10
    out[0] = 99
    again, _ = ring.snapshot(4)
    assert again[0].real == 6.0
