"""LOCK decode must not write back into the USB reader's ring."""
from __future__ import annotations

import numpy as np

from fpvscan.iqbuffer import IQRingBuffer


def test_snapshot_is_a_copy_not_a_view():
    """snapshot() documents «суцільним масивом (копія)». _do_lock does
    `iq = iq - mean` and then channelize/AFC. If snapshot ever returns
    a view into `_buf`, those writes race the reader thread and the
    next field inherits a DC hole / rotated phase — tracking looks
    like a dead VTx.
    """
    buf = IQRingBuffer(16)
    buf.write(np.arange(16, dtype=np.complex64))
    out, abs_start = buf.snapshot(16)
    assert abs_start == 0
    out[:] = 99
    again, _ = buf.snapshot(16)
    np.testing.assert_array_equal(again.real.astype(int), np.arange(16))
    assert again[0] != 99
