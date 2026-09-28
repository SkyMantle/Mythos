"""LOCK WebP must keep the decoded field size, not the photo 576 stretch."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_lock_stream_encode_keeps_native_height():
    """cvbs.encode defaults height=480. Stretching a 232-line NTSC field
    to 480 on every LOCK frame letterboxes the console and burns Pi 5
    time that the idle_ms budget does not have. Photos already lock
    height=576; the stream path must keep height=None and the shipped
    stream_quality knob — not the encode() default of 75.
    """
    src = inspect.getsource(Engine._do_lock)
    assert 'cvbs.encode(frame, "webp", int(vcfg.get("stream_quality", 75)), height=None)' in src
