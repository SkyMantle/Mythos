"""Snapshot photos must stay full-height WebP, not the stream defaults."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_save_photo_uses_full_field_webp():
    """Stream encode is height=None / quality=60. The photo path used to
    inherit those and wrote a 288-line muddy still the operator could
    not read. Lock the stretch-to-576 / quality-90 call.
    """
    src = inspect.getsource(Engine._save_photo)
    assert 'cvbs.encode(frame, "webp", 90, height=576)' in src
    assert 'stamped("shot", "webp"' in src
