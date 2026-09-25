"""Photo/rec filenames round to the nearest integer megahertz."""
from __future__ import annotations

import re

from fpvscan import paths


def test_stamped_rounds_half_mhz_to_the_named_channel():
    """`.0f` is nearest-MHz, not `int()` truncate. 5799.6 MHz is F4;
    a truncated `5799M` does not match the hit list or the sidecar
    the operator greps for after a session.
    """
    near = paths.stamped("rec", "mp4", 5799.6e6)
    over = paths.stamped("shot", "webp", 5800.4e6)
    exact = paths.stamped("rec", "mp4", 5732e6)
    assert re.fullmatch(r"rec_\d{8}-\d{6}_5800M\.mp4", near)
    assert re.fullmatch(r"shot_\d{8}-\d{6}_5800M\.webp", over)
    assert re.fullmatch(r"rec_\d{8}-\d{6}_5732M\.mp4", exact)
