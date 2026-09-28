"""Older libbladeRF must still load when quick-tune symbols are absent."""
from __future__ import annotations

import inspect

from fpvscan.sdr import bladerf


def test_declare_skips_missing_lib_symbols():
    """Pi images and some Windows Nuand builds ship a libbladeRF
    without bladerf_get_quick_tune / schedule_retune. Raising on
    AttributeError aborts load_lib() before open(), so doctor and
    bringup die with a ctypes traceback instead of running the
    ordinary settle-us retune path.
    """
    src = inspect.getsource(bladerf._declare)
    assert "except AttributeError:" in src
    block = src[src.index("except AttributeError:"):]
    assert block.splitlines()[1].strip() == "continue"
