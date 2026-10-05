"""Public bias_tee must follow a successful lib call, not the request."""
from __future__ import annotations

import inspect

from fpvscan.sdr.bladerf import BladeRF


def test_set_bias_tee_assigns_flag_only_after_success():
    """`snapshot()['bias_tee']` is `getattr(src, 'bias_tee')` (#46).
    Writing `self.bias_tee = on` before `bladerf_set_bias_tee` makes
    the console button paint ON while the 4.5 V rail never switched —
    and the next click sends `off` to a tee that was already off.
    Failure already forces False; the success assignment must sit
    after the `rc < 0` return.
    """
    src = inspect.getsource(BladeRF.set_bias_tee)
    assert src.index("if rc < 0:") < src.index("self.bias_tee = bool(on)")
    assert src.index("self.bias_tee = bool(on)") < src.index("return True")
