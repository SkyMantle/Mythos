"""Startup must apply receiver gain before turning on the 4.5 V rail."""
from __future__ import annotations

import inspect

from fpvscan.engine import Engine


def test_run_sets_gain_before_enabling_bias_tee():
    """#9 checked the final offset (30−15). It did not check order.
    `set_bias_tee(True)` at the YAML `gain_db` (35 in the shipped
    file) plus LNA is the ADC-clip window #40 warned about on
    `BladeRF.open`. Engine `_run` is the path that actually runs
    after factory (which still omits `bias_tee`). Gain first, tee
    second, then `_apply_bias_tee_gain` — not tee at full gain with
    no offset scheduled.
    """
    src = inspect.getsource(Engine._run)
    head = src.split("while not self._stop.is_set()")[0]
    assert head.index("self.src.set_gain(") < head.index("self.src.set_bias_tee(")
    assert head.index("self.src.set_bias_tee(") < head.index(
        "self._apply_bias_tee_gain(")
