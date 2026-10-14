"""Omitted bias_tee `on` must still enable the tee (console posts on/off)."""
from __future__ import annotations

from tests.helpers import make_engine


def test_omitted_bias_tee_on_defaults_true():
    """The console always posts `on` / `off`. `_handle_command` still
    defaults `on=True` when the payload is missing. A leftover False
    would silently refuse an operator 'увімкнути' if the key is
    dropped. Defaulting True without the LNA offset (already pinned)
    is the ADC-clip path; this only pins the verb.
    """
    eng = make_engine()
    assert eng.src.bias is None
    eng._handle_command("bias_tee", {})
    assert eng.src.bias is True
    # helpers pin offset 15, so ON subtracts it from gain_db=30
    assert eng.src.gain == 15.0
