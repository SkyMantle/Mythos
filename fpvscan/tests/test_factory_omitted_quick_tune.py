"""Omitted quick_tune must open a plain SC16 stream, not metadata."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_quick_tune_is_use_meta_false(monkeypatch):
    """#21 / #57 forwarded an explicit YAML `quick_tune` into
    BladeRF.use_meta. #61 checked that `_run` does not prime when the
    Engine key is missing. Factory is the other half: `use_meta`
    selects FORMAT_SC16_Q11_META on sync_config. A leftover True
    puts timestamp headers on a plain RX stream — every LOCK/inspect
    read is garbage IQ and looks like a dead VTx. Shipped YAML is
    false; a partial sdr: block must not flip it on.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    cfg = {"driver": "bladerf"}
    assert "quick_tune" not in cfg
    src = make_source(cfg)
    assert src.name == "bladerf"
    assert seen["use_meta"] is False
