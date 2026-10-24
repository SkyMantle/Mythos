"""factory must keep USB timeout and analog-filter ratio when YAML omits them."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_timeout_ms_and_bandwidth_ratio(monkeypatch):
    """#33 pinned BladeRF() `timeout_ms=3500`; #19 analog BW is
    `0.9×fs`. factory.make_source does not forward those kwargs, so
    a partial `sdr:` block relies on the constructor. A leftover
    `timeout_ms=cfg.get("timeout_ms", 0)` makes every `sync_rx`
    ERR_TIMEOUT and `_config_stream` — LOCK looks dead on a live
    USB3 hiccup. A leftover `bandwidth_ratio=0` programs the RFIC
    LPF at 0 Hz and aliases the 16.2 kHz line into DC. Omitted
    keys must keep the constructor floors.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)
            self.timeout_ms = kw.get("timeout_ms", 3500)
            self.bandwidth_ratio = kw.get("bandwidth_ratio", 0.9)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({"driver": "bladerf"})
    assert src.name == "bladerf"
    assert src.timeout_ms == 3500
    assert src.bandwidth_ratio == 0.9
    assert seen.get("timeout_ms", 3500) == 3500
    assert seen.get("bandwidth_ratio", 0.9) == 0.9
    assert seen.get("timeout_ms", 1) != 0
    assert seen.get("bandwidth_ratio", 1) != 0.0
