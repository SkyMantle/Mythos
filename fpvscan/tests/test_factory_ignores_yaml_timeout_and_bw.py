"""factory must not forward a leftover 0 USB timeout or analog-filter ratio."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_timeout_ms_and_bandwidth_ratio(monkeypatch):
    """#33 pinned BladeRF() `timeout_ms=3500`; #19 analog BW is
    `0.9×fs`. #72 omitted keys keep those constructor floors
    because factory does not pass the kwargs. The complementary
    leftover is a partial `sdr:` block that *names* them: a
    helpful `timeout_ms=cfg.get("timeout_ms", 3500)` still
    honors an explicit `0` (YAML null/placeholder) and every
    `sync_rx` ERR_TIMEOUT-loops. A forwarded `bandwidth_ratio: 0`
    programs the RFIC LPF at 0 Hz and aliases the 16.2 kHz line
    into DC. Factory must not forward either key — constructor
    floors win even when YAML lies.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)
            self.timeout_ms = kw.get("timeout_ms", 3500)
            self.bandwidth_ratio = kw.get("bandwidth_ratio", 0.9)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({
        "driver": "bladerf",
        "timeout_ms": 0,
        "bandwidth_ratio": 0.0,
    })
    assert src.name == "bladerf"
    assert "timeout_ms" not in seen
    assert "bandwidth_ratio" not in seen
    assert src.timeout_ms == 3500
    assert src.bandwidth_ratio == 0.9
    assert src.timeout_ms != 0
    assert src.bandwidth_ratio != 0.0
