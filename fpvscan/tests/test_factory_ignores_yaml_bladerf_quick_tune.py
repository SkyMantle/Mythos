"""factory must map YAML quick_tune onto BladeRF.use_meta, not the YAML name."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_bladerf_quick_tune_kwarg(monkeypatch):
    """#21 maps `quick_tune` → `use_meta` and `rx_channel` →
    `channel`. #91 still passes `channel=` (not the YAML
    name). #92 ignores `driver`; #90 ignores `lo_offset_hz`;
    #89 ignores `bias_tee_gain_offset_db`; #40 omits
    `bias_tee`; #73 omitted `timeout_ms` / `bandwidth_ratio`
    stay constructor 3500 / 0.9. The complementary leftover
    is still passing the YAML name: `BladeRF(...,
    quick_tune=cfg.get("quick_tune"), use_meta=...)`.
    BladeRF() takes `use_meta`, not `quick_tune`. Shipped
    YAML always has `quick_tune: false`; operators flip it
    after `bench_retune.py`. A leftover extra `quick_tune=`
    TypeErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. Constructor
    must still see `use_meta` (USB identity) and must not
    see `quick_tune`. Engine never reads that kwarg —
    `_run` primes quick-tune from YAML, then the driver
    uses `use_meta` for SC16_Q11_META.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({
        "driver": "bladerf",
        "gain_db": 35,
        "rx_channel": 0,
        "quick_tune": True,
    })
    assert src.name == "bladerf"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["use_meta"] is True
    assert seen["use_meta"] is not False
    assert "quick_tune" not in seen
    assert seen.get("quick_tune") is not True
    assert seen.get("use_meta") != "quick_tune"
    assert seen.get("channel") != "quick_tune"
    assert "driver" not in seen
