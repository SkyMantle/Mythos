"""factory must not forward the DSP LO-offset YAML key into BladeRF."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_bladerf_lo_offset_yaml(monkeypatch):
    """#17 / #32 LOCK applies `video.lo_offset_hz` in the
    channelizer (and clamps it past 0.45·fs). #21 maps
    `quick_tune` → `use_meta`. #40 omits `bias_tee` (Engine
    applies it after the 15 dB LNA offset). #81 ignores file /
    sim keys; #89 ignores `bias_tee_gain_offset_db`. The
    complementary leftover is a `driver: bladerf` block that
    *names* `lo_offset_hz` — shipped YAML has it under
    `video:` (0 at 35 Msps; 8e6 with 61.44). Factory is
    `BladeRF(device, lib_path, channel, gain_db, agc,
    num_buffers, buffer_size, num_transfers, settle_us,
    use_meta)`. BladeRF() has no `lo_offset_hz` kwarg. A
    leftover `BladeRF(..., lo_offset_hz=cfg.get("lo_offset_hz"))`
    TypeErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. Constructor must
    still see `gain_db` (USB identity) and must not see the
    offset. Engine._do_lock mixes it digitally, not the factory.
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
        "lo_offset_hz": 8e6,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert seen["gain_db"] == 35.0
    assert seen["use_meta"] is False
    assert "lo_offset_hz" not in seen
    assert seen.get("lo_offset_hz") != 8e6
