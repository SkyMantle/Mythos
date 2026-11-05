"""factory must not forward the LNA offset YAML key into BladeRF."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_bladerf_bias_offset_yaml(monkeypatch):
    """#40 factory omits `bias_tee` for BladeRF (Engine applies it
    after the 15 dB LNA offset). #88 ignores `bias_tee` /
    `bias_tee_gain_offset_db` for file and sim. #73 ignores USB
    floors `timeout_ms` / `bandwidth_ratio`. The complementary
    leftover is a `driver: bladerf` block that *names*
    `bias_tee_gain_offset_db` — shipped YAML has
    `bias_tee_gain_offset_db: 15`. Factory is `BladeRF(device,
    lib_path, channel, gain_db, agc, num_buffers, buffer_size,
    num_transfers, settle_us, use_meta)`. BladeRF() accepts
    `bias_tee` but has no `bias_tee_gain_offset_db` kwarg. A
    leftover `BladeRF(...,
    bias_tee_gain_offset_db=cfg.get("bias_tee_gain_offset_db"))`
    TypeErrors on start (constructor has no such kwarg) and the
    board never opens, so `confirm_hits=2` never lights the bird.
    Constructor must still see `gain_db` (USB identity) and must
    not see the offset. Engine._run applies the 15 dB LNA
    compensation after `set_bias_tee`, not the factory.
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
        "bias_tee": True,
        "bias_tee_gain_offset_db": 15,
    })
    assert src.name == "bladerf"
    assert seen["gain_db"] == 35.0
    assert "bias_tee_gain_offset_db" not in seen
    assert seen.get("bias_tee_gain_offset_db") != 15
