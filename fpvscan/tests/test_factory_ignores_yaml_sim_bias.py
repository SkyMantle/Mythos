"""factory must not forward bias-tee YAML keys into SimSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_sim_bias_tee_yaml(monkeypatch):
    """#40 factory omits `bias_tee` for BladeRF. #85 ignores USB
    stream keys for sim. #80 ignores sim `sample_rate` /
    `center_hz` / `gain_db` / `path` / `loop` / `realtime` /
    `emitters` / `noise_db` / `seed`. #87 ignores those scene
    keys for file. The complementary leftover is a `driver: sim`
    block that *names* bias-tee keys operators copy from a
    bladeRF stanza — `bias_tee` / `bias_tee_gain_offset_db`.
    Factory is `SimSource()` with no kwargs. A leftover
    `SimSource(bias_tee=cfg.get("bias_tee"),
    bias_tee_gain_offset_db=...)` TypeErrors on start
    (constructor has no such kwarg) and the default scene never
    opens, so `confirm_hits=2` never lights the bird.
    Constructor must not see bias-tee keys. Defaults for
    `emitters` / `noise_db` / `seed` stay the Fake's, not YAML.
    """
    seen: dict = {}

    class Fake:
        name = "sim"

        def __init__(self, emitters=None, noise_db=-70.0, seed=0, **kw):
            seen["emitters"] = emitters
            seen["noise_db"] = noise_db
            seen["seed"] = seed
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.sim.SimSource", Fake)
    src = make_source({
        "driver": "sim",
        "bias_tee": True,
        "bias_tee_gain_offset_db": 15,
    })
    assert src.name == "sim"
    assert seen["emitters"] is None
    assert seen["noise_db"] == -70.0
    assert seen["seed"] == 0
    assert "bias_tee" not in seen
    assert "bias_tee_gain_offset_db" not in seen
    assert seen.get("bias_tee") is not True
    assert seen.get("bias_tee_gain_offset_db") != 15
