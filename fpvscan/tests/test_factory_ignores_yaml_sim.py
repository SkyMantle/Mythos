"""factory must not forward YAML rate / path / emitters into SimSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_sim_yaml(monkeypatch):
    """#79 factory ignores explicit file `sample_rate` / `center_hz`
    / `gain_db` (sidecar stays). #77 ignores file `realtime: true`;
    #74 forwards file `loop: false`. The complementary leftover
    is a sim: block that *names* `sample_rate: 35e6` (the shipped
    scan rate) next to `path` / `loop` / `emitters` / `noise_db`
    / `seed`. Factory is `SimSource()` with no kwargs. A leftover
    `SimSource(sample_rate=cfg.get("sample_rate"))` TypeErrors on
    start (constructor has no such kwarg) or, if honored, overrides
    the 40 Msps default and mis-bins the 16.2 kHz line so
    `confirm_hits=2` never lights the bird. Honoring YAML
    `emitters: []` / `noise_db: -20` makes a dead band; honoring
    `seed: 99` changes the noise realization vs the default scene.
    Constructor must not see YAML rate / LO / gain / path / loop
    / realtime / emitters / noise / seed.
    """
    seen: dict = {}

    class Fake:
        name = "sim"

        def __init__(self, emitters=None, noise_db=-70.0, seed=0, **kw):
            seen["emitters"] = emitters
            seen["noise_db"] = noise_db
            seen["seed"] = seed
            seen.update(kw)
            self.sample_rate = kw.get("sample_rate", None)

    monkeypatch.setattr("fpvscan.sdr.sim.SimSource", Fake)
    src = make_source({
        "driver": "sim",
        "sample_rate": 35e6,
        "center_hz": 1280e6,
        "gain_db": 60,
        "path": "caps/vtx.cf32",
        "loop": False,
        "realtime": True,
        "emitters": [],
        "noise_db": -20,
        "seed": 99,
    })
    assert src.name == "sim"
    assert seen["emitters"] is None
    assert seen["noise_db"] == -70.0
    assert seen["seed"] == 0
    assert "sample_rate" not in seen
    assert "center_hz" not in seen
    assert "gain_db" not in seen
    assert "path" not in seen
    assert "loop" not in seen
    assert "realtime" not in seen
    assert seen["emitters"] != []
    assert seen["noise_db"] != -20
    assert seen["seed"] != 99
    assert src.sample_rate is None
    assert src.sample_rate != 35e6
