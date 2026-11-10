"""factory must not forward the YAML driver selector into SimSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_sim_driver_kwarg(monkeypatch):
    """#47 unknown `driver` raises `ValueError` before any
    constructor. #92 ignores `driver` for bladerf; #93
    ignores it for file. #80 ignores sim scene / rate /
    path keys; #85 ignores USB stream keys; #88 ignores
    bias-tee. The complementary leftover is forwarding
    the selector itself into the simulator constructor:
    `SimSource(driver=cfg.get("driver"))`. SimSource()
    has no `driver` kwarg (emitters / noise_db / seed
    only) and factory is `SimSource()` with no kwargs.
    Operators pass `--driver sim` or set `driver: sim`
    in YAML while the rest of the bladeRF stanza stays.
    A leftover extra `driver=` TypeErrors on start and
    the default scene never opens, so `confirm_hits=2`
    never lights the bird. Constructor must still see
    the default emitters / noise / seed and must not
    see `driver`. Engine never reads that kwarg —
    `make_source` already branched.
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
        "gain_db": 35,
        "path": "caps/vtx.cf32",
    })
    assert src.name == "sim"
    assert seen["emitters"] is None
    assert seen["noise_db"] == -70.0
    assert seen["seed"] == 0
    assert "driver" not in seen
    assert seen.get("driver") != "sim"
    assert seen.get("emitters") != "sim"
    assert seen.get("noise_db") != "sim"
    assert seen.get("gain_db") != "sim"
    assert "gain_db" not in seen
    assert "path" not in seen
