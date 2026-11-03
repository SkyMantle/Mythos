"""factory must not forward sim scene YAML keys into FileSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_file_sim_yaml(monkeypatch):
    """#80 factory ignores sim `sample_rate` / `center_hz` /
    `gain_db` / `path` / `loop` / `realtime` / `emitters` /
    `noise_db` / `seed` for SimSource. #81 ignores those keys
    for bladerf. #79 ignores file `sample_rate` / `center_hz` /
    `gain_db`. #77 ignores file `realtime`. The complementary
    leftover is a `driver: file` block that *names* sim scene
    keys operators copy from a SimSource stanza — `emitters` /
    `noise_db` / `seed`. Factory is `FileSource(path, loop=...)`.
    A leftover `FileSource(..., emitters=cfg.get("emitters"),
    noise_db=..., seed=...)` TypeErrors on start (constructor
    has no such kwarg) and the capture never opens, so
    `confirm_hits=2` never lights the bird. Constructor must
    still see `path` / `loop` (file identity) and must not see
    scene keys. Empty `emitters: []` would also honor a dead
    band if it were forwarded onto a Fake that accepted it.
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True, realtime=False, **kw):
            seen["path"] = path
            seen["loop"] = loop
            seen["realtime"] = realtime
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    src = make_source({
        "driver": "file",
        "path": "caps/vtx.cf32",
        "loop": False,
        "emitters": [],
        "noise_db": -120.0,
        "seed": 99,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is False
    assert seen["realtime"] is False
    assert "emitters" not in seen
    assert "noise_db" not in seen
    assert "seed" not in seen
    assert seen.get("emitters") != []
    assert seen.get("noise_db") != -120.0
    assert seen.get("seed") != 99
