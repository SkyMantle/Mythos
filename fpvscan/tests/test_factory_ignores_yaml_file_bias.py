"""factory must not forward bias-tee YAML keys into FileSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_file_bias_tee_yaml(monkeypatch):
    """#40 factory omits `bias_tee` for BladeRF (Engine applies it
    after the 15 dB LNA offset). #83 ignores USB stream keys for
    file. #87 ignores sim scene keys for file. #79 ignores file
    `sample_rate` / `center_hz` / `gain_db`. The complementary
    leftover is a `driver: file` block that *names* bias-tee keys
    operators copy from a bladeRF stanza — `bias_tee` /
    `bias_tee_gain_offset_db`. Factory is `FileSource(path,
    loop=...)`. A leftover `FileSource(...,
    bias_tee=cfg.get("bias_tee"),
    bias_tee_gain_offset_db=cfg.get("bias_tee_gain_offset_db"))`
    TypeErrors on start (constructor has no such kwarg) and the
    capture never opens, so `confirm_hits=2` never lights the
    bird. Constructor must still see `path` / `loop` (file
    identity) and must not see bias-tee keys. `bias_tee: true`
    would also raise 4.5 V on a file handle if a Fake accepted it.
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
        "bias_tee": True,
        "bias_tee_gain_offset_db": 15,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is False
    assert seen["realtime"] is False
    assert "bias_tee" not in seen
    assert "bias_tee_gain_offset_db" not in seen
    assert seen.get("bias_tee") is not True
    assert seen.get("bias_tee_gain_offset_db") != 15
