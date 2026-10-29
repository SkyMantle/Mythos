"""factory must not forward YAML `sample_rate` into FileSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_file_sample_rate(monkeypatch):
    """#66 FileSource `set_sample_rate` is a no-op (sidecar 40 Msps
    stays). #17 `_run` copies `src.sample_rate` after open.
    #77 factory ignores explicit `realtime: true`; #74 explicit
    `loop: false` is forwarded. The complementary leftover is a
    file: block that *names* `sample_rate: 35e6` (the shipped
    scan rate) next to `path`. Factory only passes `path` and
    `loop` (`FileSource(cfg["path"], loop=bool(cfg.get("loop",
    True)))`). A leftover `sample_rate=cfg.get("sample_rate")`
    either TypeErrors on start (constructor has no such kwarg)
    or, if honored, overrides the sidecar and mis-bins the
    16.2 kHz line so `confirm_hits=2` never lights the bird.
    Constructor must not see YAML rate / LO / gain.
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True, **kw):
            seen["path"] = path
            seen["loop"] = loop
            seen.update(kw)
            self.sample_rate = kw.get("sample_rate", None)

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    src = make_source({
        "driver": "file",
        "path": "caps/vtx.cf32",
        "loop": True,
        "sample_rate": 35e6,
        "center_hz": 1280e6,
        "gain_db": 60,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is True
    assert "sample_rate" not in seen
    assert "center_hz" not in seen
    assert "gain_db" not in seen
    assert src.sample_rate is None
    assert src.sample_rate != 35e6
