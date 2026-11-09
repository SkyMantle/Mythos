"""factory must not forward the YAML driver selector into FileSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_file_driver_kwarg(monkeypatch):
    """#47 unknown `driver` raises `ValueError` before any
    constructor. #92 ignores `driver` for bladerf. #83
    ignores USB stream keys for file; #74 forwards
    `loop: false`; #77 ignores `realtime: true`. The
    complementary leftover is forwarding the selector
    itself into the replay constructor:
    `FileSource(cfg["path"], loop=..., driver=cfg.get("driver"))`.
    FileSource() has no `driver` kwarg (path / loop /
    realtime only). Operators uncomment `# driver: file`
    and `# path: caps/vtx.cf32` in shipped YAML. A leftover
    extra `driver=` TypeErrors on start and replay never
    opens, so `confirm_hits=2` never lights the bird from a
    capture. Constructor must still see `path` / `loop` and
    must not see `driver`. Engine never reads that kwarg —
    `make_source` already branched.
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
        "loop": True,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is True
    assert seen["realtime"] is False
    assert "driver" not in seen
    assert seen.get("driver") != "file"
    assert seen.get("path") != "file"
    assert seen.get("loop") != "file"
