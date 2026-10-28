"""factory must not forward YAML `realtime: true` into FileSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_file_realtime(monkeypatch):
    """#63 omitted `loop` is constructor True; #74 explicit
    `loop: false` is forwarded. #72 sidecar `loop` / `realtime`
    are ignored on `open()` — constructor wins. The complementary
    leftover is a file: block that *names* `realtime: true` for a
    bench capture. Factory only passes `path` and `loop`
    (`FileSource(cfg["path"], loop=bool(cfg.get("loop", True)))`).
    A leftover `realtime=cfg.get("realtime", False)` still honors
    an explicit True and `read()` sleeps `n/fs` per inspect grab
    — a 25 ms classify at 40 Msps stalls the 400 MHz–6 GHz walk
    on wall-clock. Constructor `realtime=False` must win even
    when YAML asks to pace the .cf32.
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True, **kw):
            seen["path"] = path
            seen["loop"] = loop
            seen.update(kw)
            self.realtime = kw.get("realtime", False)

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    src = make_source({
        "driver": "file",
        "path": "caps/vtx.cf32",
        "loop": True,
        "realtime": True,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is True
    assert "realtime" not in seen
    assert src.realtime is False
    assert src.realtime is not True
