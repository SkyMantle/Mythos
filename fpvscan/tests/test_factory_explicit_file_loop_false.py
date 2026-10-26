"""Explicit YAML `loop: false` must reach FileSource, not the omitted True."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_forwards_explicit_file_loop_false(monkeypatch):
    """#63 omitted `loop` is constructor True so a short .cf32
    wraps instead of EOFError mid-inspect. #43 / #72 sidecar
    `loop` is ignored on `open()` — constructor wins. The
    complementary leftover is a one-shot bringup that *names*
    `loop: false` in the file: block. Dropping the key, or
    forcing True, splices tail onto head and `confirm_hits=2`
    lights the same bird twice. Factory must pass the explicit
    False through (`loop=bool(cfg.get("loop", True))`).
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True):
            seen["path"] = path
            seen["loop"] = loop

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    src = make_source({
        "driver": "file",
        "path": "caps/oneshot.cf32",
        "loop": False,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/oneshot.cf32"
    assert seen["loop"] is False
    assert seen["loop"] is not True
