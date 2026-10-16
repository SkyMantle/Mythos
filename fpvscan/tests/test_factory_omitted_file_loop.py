"""Omitted file loop must wrap the capture, not EOF after one pass."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_file_loop_is_true(monkeypatch):
    """#21 / #43 forwarded an explicit `loop: True` and refused to
    pass `realtime`. Factory's fallback is `loop=bool(cfg.get("loop",
    True))`. A leftover False raises EOFError once inspect/sweep
    walks past the .cf32 — a 3 s record.py dump dies mid-scan and
    looks like a dead radio. YAML has no loop key; a partial file:
    block must keep wrapping.
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True):
            seen["path"] = path
            seen["loop"] = loop

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    cfg = {"driver": "file", "path": "caps/vtx.cf32"}
    assert "loop" not in cfg
    src = make_source(cfg)
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is True
