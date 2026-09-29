"""factory must not pace file replay at wall-clock n/fs."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_does_not_forward_file_realtime(monkeypatch):
    """FileSource.realtime=True sleeps n/fs per read. A 25 ms inspect
    grab at 40 Msps is ~1 s of wall clock per candidate, and a
    400 MHz–6 GHz file sweep looks hung. YAML has no realtime key
    today; forwarding a future/typo `realtime: true` would turn the
    Windows debug path into a wall-clock wait. factory only passes
    path + loop — FileSource's default realtime=False stays.
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
        "path": "caps/vtx_f4.cf32",
        "loop": True,
        "realtime": True,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx_f4.cf32"
    assert seen["loop"] is True
    assert "realtime" not in seen
