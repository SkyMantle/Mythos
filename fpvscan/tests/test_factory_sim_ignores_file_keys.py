"""Switching YAML to sim must not require a leftover file path."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_sim_is_constructed_with_no_yaml_kwargs(monkeypatch):
    """Operators comment in `driver: file` / `path:` to replay a
    capture, then flip `driver` back to `sim`. Forwarding the
    leftover `path`/`loop` into SimSource() TypeErrors on start
    and looks like a broken install. factory calls SimSource()
    with no args — the default scene stays.
    """
    seen: dict = {}

    class Fake:
        name = "sim"

        def __init__(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr("fpvscan.sdr.sim.SimSource", Fake)
    src = make_source({
        "driver": "sim",
        "path": "caps/vtx_f4.cf32",
        "loop": True,
        "gain_db": 35,
    })
    assert src.name == "sim"
    assert seen["args"] == ()
    assert seen["kwargs"] == {}
