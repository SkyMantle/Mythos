"""Ctrl-C / a clean uvicorn exit must release the exclusive bladeRF handle."""
from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import run as run_mod


SRC = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")


def _base_cfg():
    return {
        "sdr": {"driver": "sim", "gain_db": 30, "lib_path": ""},
        "web": {"host": "127.0.0.1", "port": 8080},
        "scan": {},
        "video": {},
    }


def _run_with_uvicorn(side_effect=None):
    class _Src:
        name = "mock"

    fake_engine = MagicMock()
    with patch.object(run_mod, "config") as cfg_mod, \
            patch.object(run_mod, "make_source", return_value=_Src()), \
            patch.object(run_mod, "Engine", return_value=fake_engine), \
            patch.object(run_mod, "create_app", return_value=object()), \
            patch.object(run_mod, "_port_busy", return_value=False), \
            patch.object(run_mod.uvicorn, "run") as uv, \
            patch.object(run_mod.sys, "argv", ["run.py"]):
        cfg_mod.load.return_value = copy.deepcopy(_base_cfg())
        if side_effect is not None:
            uv.side_effect = side_effect
        try:
            run_mod.main()
            raised = None
        except KeyboardInterrupt as e:
            raised = e
        except Exception as e:
            raised = e
    return fake_engine, raised


def test_uvicorn_return_stops_the_engine():
    """The console process owns the radio. If uvicorn.run() returns
    (bind error after the busy-port check, or a clean shutdown) and
    `engine.stop()` is only on the busy-port path, the next
    doctor/bringup/run.py dies with 'device busy' until a reboot.
    """
    engine, raised = _run_with_uvicorn()
    assert raised is None
    assert engine.start.called
    assert engine.stop.called


def test_keyboard_interrupt_still_stops_the_engine():
    """Ctrl-C of the console used to unwind through uvicorn and skip
    Engine.stop() — same exclusive-handle leak as a forgotten
    headless `finally`. The call must sit in `finally` after
    uvicorn.run, not only on the happy path.
    """
    engine, raised = _run_with_uvicorn(side_effect=KeyboardInterrupt())
    assert isinstance(raised, KeyboardInterrupt)
    assert engine.stop.called
    finally_at = SRC.index("finally:")
    assert SRC.index("uvicorn.run(") < finally_at
    assert "engine.stop()" in SRC[finally_at:]
