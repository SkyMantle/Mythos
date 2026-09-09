"""CLI flags are how operators pick file/sim without editing yaml."""
from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import run as run_mod


def _base_cfg():
    return {
        "sdr": {"driver": "bladerf", "gain_db": 30, "lib_path": ""},
        "web": {"host": "127.0.0.1", "port": 8080},
        "scan": {},
        "video": {},
    }


def _run_main(argv, cfg):
    seen = {}

    class _Src:
        name = "mock"

    def fake_make(sdr):
        seen["sdr"] = dict(sdr)
        return _Src()

    fake_engine = MagicMock()
    fake_app = object()

    with patch.object(run_mod, "config") as cfg_mod, \
            patch.object(run_mod, "make_source", fake_make), \
            patch.object(run_mod, "Engine", return_value=fake_engine) as eng_cls, \
            patch.object(run_mod, "create_app", return_value=fake_app), \
            patch.object(run_mod, "_port_busy", return_value=False), \
            patch.object(run_mod.uvicorn, "run") as uv, \
            patch.object(run_mod.sys, "argv", ["run.py", *argv]):
        cfg_mod.load.return_value = copy.deepcopy(cfg)
        run_mod.main()
    seen["engine_started"] = fake_engine.start.called
    seen["uvicorn"] = uv.call_args
    seen["engine_cls"] = eng_cls
    return seen


def test_file_flag_selects_file_driver_and_path():
    seen = _run_main(["--file", "caps/vtx.cf32"], _base_cfg())
    assert seen["sdr"]["driver"] == "file"
    assert seen["sdr"]["path"] == "caps/vtx.cf32"
    assert seen["engine_started"]


def test_driver_flag_overrides_file_flag():
    """--file is applied first, then --driver. An explicit driver must win."""
    seen = _run_main(["--file", "caps/vtx.cf32", "--driver", "sim"], _base_cfg())
    assert seen["sdr"]["driver"] == "sim"
    assert seen["sdr"]["path"] == "caps/vtx.cf32"


def test_lib_and_port_flags_land_in_cfg():
    cfg = _base_cfg()
    seen = _run_main(["--lib", "/opt/libbladeRF.so", "--port", "9090"], cfg)
    assert seen["sdr"]["lib_path"] == "/opt/libbladeRF.so"
    host, port = seen["uvicorn"].kwargs["host"], seen["uvicorn"].kwargs["port"]
    assert port == 9090
    assert host == "127.0.0.1"


def test_busy_port_stops_engine_without_serving():
    class _Src:
        name = "mock"

    fake_engine = MagicMock()
    with patch.object(run_mod, "config") as cfg_mod, \
            patch.object(run_mod, "make_source", return_value=_Src()), \
            patch.object(run_mod, "Engine", return_value=fake_engine), \
            patch.object(run_mod, "create_app", return_value=object()), \
            patch.object(run_mod, "_port_busy", return_value=True), \
            patch.object(run_mod.uvicorn, "run") as uv, \
            patch.object(run_mod.sys, "argv", ["run.py"]):
        cfg_mod.load.return_value = _base_cfg()
        run_mod.main()
    assert fake_engine.start.called
    assert fake_engine.stop.called
    uv.assert_not_called()
