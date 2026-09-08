"""Thin adapter around the scan engine. Does not touch DSP internals."""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Any

from fpvscan.app.domain.exceptions import ValidationError
from fpvscan.app.services.catalog import catalog_values

log = logging.getLogger("fpvscan.test.engine")


class EngineAdapter:
    def __init__(self, engine: Any, yaml_path: Path) -> None:
        self._engine = engine
        self._yaml_path = yaml_path
        self._defaults = copy.deepcopy(getattr(engine, "cfg", {}))
        self._lock = threading.Lock()
        try:
            self._yaml_text = yaml_path.read_text(encoding="utf-8")
        except OSError:
            self._yaml_text = ""

    def snapshot(self) -> dict[str, Any]:
        snap = self._engine.snapshot()
        return dict(snap) if isinstance(snap, dict) else {}

    def current_cfg(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._engine.cfg)

    def default_cfg(self) -> dict[str, Any]:
        return copy.deepcopy(self._defaults)

    def yaml_text(self) -> str:
        return self._yaml_text

    def current_param_values(self) -> dict[str, Any]:
        return catalog_values(self.current_cfg())

    def apply_values(self, values: dict[str, Any]) -> None:
        with self._lock:
            for key, value in values.items():
                if "." not in key:
                    raise ValidationError(f"invalid key: {key}", {key: "key"})
                section, name = key.split(".", 1)
                block = self._engine.cfg.get(section)
                if not isinstance(block, dict) or name not in block:
                    raise ValidationError(f"unknown parameter: {key}", {key: "unknown"})
                block[name] = value
            if (
                "sdr.gain_db" in values
                and values.get("sdr.auto_gain") is not True
            ):
                sdr = self._engine.cfg.get("sdr")
                if isinstance(sdr, dict):
                    sdr["auto_gain"] = False
        if any(k in values for k in ("sdr.gain_db", "sdr.bias_tee_gain_offset_db")):
            apply_bt = getattr(self._engine, "_apply_bias_tee_gain", None)
            try:
                if apply_bt:
                    apply_bt(bool(self._engine.cfg.get("sdr", {}).get("bias_tee", False)))
                elif hasattr(self._engine.src, "set_gain"):
                    self._engine.src.set_gain(float(
                        self._engine.cfg.get("sdr", {}).get("gain_db", 0)
                    ))
            except Exception as exc:
                log.warning("gain apply failed: %s", exc)
            if "sdr.gain_db" in values and values.get("sdr.auto_gain") is not True:
                disable = getattr(self._engine, "disable_auto_mgc", None)
                if disable:
                    disable()
        if values.get("sdr.auto_gain") is True:
            reset = getattr(self._engine, "reset_auto_mgc", None)
            if reset:
                reset()
        if "sdr.bias_tee" in values:
            self._engine.command("bias_tee", on=bool(values["sdr.bias_tee"]))
        if "scan.cluster_step_mhz" in values:
            reset = getattr(self._engine, "reset_sweep_plan", None)
            if reset:
                reset()

    def refresh_lock(self) -> None:
        refresh = getattr(self._engine, "refresh_lock", None)
        if refresh:
            refresh()
