"""Читання конфігу.

YAML 1.1 вимагає знак в експоненті: 400.0e+6 — число, а 400.0e6 —
рядок. Писати в конфізі "e+6" незручно і легко забути, тому числові
рядки приводимо до float одразу після читання. Інакше помилка
вилазить далеко від причини, у вигляді дивних порівнянь і ділень.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce(v):
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, str) and _NUM.match(v.strip()):
        return float(v)
    return v


def load(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _coerce(cfg)