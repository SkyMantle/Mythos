"""Validation for operator-configurable observation fields."""
from __future__ import annotations

import re
from typing import Any

from fpvscan.app.domain.exceptions import ValidationError
from fpvscan.app.domain.models import FIELD_TYPES

_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def normalize_schema_fields(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not fields:
        raise ValidationError("schema must contain at least one field")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in fields:
        field = _normalize_one(raw)
        if field["key"] in seen:
            raise ValidationError(
                f"duplicate field key: {field['key']}", {field["key"]: "duplicate"}
            )
        seen.add(field["key"])
        out.append(field)
    return out


def _normalize_one(raw: dict[str, Any]) -> dict[str, Any]:
    key = str(raw.get("key") or "").strip()
    if not _KEY.match(key):
        raise ValidationError(f"invalid field key: {key}", {key or "_": "key"})
    kind = str(raw.get("type") or "")
    if kind not in FIELD_TYPES:
        raise ValidationError(f"{key}: unknown type {kind}", {key: "type"})
    label = str(raw.get("label") or key).strip()
    required = bool(raw.get("required", False))
    enum_values = raw.get("enum_values")
    lo = raw.get("min")
    hi = raw.get("max")
    if kind == "enum":
        if not enum_values:
            raise ValidationError(f"{key}: enum_values required", {key: "enum_values"})
        enum_values = [str(v) for v in enum_values]
    else:
        enum_values = None
    if kind == "rating":
        lo = 1.0 if lo is None else float(lo)
        hi = 5.0 if hi is None else float(hi)
        if lo >= hi:
            raise ValidationError(f"{key}: min must be < max", {key: "min"})
    if kind == "number" and lo is not None and hi is not None and float(lo) > float(hi):
        raise ValidationError(f"{key}: min must be <= max", {key: "min"})
    return {
        "key": key,
        "label": label,
        "type": kind,
        "required": required,
        "enum_values": enum_values,
        "min": None if lo is None else float(lo),
        "max": None if hi is None else float(hi),
    }


def _is_blank(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    if isinstance(raw, list) and not raw:
        return True
    return False


def validate_field_values(
    schema_fields: list[dict[str, Any]],
    values: dict[str, Any],
    *,
    partial: bool = False,
) -> dict[str, Any]:
    index = {f["key"]: f for f in schema_fields}
    out: dict[str, Any] = {}
    for spec in schema_fields:
        key = spec["key"]
        if key not in values:
            if spec["required"] and not partial:
                raise ValidationError(f"{key} is required", {key: "required"})
            continue
        raw = values[key]
        if _is_blank(raw):
            if spec["required"] and not partial:
                raise ValidationError(f"{key} is required", {key: "required"})
            continue
        out[key] = _coerce_field(spec, raw)
    return out


def _coerce_field(spec: dict[str, Any], raw: Any) -> Any:
    key, kind = spec["key"], spec["type"]
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        raise ValidationError(f"{key}: expected bool", {key: "type"})
    if kind == "text":
        return str(raw)
    if kind == "enum":
        text = str(raw)
        allowed = spec.get("enum_values") or []
        if text not in allowed:
            raise ValidationError(f"{key}: not in {allowed}", {key: "enum"})
        return text
    if kind == "tag_list":
        if isinstance(raw, str):
            items = [p.strip() for p in raw.split(",") if p.strip()]
        elif isinstance(raw, list):
            items = [str(p).strip() for p in raw if str(p).strip()]
        else:
            raise ValidationError(f"{key}: expected tag list", {key: "type"})
        return items
    if kind in {"number", "rating"}:
        if isinstance(raw, bool):
            raise ValidationError(f"{key}: expected number", {key: "type"})
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{key}: expected number", {key: "type"}) from exc
        if kind == "rating":
            if value != int(value):
                raise ValidationError(f"{key}: rating must be an integer", {key: "type"})
            value = int(value)
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo:
            raise ValidationError(f"{key}: below min {lo}", {key: "min"})
        if hi is not None and value > hi:
            raise ValidationError(f"{key}: above max {hi}", {key: "max"})
        return value
    raise ValidationError(f"{key}: unsupported type", {key: "type"})
