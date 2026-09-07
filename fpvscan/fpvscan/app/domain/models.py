"""Domain objects returned by repositories and used by services."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    group: str
    type: str
    unit: str | None
    min: float | None
    max: float | None
    step: float | None
    default: Any
    current: Any
    enum_values: list[str] | None
    description: str
    applies_to: list[str]
    modes: list[str] = field(default_factory=list)
    task: str | None = None
    affects: str | None = None
    enum_labels: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AppliedParams:
    values: dict[str, Any]
    applied_keys: list[str]
    pending_keys: list[str] = field(default_factory=list)
    pending_reasons: dict[str, str] = field(default_factory=dict)
    affects: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "applied_keys": self.applied_keys,
            "pending_keys": self.pending_keys,
            "pending_reasons": self.pending_reasons,
            "affects": self.affects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppliedParams:
        return cls(
            values=dict(data["values"]),
            applied_keys=list(data["applied_keys"]),
            pending_keys=list(data.get("pending_keys") or []),
            pending_reasons=dict(data.get("pending_reasons") or {}),
            affects=dict(data.get("affects") or {}),
        )


@dataclass
class Session:
    id: str
    title: str
    mode: str
    status: str
    notes: str
    schema_id: str
    schema_snapshot: list[dict[str, Any]]
    parameter_snapshot: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or ""),
            mode=str(data["mode"]),
            status=str(data["status"]),
            notes=str(data.get("notes") or ""),
            schema_id=str(data["schema_id"]),
            schema_snapshot=list(data.get("schema_snapshot") or []),
            parameter_snapshot=dict(data.get("parameter_snapshot") or {}),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


@dataclass
class ObservationSchema:
    id: str
    fields: list[dict[str, Any]]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationSchema:
        return cls(
            id=str(data["id"]),
            fields=list(data.get("fields") or []),
            updated_at=str(data["updated_at"]),
        )


@dataclass
class Observation:
    id: str
    session_id: str
    timestamp_ms: int
    frequency_hz: float | None
    rssi: float | None
    snr: float | None
    video_metrics: dict[str, Any]
    parameter_snapshot: dict[str, Any]
    fields: dict[str, Any]
    frame_ref: str | None
    engine_status: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            timestamp_ms=int(data["timestamp_ms"]),
            frequency_hz=data.get("frequency_hz"),
            rssi=data.get("rssi"),
            snr=data.get("snr"),
            video_metrics=dict(data.get("video_metrics") or {}),
            parameter_snapshot=dict(data.get("parameter_snapshot") or {}),
            fields=dict(data.get("fields") or {}),
            frame_ref=data.get("frame_ref"),
            engine_status=dict(data.get("engine_status") or {}),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


DEFAULT_SCHEMA_FIELDS: list[dict[str, Any]] = [
    {
        "key": "signal_quality",
        "label": "Signal quality",
        "type": "rating",
        "required": False,
        "min": 1,
        "max": 5,
    },
    {
        "key": "picture_lock",
        "label": "Picture lock",
        "type": "bool",
        "required": False,
    },
    {
        "key": "picture_jump",
        "label": "Picture jump",
        "type": "bool",
        "required": False,
    },
    {
        "key": "line_tearing",
        "label": "Line tearing",
        "type": "bool",
        "required": False,
    },
    {
        "key": "spectrum_stutter",
        "label": "Spectrum stutter",
        "type": "bool",
        "required": False,
    },
    {
        "key": "analog_noise",
        "label": "Analog noise",
        "type": "rating",
        "required": False,
        "min": 1,
        "max": 5,
    },
    {
        "key": "osd_present",
        "label": "OSD present",
        "type": "bool",
        "required": False,
    },
    {
        "key": "frequency_mhz",
        "label": "Frequency",
        "type": "number",
        "required": False,
        "min": 100,
        "max": 6200,
    },
    {
        "key": "bandwidth_note",
        "label": "Bandwidth note",
        "type": "text",
        "required": False,
    },
    {
        "key": "aircraft_type",
        "label": "Aircraft type",
        "type": "text",
        "required": False,
    },
    {
        "key": "notes",
        "label": "Notes",
        "type": "text",
        "required": False,
    },
    {
        "key": "tags",
        "label": "Tags",
        "type": "tag_list",
        "required": False,
    },
]


SESSION_MODES = frozenset({"sweep", "lock", "manual"})
SESSION_STATUSES = frozenset({"active", "paused", "completed"})
FIELD_TYPES = frozenset({"number", "text", "bool", "enum", "rating", "tag_list"})
PARAM_TYPES = frozenset({"float", "int", "bool", "enum", "string"})
