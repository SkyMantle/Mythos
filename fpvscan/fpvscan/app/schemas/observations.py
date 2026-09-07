from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fpvscan.app.domain.models import Observation, ObservationSchema

_WRITE_RESERVED = frozenset({
    "idempotency_key",
    "timestamp_ms",
    "frequency_hz",
    "rssi",
    "snr",
    "video_metrics",
    "parameter_snapshot",
    "fields",
    "frame_ref",
})


def _blank_to_none(value: Any) -> Any:
    if value == "" or value is None:
        return None
    return value


class _ObservationPayloadMixin(BaseModel):
    """Fold leftover form keys into fields; treat empty optional scalars as omitted."""

    timestamp_ms: int | None = None
    frequency_hz: float | None = None
    rssi: float | None = None
    snr: float | None = None
    video_metrics: dict[str, Any] | None = None
    parameter_snapshot: dict[str, Any] | None = None
    frame_ref: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fold_loose_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fields = dict(data.get("fields") or {})
        kept: dict[str, Any] = {}
        for key, value in data.items():
            if key in _WRITE_RESERVED:
                kept[key] = value
            elif key not in fields:
                fields[key] = value
        kept["fields"] = fields
        return kept

    @field_validator("timestamp_ms", "frequency_hz", "rssi", "snr", "frame_ref", mode="before")
    @classmethod
    def empty_optional(cls, value: Any) -> Any:
        return _blank_to_none(value)


class SchemaFieldModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    type: Literal["number", "text", "bool", "enum", "rating", "tag_list"]
    required: bool = False
    enum_values: list[str] | None = None
    min: float | None = None
    max: float | None = None


class ObservationSchemaResponse(BaseModel):
    id: str
    fields: list[SchemaFieldModel]
    updated_at: str

    @classmethod
    def from_domain(cls, schema: ObservationSchema) -> ObservationSchemaResponse:
        return cls.model_validate(schema.to_dict())


class ObservationSchemaPutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: UUID
    fields: list[SchemaFieldModel]


class ObservationWriteRequest(_ObservationPayloadMixin):
    model_config = ConfigDict(extra="ignore")

    idempotency_key: UUID
    fields: dict[str, Any] = Field(default_factory=dict)


class ObservationPatchRequest(_ObservationPayloadMixin):
    model_config = ConfigDict(extra="ignore")

    fields: dict[str, Any] | None = None


class ObservationResponse(BaseModel):
    id: str
    session_id: str
    timestamp_ms: int
    frequency_hz: float | None = None
    rssi: float | None = None
    snr: float | None = None
    video_metrics: dict[str, Any] = Field(default_factory=dict)
    parameter_snapshot: dict[str, Any] = Field(default_factory=dict)
    fields: dict[str, Any] = Field(default_factory=dict)
    frame_ref: str | None = None
    engine_status: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, obs: Observation) -> ObservationResponse:
        return cls.model_validate(obs.to_dict())


class ObservationListResponse(BaseModel):
    items: list[ObservationResponse]

    @classmethod
    def from_domain(cls, items: list[Observation]) -> ObservationListResponse:
        return cls(items=[ObservationResponse.from_domain(o) for o in items])
