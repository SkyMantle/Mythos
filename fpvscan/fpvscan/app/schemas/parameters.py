from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from fpvscan.app.domain.models import AppliedParams, ParamSpec


class ParameterSpecResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    group: str
    type: str
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    current: Any = None
    enum_values: list[str] | None = None
    enum_labels: dict[str, str] | None = None
    description: str = ""
    applies_to: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)
    task: str | None = None
    affects: str | None = None

    @classmethod
    def from_domain(cls, spec: ParamSpec) -> ParameterSpecResponse:
        return cls.model_validate(spec.to_dict())


class ParameterCatalogResponse(BaseModel):
    items: list[ParameterSpecResponse]

    @classmethod
    def from_domain(cls, specs: list[ParamSpec]) -> ParameterCatalogResponse:
        return cls(items=[ParameterSpecResponse.from_domain(s) for s in specs])


class CurrentParametersResponse(BaseModel):
    values: dict[str, Any]


class ApplyParametersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: UUID
    values: dict[str, Any] = Field(default_factory=dict)


class ApplyParametersResponse(BaseModel):
    values: dict[str, Any]
    applied_keys: list[str]
    pending_keys: list[str] = Field(default_factory=list)
    pending_reasons: dict[str, str] = Field(default_factory=dict)
    affects: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, result: AppliedParams) -> ApplyParametersResponse:
        return cls(
            values=result.values,
            applied_keys=result.applied_keys,
            pending_keys=result.pending_keys,
            pending_reasons=result.pending_reasons,
            affects=result.affects,
        )
