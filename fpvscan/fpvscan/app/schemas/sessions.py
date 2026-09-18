from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from fpvscan.app.domain.models import Session


class HealthResponse(BaseModel):
    status: str
    service: str
    engine_alive: bool | None = None
    sdr_state: str | None = None
    sdr_state_age_ms: int = 0
    sdr_operation: str | None = None
    sdr_last_operation: str | None = None
    sdr_consecutive_errors: int = 0
    sdr_open_attempts: int = 0
    sdr_open_failures: int = 0
    sdr_next_retry_ms: int = 0
    last_error: str | None = None
    last_error_code: int | None = None
    last_error_time: str | None = None
    last_error_operation: str | None = None
    last_action: str | None = None
    message: str | None = None


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: UUID
    title: str | None = None
    mode: Literal["sweep", "lock", "manual"]
    parameter_snapshot: dict[str, Any] | None = None
    schema_id: str | None = None


class SessionPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    notes: str | None = None
    status: Literal["active", "paused", "completed"] | None = None


class SessionResponse(BaseModel):
    id: str
    title: str
    mode: str
    status: str
    notes: str
    schema_id: str
    schema_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    parameter_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, session: Session) -> SessionResponse:
        return cls.model_validate(session.to_dict())


class SessionListResponse(BaseModel):
    items: list[SessionResponse]

    @classmethod
    def from_domain(cls, sessions: list[Session]) -> SessionListResponse:
        return cls(items=[SessionResponse.from_domain(s) for s in sessions])
